import os
import json
import datetime
import pandas as pd
import requests
import re
import cv2
import numpy as np
from urllib.request import urlopen
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Transformer
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import dateparser

# --- Imports for OCRFlux ---
from vllm import LLM
from ocrflux.inference import parse

# ================================
# --- Global Constants ---
# ================================

BASE_LOTS_FILE = 'hunting_lots.csv'
DATES_FILE = 'hunting_dates.csv'
TILES_DIR = 'tiles'
CROPPED_TILES_DIR = 'tiles_cropped'
DEBUG_DIR = 'debug_vlm'
RAW_TEXT_DIR = 'raw_text_output'
API_URL = "https://wms.inspire.geoportail.lu/geoserver/am/ogc/features/v1/collections/AM.HuntingLots/items?f=json&limit=1000&startIndex=0"
WMS_BASE_URL = "https://wmsproxy.geoportail.lu/ogcproxywms"

# --- OCRFlux Configuration ---
OCRFLUX_MODEL_PATH = "ChatDOC/OCRFlux-3B"
GPU_MEMORY_UTILIZATION = 0.8
MAX_MODEL_LEN = 8192

# ================================
# --- Utility Functions ---
# ================================

def safe_literal_eval(val):
	"""Safely evaluate string representations of lists/tuples."""
	try:
		if isinstance(val, str) and (val.startswith('[') or val.startswith('(')):
			return json.loads(val.replace("'", '"').replace("(", "[").replace(")", "]"))
		return val
	except Exception:
		return []

def ensure_directory(directory: str):
	"""Create directory if it doesn't exist."""
	if not os.path.exists(directory):
		os.makedirs(directory)

def file_uptodate(file_path: str, days: int, required_columns: List[str] = None) -> bool:
	"""Check if file exists, is recent, and has required columns."""
	if not os.path.exists(file_path):
		return False
	mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
	if (datetime.datetime.now() - mod_time).days > days:
		return False
	if required_columns:
		try:
			df = pd.read_csv(file_path, converters={'dates': safe_literal_eval})
			return all(col in df.columns for col in required_columns)
		except Exception:
			return False
	return True

def crop_image_center(image_path: str, output_path: str, crop_percentage: float = 0.66):
	"""
	Crops the center of an image and saves it.

	Args:
		image_path: Path to the input image.
		output_path: Path to save the cropped image.
		crop_percentage: The percentage of the image to keep (e.g., 0.66 for 66%).

	Returns:
		True if successful, False otherwise.
	"""
	try:
		image = cv2.imread(image_path)
		if image is None:
			return False

		height, width, _ = image.shape
		new_width = int(width * crop_percentage)
		new_height = int(height * crop_percentage)

		left = (width - new_width) // 2
		top = (height - new_height) // 2
		right = left + new_width
		bottom = top + new_height

		cropped_image = image[top:bottom, left:right]
		cv2.imwrite(output_path, cropped_image)
		return True
	except Exception as e:
		print(f"Error cropping image {image_path}: {e}")
		return False

# ================================
# --- Part 1: Hunting Lot Data ---
# ================================

def update_lots(url: str) -> pd.DataFrame:
	"""Fetches and processes hunting lot geo data."""
	print("Fetching latest hunting lot data from server...")
	try:
		with urlopen(url) as response:
			data = json.loads(response.read().decode())
	except Exception as e:
		print(f"Error fetching lot data: {e}")
		return pd.DataFrame()

	features = data.get('features', [])
	transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

	processed_lots: List[Dict[str, Any]] = []
	for item in tqdm(features, desc="Processing Lot Geometry"):
		properties = item.get('properties', {})
		lot_num = properties.get('gml_description', 'Unknown')
		geometry = item.get('geometry')

		lot_data = {
			'lot': lot_num,
			'polygon': None,
			'centroid': None,
			'bbox': None
		}

		if geometry:
			geom_type = geometry.get('type')
			coords = geometry.get('coordinates')
			lot_data['polygon'] = coords
			try:
				poly_obj = None
				if geom_type == 'Polygon' and coords:
					poly_obj = Polygon(coords[0])
				elif geom_type == 'MultiPolygon' and coords:
					poly_obj = MultiPolygon([Polygon(p[0]) for p in coords if len(p) > 0])

				if poly_obj:
					centroid = poly_obj.centroid
					lot_data['centroid'] = (centroid.x, centroid.y)
					x, y = transformer.transform(centroid.x, centroid.y)
					lot_data['bbox'] = (x - 1000, y - 1000, x + 1000, y + 1000)
			except Exception as e:
				print(f"Geometry error for lot {lot_num}: {e}")

		processed_lots.append(lot_data)

	df = pd.DataFrame(processed_lots)
	df = df[df['lot'] != 'Unknown'].copy()
	df['lot'] = pd.to_numeric(df['lot'], errors='coerce')
	df = df.dropna(subset=['lot']).astype({'lot': int}).sort_values('lot').reset_index(drop=True)

	df.to_csv(BASE_LOTS_FILE, index=False)
	print(f"Saved full lot geometry data → {BASE_LOTS_FILE}")
	return df

# ================================
# --- Part 2: Tile Download & Preprocessing ---
# ================================

def get_tile_path(lot_number: int) -> str:
	"""Get path to original tile."""
	return os.path.join(TILES_DIR, f"{lot_number:03d}.png")

def get_cropped_tile_path(lot_number: int) -> str:
	"""Get path to cropped tile."""
	return os.path.join(CROPPED_TILES_DIR, f"{lot_number:03d}_crop.png")

def tile_uptodate(lot_number: int, days: int = 7) -> bool:
	"""Check if tile is recent."""
	path = get_tile_path(lot_number)
	if not os.path.exists(path):
		return False
	mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
	return (datetime.datetime.now() - mod_time).days <= days

def download_tile(lot_number: int, bounds: Tuple[float, float, float, float]) -> bool:
	"""Download WMS tile for a lot."""
	try:
		bbox_str = ",".join([f"{coord:.2f}" for coord in bounds])
		params = {
			'SERVICE': 'WMS', 'VERSION': '1.3.0', 'REQUEST': 'GetMap',
			'FORMAT': 'image/png', 'TRANSPARENT': 'true',
			'LAYERS': 'anf_dates_battues', 'CRS': 'EPSG:3857',
			'STYLES': '', 'WIDTH': '512', 'HEIGHT': '512', 'BBOX': bbox_str
		}
		response = requests.get(WMS_BASE_URL, params=params, timeout=15)
		response.raise_for_status()
		with open(get_tile_path(lot_number), 'wb') as f:
			f.write(response.content)
		return True
	except requests.exceptions.RequestException as e:
		print(f"Tile download failed for lot {lot_number}: {e}")
		return False

# ================================
# --- Enhanced OCRFlux Wrapper ---
# ================================

class OCRFlux_Wrapper:
	def __init__(self, model_path: str = OCRFLUX_MODEL_PATH,
				 gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
				 max_model_len: int = MAX_MODEL_LEN):
		"""Initializes the OCRFlux model."""
		print("Loading OCRFlux model...")

		try:
			self.llm = LLM(
				model=model_path,
				gpu_memory_utilization=gpu_memory_utilization,
				max_model_len=max_model_len
			)
			print("OCRFlux model loaded successfully.")

		except Exception as e:
			print(f"FATAL: Failed to load OCRFlux model: {e}")
			raise e

	def _get_current_hunting_season(self) -> Tuple[datetime.datetime, datetime.datetime]:
		"""
		Determines the current hunting season based on current date.
		Season runs: September of year N to February of year N+1
		"""
		now = datetime.datetime.now()

		if now.month >= 9:
			season_start = datetime.datetime(now.year, 9, 1)
			season_end = datetime.datetime(now.year + 1, 2, 28)
		else:
			season_start = datetime.datetime(now.year - 1, 9, 1)
			season_end = datetime.datetime(now.year, 2, 28)

		if season_end.month == 2 and ((season_end.year % 4 == 0 and season_end.year % 100 != 0) or (season_end.year % 400 == 0)):
			season_end = season_end.replace(day=29)

		return season_start, season_end

	def _correct_date_to_season(self, date_obj: datetime.datetime, season_start: datetime.datetime,
								season_end: datetime.datetime) -> Optional[datetime.datetime]:
		"""Corrects a date to fit within the current hunting season and valid months."""
		# Only consider dates within valid hunting months (Sep-Feb)
		if date_obj.month not in [9, 10, 11, 12, 1, 2]:
			return None

		# Try to fit the date into the current season by adjusting the year
		for year_to_try in [season_start.year, season_end.year]:
			try:
				corrected_date = date_obj.replace(year=year_to_try)
				if season_start <= corrected_date <= season_end:
					return corrected_date
			except ValueError:  # Handles cases like Feb 29 in a non-leap year
				continue

		return None

	def _extract_structural_dates(self, lines: List[str], season_start: datetime.datetime,
									season_end: datetime.datetime) -> List[datetime.datetime]:
		"""
		Extracts dates following the known structure:
		Line 1: NNN
		Line 2: Battue/Treibjagd
		Lines 3+: Dates

		Returns parsed and validated date objects.
		"""
		parsed_dates = []
		date_pattern = r'\b(\d{2})/(\d{2})/(\d{4})\b'

		# Skip first 2 lines (lot number and Battue), process rest
		for line in lines[2:]:
			found_dates = re.finditer(date_pattern, line)
			for match in found_dates:
				date_str = match.group(0)
				day, month, year = match.groups()

				if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
					date_obj = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
					if date_obj:
						corrected_date = self._correct_date_to_season(date_obj, season_start, season_end)
						if corrected_date:
							parsed_dates.append(corrected_date)

		return parsed_dates

	def _is_text_centered(self, text: str) -> Tuple[bool, Dict[str, Any]]:
		"""
		Determines if the main text content is centered in the image.
		Uses heuristics based on expected structure.

		Expected structure:
		Line 1: NNN (3-digit lot number)
		Line 2: Battue/Treibjagd
		Lines 3+: DD/MM/YYYY dates

		Returns:
			(is_centered, position_info) tuple
		"""
		position_info = {
			'is_centered': False,
			'confidence': 0.0,
			'indicators': []
		}

		# If text is very short, assume it's centered (minimal content)
		if len(text.strip()) < 50:
			position_info['is_centered'] = True
			position_info['confidence'] = 0.8
			position_info['indicators'].append('short_text')
			return True, position_info

		lines = [line.strip() for line in text.split('\n') if line.strip()]

		if len(lines) < 2:
			return False, position_info

		# Check for key indicators of centered content with proper structure
		has_lot_start = False
		has_battue_word = False
		has_dates_following = False

		# Check if first line has a 3-digit number
		if re.search(r'^\s*\d{3}\s*$', lines[0]):
			has_lot_start = True
			position_info['indicators'].append('lot_number_first_line')

		# Check for Battue/Treibjagd in second line (fuzzy match)
		if len(lines) > 1:
			battue_pattern = r'[Bb]att[ueoi]{1,3}|[Tt]reib[jl]ag[dt]|[A-Z][a-z]{4,9}'
			if re.search(battue_pattern, lines[1], re.IGNORECASE):
				has_battue_word = True
				position_info['indicators'].append('battue_keyword_found')

		# Check if subsequent lines contain dates
		date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
		date_lines = sum(1 for line in lines[2:] if re.search(date_pattern, line))
		if date_lines > 0:
			has_dates_following = True
			position_info['indicators'].append(f'dates_in_{date_lines}_lines')

		# Scoring: centered if we see the expected structure
		score = 0
		if has_lot_start:
			score += 0.4
		if has_battue_word:
			score += 0.4
		if has_dates_following:
			score += 0.2

		position_info['confidence'] = score
		position_info['is_centered'] = score >= 0.6

		return score >= 0.6, position_info

	def _parse_ocr_output(self, text: str, lot_number: int, is_cropped: bool = True) -> Tuple[List[str], Dict[str, Any]]:
			"""
			Parses OCR output using a hybrid "best of both worlds" approach.
			
			1.  First, it attempts a high-precision search for the exact lot number block.
			2.  If that fails, it falls back to a positional check, assuming a centered,
				well-structured block of dates is correct.
			"""
			debug_info = {
				'raw_text': text,
				'logic_used': 'Not yet determined',
				'lot_match_found': False,
				'battue_match_found': False,
				'all_dates_found_raw': [],
				'valid_dates_final': [],
				'season_info': {},
			}

			season_start, season_end = self._get_current_hunting_season()
			debug_info['season_info'] = {
				'start': season_start.strftime('%d/%m/%Y'),
				'end': season_end.strftime('%d/%m/%Y')
			}

			if not text or len(text.strip()) < 3:
				return [], debug_info

			text_clean = text.replace('\r', ' ').strip()
			lines = [line.strip() for line in text_clean.split('\n') if line.strip()]

			if not lines:
				return [], debug_info

			parsed_dates = []
			final_dates = []

			# --- STEP 1: High-Precision Block-Aware Search ---
			debug_info['logic_used'] = 'Step 1: Strict Block Search'
			for i, line in enumerate(lines):
				lot_match = re.search(rf'^\s*0*{lot_number}\s*$', line)
				if lot_match:
					debug_info['lot_match_found'] = True
					if (i + 1) < len(lines) and re.search(r'[A-Za-z]{4,}', lines[i+1], re.IGNORECASE):
						debug_info['battue_match_found'] = True

					for subsequent_line in lines[i+2:]:
						if re.search(r'^\s*\d{2,3}\s*$', subsequent_line):
							break # Stop at the next lot number

						date_pattern = r'\b(\d{2})/(\d{2})/(\d{4})\b'
						for match in re.finditer(date_pattern, subsequent_line):
							date_obj = dateparser.parse(match.group(0), settings={'DATE_ORDER': 'DMY'})
							if date_obj:
								corrected_date = self._correct_date_to_season(date_obj, season_start, season_end)
								if corrected_date:
									parsed_dates.append(corrected_date)
					break # Found our lot, stop searching

			if parsed_dates:
				final_dates = sorted(list(set([d.strftime('%d/%m/%Y') for d in sorted(parsed_dates)])))

			# --- STEP 2: Positional Fallback Logic ---
			# This runs ONLY if the strict search in Step 1 found nothing.
			if not final_dates:
				debug_info['logic_used'] = 'Step 2: Positional Fallback'
				is_centered, position_info = self._is_text_centered(text_clean)
				
				if is_centered:
					# Use the original structural extraction method, assuming the first block is correct
					fallback_dates = self._extract_structural_dates(lines, season_start, season_end)
					if fallback_dates:
						parsed_dates.extend(fallback_dates)
						final_dates = sorted(list(set([d.strftime('%d/%m/%Y') for d in sorted(parsed_dates)])))
						# Note: We can't be sure about lot/battue match here, so we don't set them to True.
						debug_info['logic_used'] = 'Step 2: Positional Fallback - SUCCESS'


			debug_info['all_dates_found_raw'] = final_dates.copy()
			debug_info['valid_dates_final'] = final_dates
			return final_dates, debug_info

	def recognize_dates(self, image_path: str, lot_number: int,
					   crop_percentage: float = 0.66) -> Tuple[List[str], str, Dict[str, Any]]:
		"""
		Performs OCR on a CROPPED image using OCRFlux model.

		Args:
			image_path: Path to the original tile image
			lot_number: Expected lot number
			crop_percentage: Percentage of image to keep (0.66 = 66% center)

		Returns:
			(dates, raw_text, debug_info) tuple
		"""
		if not os.path.exists(image_path):
			return [], "Image path does not exist.", {}

		# --- Pre-processing: Crop the center of the image ---
		cropped_path = get_cropped_tile_path(lot_number)
		if not crop_image_center(image_path, cropped_path, crop_percentage):
			return [], "Failed to crop image.", {}

		try:
			# Use OCRFlux pipeline on the CROPPED image
			result = parse(self.llm, cropped_path)

			if result is None:
				return [], "OCRFlux parse returned None", {}

			raw_text = result.get('document_text', '')

			# Parse the output with position-aware logic
			dates, debug_info = self._parse_ocr_output(raw_text, lot_number, is_cropped=True)

			return dates, raw_text, debug_info

		except Exception as e:
			print(f"Error during OCRFlux processing for lot {lot_number}: {e}")
			return [], str(e), {}

# ================================
# --- Main Processing Logic ---
# ================================

def get_hunt_dates_with_ocr(df: pd.DataFrame) -> pd.DataFrame:
	"""Main function to extract hunting dates using OCRFlux."""
	ensure_directory(TILES_DIR)
	ensure_directory(CROPPED_TILES_DIR)
	ensure_directory(DEBUG_DIR)
	ensure_directory(RAW_TEXT_DIR)

	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	log_file = os.path.join(RAW_TEXT_DIR, f"ocrflux_log_{timestamp}.txt")
	detailed_log = os.path.join(RAW_TEXT_DIR, f"ocrflux_debug_log_{timestamp}.txt")

	print("\nStep 1: Downloading and preparing tiles...")
	for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing Tiles"):
		lot_num = int(row['lot'])

		# Download if needed
		if row['bbox'] and not tile_uptodate(lot_num):
			download_tile(lot_num, row['bbox'])

	try:
		ocr = OCRFlux_Wrapper()
	except Exception as e:
		print(f"\nFATAL ERROR during OCRFlux initialization: {e}")
		exit(1)

	all_dates = []
	stats = {
		'total_lots': len(df),
		'lots_with_dates': 0,
		'failed_lots': 0,
		'no_tile': 0,
		'no_lot_match': 0,
		'no_battue_match': 0
	}

	print("\nStep 2: Running OCRFlux to extract dates...")
	with open(log_file, 'w', encoding='utf-8') as log, \
		 open(detailed_log, 'w', encoding='utf-8') as detail_log:

		log.write("OCRFlux Log\n" + "="*80 + "\n\n")
		detail_log.write("OCRFlux Detailed Debug Log\n" + "="*80 + "\n\n")

		for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Dates"):
			lot_number = int(row['lot'])
			tile_path = get_tile_path(lot_number)

			log.write(f"\n--- LOT {lot_number:03d} ---\n")
			detail_log.write(f"\n{'='*80}\nLOT {lot_number:03d}\n{'='*80}\n")

			if not os.path.exists(tile_path):
				log.write("Status: Tile not found.\n")
				detail_log.write("Status: Tile not found.\n")
				all_dates.append([])
				stats['failed_lots'] += 1
				stats['no_tile'] += 1
				continue

			dates, raw_output, debug_info = ocr.recognize_dates(tile_path, lot_number)

			# Write to logs
			log.write(f"OCRFlux Raw Output:\n{raw_output}\n")
			log.write(f"Parsed Dates: {dates}\n")

			detail_log.write(f"Raw Output:\n{raw_output}\n\n")
			detail_log.write(f"Debug Info:\n")
			detail_log.write(f"  Season: {debug_info.get('season_info', {})}\n")
			detail_log.write(f"  Lot Match: {debug_info.get('lot_match', False)}\n")
			detail_log.write(f"  Battue Match: {debug_info.get('battue_match', False)}\n")
			detail_log.write(f"  All Dates Found: {debug_info.get('all_dates_found', [])}\n")
			detail_log.write(f"  Corrected Dates: {debug_info.get('corrected_dates', [])}\n")
			detail_log.write(f"  Valid Dates: {debug_info.get('valid_dates', [])}\n\n")

			all_dates.append(dates)

			if dates:
				stats['lots_with_dates'] += 1
				# Save successful images
				cv2.imwrite(
					os.path.join(DEBUG_DIR, f"{lot_number:03d}_success.png"),
					cv2.imread(tile_path)
				)
			else:
				if not debug_info.get('lot_match'):
					stats['no_lot_match'] += 1
				elif not debug_info.get('battue_match'):
					stats['no_battue_match'] += 1

	print(f"\nLogs saved:")
	print(f"  Summary: {log_file}")
	print(f"  Detailed: {detailed_log}")
	print(f"  Success images: {DEBUG_DIR}")

	print("\n=== OCRFlux Summary ===")
	print(f"Total lots processed: {stats['total_lots']}")
	print(f"Lots with dates found: {stats['lots_with_dates']}")
	print(f"Failed (no tile): {stats['no_tile']}")
	print(f"Failed (no lot match): {stats['no_lot_match']}")
	print(f"Failed (no battue match): {stats['no_battue_match']}")
	success_rate = (stats['lots_with_dates'] / stats['total_lots']) * 100 if stats['total_lots'] > 0 else 0
	print(f"Success Rate: {success_rate:.1f}%")

	df['dates'] = all_dates
	return df

# ================================
# --- Main Execution ---
# ================================

if __name__ == "__main__":
	REQUIRED_COLUMNS = ['lot', 'polygon', 'centroid', 'bbox', 'dates']

	# --- Part 1: Get Lot Data ---
	if file_uptodate(BASE_LOTS_FILE, days=30, required_columns=['lot', 'polygon', 'centroid', 'bbox']):
		print(f"Using recent lot data from '{BASE_LOTS_FILE}'")
		df_lots = pd.read_csv(BASE_LOTS_FILE, converters={'polygon': safe_literal_eval, 'centroid': safe_literal_eval, 'bbox': safe_literal_eval})
	else:
		print(f"Lot data is outdated or missing. Fetching new data...")
		df_lots = update_lots(API_URL)
		if df_lots.empty:
			print("Failed to get lot data. Exiting.")
			exit(1)

	# --- Part 2: Get Hunt Dates ---
	if file_uptodate(DATES_FILE, days=1, required_columns=REQUIRED_COLUMNS):
		print(f"\nUsing recent hunting dates from '{DATES_FILE}'")
		df_dates = pd.read_csv(DATES_FILE, converters={'dates': safe_literal_eval, 'polygon': safe_literal_eval, 'centroid': safe_literal_eval, 'bbox': safe_literal_eval})
	else:
		print("\nHunting dates file is outdated or missing. Running OCRFlux extraction process...")
		df_dates = get_hunt_dates_with_ocr(df_lots.copy())

		# Save with proper formatting
		df_save = df_dates.copy()
		df_save['dates'] = df_save['dates'].apply(lambda d: tuple(d) if isinstance(d, list) else ())
		df_save[REQUIRED_COLUMNS].to_csv(DATES_FILE, index=False)
		print(f"\nSaved latest hunting dates to '{DATES_FILE}'")

	# --- Part 3: Display Results ---
	print("\n--- Final Results ---")

	# Properly filter rows with dates
	df_dates['has_dates'] = df_dates['dates'].apply(lambda d: isinstance(d, (list, tuple)) and len(d) > 0)
	lots_with_dates = df_dates[df_dates['has_dates']].copy()

	print(f"Found dates for {len(lots_with_dates)} / {len(df_dates)} lots.")

	if not lots_with_dates.empty:
		print("\nFirst 15 lots with dates found:")
		for _, row in lots_with_dates.head(15).iterrows():
			dates = row['dates'] if isinstance(row['dates'], list) else list(row['dates'])
			print(f"Lot {row['lot']:03d}: {dates}")
	else:
		print("\nNo hunting dates were found for any lots.")
		print("\nTroubleshooting:")
		print("1. Check that tiles are downloaded in tiles/ directory")
		print("2. Review detailed log for OCRFlux output quality")
