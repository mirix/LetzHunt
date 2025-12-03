# lux_hunt_vlm_deepseek.py
import os
import io
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
from typing import List, Dict, Any, Tuple, Optional, Set
import dateparser
import tempfile   # << kept (no-op, left as requested)

# --- Imports for OCRFlux ---
from vllm import LLM
from ocrflux.inference import parse

# ================================
# --- Global Constants ---
# ================================

BASE_LOTS_FILE = 'hunting_lots.csv'
DATES_FILE = 'hunting_dates.csv'
TILES_DIR = 'tiles'
DEBUG_DIR = 'debug_vlm'
RAW_TEXT_DIR = 'raw_text_output'
API_URL = "https://wms.inspire.geoportail.lu/geoserver/am/ogc/features/v1/collections/AM.HuntingLots/items?f=json&limit=1000&startIndex=0"
WMS_BASE_URL = "https://wmsproxy.geoportail.lu/ogcproxywms"

# --- Progressive Cropping Configuration ---
CROP_LEVELS = [0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66]  # Smaller progressive crop percentages
MIN_CROP_SIZE = 100  # Minimum dimension in pixels for cropping

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

def load_image_safe(image_path: str) -> Optional[np.ndarray]:
	"""Safely load an image, returning None if failed."""
	try:
		image = cv2.imread(image_path)
		if image is None:
			print(f"Warning: Could not load image {image_path}")
			return None
		return image
	except Exception as e:
		print(f"Error loading image {image_path}: {e}")
		return None

def crop_image_center_in_memory(image: np.ndarray, crop_percentage: float) -> Optional[np.ndarray]:
	"""
	Crops the center of an image in memory.
	
	Args:
		image: Input image as numpy array
		crop_percentage: The percentage of the image to keep
		
	Returns:
		Cropped image as numpy array or None if failed
	"""
	try:
		if image is None:
			return None
			
		height, width = image.shape[:2]
		
		# Calculate new dimensions
		new_width = max(int(width * crop_percentage), MIN_CROP_SIZE)
		new_height = max(int(height * crop_percentage), MIN_CROP_SIZE)
		
		# Ensure we don't crop beyond image boundaries
		new_width = min(new_width, width)
		new_height = min(new_height, height)
		
		# Calculate crop coordinates
		left = (width - new_width) // 2
		top = (height - new_height) // 2
		right = left + new_width
		bottom = top + new_height
		
		cropped = image[top:bottom, left:right]
		
		# Check if crop produced valid image
		if cropped.size == 0:
			return None
			
		return cropped
		
	except Exception as e:
		print(f"Error cropping image: {e}")
		return None

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
		response = requests.get(WMS_BASE_URL, params=params, timeout=30)
		response.raise_for_status()
		
		# Ensure directory exists
		ensure_directory(TILES_DIR)
		
		with open(get_tile_path(lot_number), 'wb') as f:
			f.write(response.content)
		
		# Verify the downloaded file
		if os.path.exists(get_tile_path(lot_number)) and os.path.getsize(get_tile_path(lot_number)) > 0:
			return True
		else:
			print(f"Downloaded file is empty or missing for lot {lot_number}")
			return False
			
	except requests.exceptions.RequestException as e:
		print(f"Tile download failed for lot {lot_number}: {e}")
		return False
	except Exception as e:
		print(f"Unexpected error downloading tile for lot {lot_number}: {e}")
		return False

# ================================
# --- Enhanced OCRFlux Wrapper with Progressive Cropping ---
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

		# Handle leap years for February
		if season_end.month == 2:
			if (season_end.year % 4 == 0 and season_end.year % 100 != 0) or (season_end.year % 400 == 0):
				season_end = season_end.replace(day=29)
			else:
				season_end = season_end.replace(day=28)

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

	def _extract_dates_from_text(self, text: str, season_start: datetime.datetime,
								season_end: datetime.datetime) -> List[datetime.datetime]:
		"""Extract all valid dates from text within hunting season."""
		parsed_dates = []
		date_pattern = r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'

		for match in re.finditer(date_pattern, text):
			date_str = match.group(0)
			try:
				day, month, year = match.groups()
				day_int, month_int, year_int = int(day), int(month), int(year)

				if 1 <= day_int <= 31 and 1 <= month_int <= 12:
					date_obj = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
					if date_obj:
						corrected_date = self._correct_date_to_season(date_obj, season_start, season_end)
						if corrected_date:
							parsed_dates.append(corrected_date)
			except (ValueError, TypeError):
				continue

		return parsed_dates

	def _find_lot_blocks(self, text: str, target_lot: int) -> Dict[str, Any]:
		"""
		Identify blocks of text belonging to different lots.
		Returns information about the target lot block and other lots.
		"""
		lines = [line.strip() for line in text.split('\n') if line.strip()]
		blocks = []
		current_block = []
		current_lot = None
		
		for line in lines:
			# Check if line contains a lot number (handle various formats)
			lot_match = re.search(r'^\s*(\d{2,3}[A-Z]?)\s*$', line, re.IGNORECASE)
			if lot_match:
				# Save previous block if it exists
				if current_block and current_lot is not None:
					blocks.append({
						'lot': current_lot,
						'lines': current_block,
						'text': '\n'.join(current_block)
					})
				
				# Start new block
				current_lot = lot_match.group(1)
				current_block = [line]
			elif current_lot is not None:
				current_block.append(line)
		
		# Add the last block
		if current_block and current_lot is not None:
			blocks.append({
				'lot': current_lot,
				'lines': current_block,
				'text': '\n'.join(current_block)
			})
		
		# Find target lot block and other lots
		target_block = None
		other_blocks = []
		
		for block in blocks:
			# Try to match lot number (handle formats like 134, 0134, 13A, etc.)
			block_lot_str = str(block['lot']).upper().replace('O', '0').lstrip('0')  # Handle OCR errors
			target_lot_str = str(target_lot)
			
			# Various matching strategies
			if (block_lot_str == target_lot_str or  # Direct match
				block_lot_str.zfill(3) == target_lot_str.zfill(3) or  # With leading zeros
				re.sub(r'[^A-Z0-9]', '', block_lot_str) == re.sub(r'[^A-Z0-9]', '', target_lot_str)):  # Alphanumeric
				target_block = block
			else:
				other_blocks.append(block)
		
		return {
			'target_block': target_block,
			'other_blocks': other_blocks,
			'all_blocks': blocks
		}

	def _analyze_text_structure(self, text: str, target_lot: int) -> Dict[str, Any]:
		"""
		Analyze text structure to determine confidence in date attribution.
		"""
		analysis = {
			'target_lot_found': False,
			'target_lot_confidence': 0.0,
			'has_battue_keyword': False,
			'date_count': 0,
			'structure_indicators': [],
			'position_confidence': 0.0
		}
		
		lines = [line.strip() for line in text.split('\n') if line.strip()]
		if not lines:
			return analysis
		
		# Check for target lot number with various formats
		target_lot_str = str(target_lot)
		for i, line in enumerate(lines):
			# Try different lot number patterns
			patterns = [
				rf'^\s*{target_lot_str}\s*$',  # Exact match
				rf'^\s*0*{target_lot_str}\s*$',  # With leading zeros
				rf'^\s*{target_lot_str}[A-Z]?\s*$',  # With optional letter
			]
			
			for pattern in patterns:
				if re.search(pattern, line, re.IGNORECASE):
					analysis['target_lot_found'] = True
					analysis['target_lot_confidence'] += 0.4
					analysis['structure_indicators'].append(f'target_lot_at_line_{i}')
					
					# Check if next line has battue keyword
					if i + 1 < len(lines):
						battue_pattern = r'[Bb]att[ueoi]{1,3}|[Tt]reib[jl]ag[dt]'
						if re.search(battue_pattern, lines[i + 1], re.IGNORECASE):
							analysis['has_battue_keyword'] = True
							analysis['target_lot_confidence'] += 0.3
							analysis['structure_indicators'].append('battue_keyword_found')
					break
		
		# Count dates
		date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'
		date_count = sum(1 for line in lines if re.search(date_pattern, line))
		analysis['date_count'] = date_count
		
		if date_count > 0:
			analysis['target_lot_confidence'] += min(0.3, date_count * 0.1)
			analysis['structure_indicators'].append(f'found_{date_count}_dates')
		
		return analysis
		
	def _process_single_crop(self, image: np.ndarray, lot_number: int, crop_level: float) -> Dict[str, Any]:
		"""
		Process a single crop level using memory-mapped temporary files for maximum performance.
		"""
		result = {
			'crop_level': crop_level,
			'raw_text': '',
			'dates': [],
			'lot_blocks': {},
			'structure_analysis': {},
			'error': None
		}

		try:
			if image is None:
				result['error'] = "Image is None"
				return result

			# Use a memory-mapped temporary file for maximum performance
			with tempfile.NamedTemporaryFile(suffix='.png', delete=True, dir='/dev/shm' if os.path.exists('/dev/shm') else None) as temp_file:
				# Save to what's essentially memory (RAM disk if available)
				success = cv2.imwrite(temp_file.name, image)
				if not success:
					result['error'] = "cv2.imwrite failed"
					return result
				
				# Force flush to ensure data is written
				temp_file.flush()
				
				# Process immediately
				ocr_result = parse(self.llm, temp_file.name)

			if ocr_result is None:
				result['error'] = "OCRFlux returned None"
				return result

			raw_text = ocr_result.get('document_text', '')
			result['raw_text'] = raw_text

			if not raw_text.strip():
				return result

			# Extract season information
			season_start, season_end = self._get_current_hunting_season()

			# Find lot blocks
			lot_blocks = self._find_lot_blocks(raw_text, lot_number)
			result['lot_blocks'] = {
				'target_block': lot_blocks['target_block'],
				'other_blocks_count': len(lot_blocks['other_blocks'])
			}

			# Analyze structure
			structure_analysis = self._analyze_text_structure(raw_text, lot_number)
			result['structure_analysis'] = structure_analysis

			# Extract dates
			if structure_analysis['target_lot_confidence'] >= 0.6:
				if lot_blocks['target_block']:
					dates = self._extract_dates_from_text(lot_blocks['target_block']['text'], season_start, season_end)
					result['dates'] = sorted(list(set([d.strftime('%d/%m/%Y') for d in dates])))
			else:
				all_dates = self._extract_dates_from_text(raw_text, season_start, season_end)
				result['dates'] = sorted(list(set([d.strftime('%d/%m/%Y') for d in all_dates])))

		except Exception as e:
			result['error'] = f"Processing error: {str(e)}"

		return result

	def _detect_discontinuity(self, current_crop_result: Dict[str, Any], previous_crop_result: Dict[str, Any], 
							 all_dates: Set[str], lot_number: int) -> Tuple[bool, str]:
		"""
		Detect discontinuities that suggest we should stop cropping.
		
		Returns:
			Tuple of (should_stop, reason)
		"""
		current_text = current_crop_result.get('raw_text', '').lower()
		
		# Rule 1: New lot number appears (other than target)
		# Look for any lot numbers in text that aren't our target
		lot_pattern = r'\b(\d{2,3}[a-z]?)\b'
		all_lots = set(re.findall(lot_pattern, current_text))
		target_lot_str = str(lot_number).lower()
		
		other_lots = []
		for found_lot in all_lots:
			# Normalize the found lot for comparison
			normalized_found = found_lot.replace('o', '0').lstrip('0')  # Handle OCR errors
			normalized_target = target_lot_str.replace('o', '0').lstrip('0')
			
			# Check if this is NOT our target lot
			if (normalized_found != normalized_target and 
				normalized_found.zfill(3) != normalized_target.zfill(3) and
				re.sub(r'[^a-z0-9]', '', normalized_found) != re.sub(r'[^a-z0-9]', '', normalized_target)):
				other_lots.append(found_lot)
		
		if other_lots:
			return True, f"Found other lot numbers: {other_lots}"
		
		# Rule 2: Two consecutive crops add nothing new
		current_dates = set(current_crop_result.get('dates', []))
		new_dates = current_dates - all_dates
		
		if not new_dates and previous_crop_result:
			prev_dates = set(previous_crop_result.get('dates', []))
			prev_new_dates = prev_dates - (all_dates - current_dates)
			
			if not prev_new_dates:
				return True, "Two consecutive crops added no new dates"
		
		return False, ""

	def recognize_dates_progressive(self, image_path: str, lot_number: int) -> Tuple[List[str], str, Dict[str, Any]]:
		"""
		Perform progressive cropping to extract dates with high precision.
		Uses smaller crop steps and stopping rules to avoid over-collecting.
		"""
		debug_info = {
			'progressive_results': [],
			'final_strategy': '',
			'total_crops_processed': 0,
			'dates_evolution': [],
			'season_info': {},
			'stopping_reason': '',
			'errors': []
		}
		
		if not os.path.exists(image_path):
			return [], f"Image path does not exist: {image_path}", debug_info
		
		# Load original image safely
		original_image = load_image_safe(image_path)
		if original_image is None:
			return [], f"Failed to load image: {image_path}", debug_info
		
		season_start, season_end = self._get_current_hunting_season()
		debug_info['season_info'] = {
			'start': season_start.strftime('%d/%m/%Y'),
			'end': season_end.strftime('%d/%m/%Y')
		}
		
		all_dates = set()
		crop_results = []
		consecutive_no_new_dates = 0
		previous_crop_result = None
		
		print(f"  Processing lot {lot_number} with progressive cropping...")
		
		for i, crop_level in enumerate(CROP_LEVELS):
			# Crop image
			cropped_image = crop_image_center_in_memory(original_image, crop_level)
			if cropped_image is None:
				debug_info['errors'].append(f"Failed to crop at level {crop_level}")
				continue
			
			# Process this crop level
			crop_result = self._process_single_crop(cropped_image, lot_number, crop_level)
			crop_results.append(crop_result)
			debug_info['progressive_results'].append(crop_result)
			
			current_dates = set(crop_result['dates'])
			new_dates = current_dates - all_dates
			
			# Update debug info
			debug_info['dates_evolution'].append({
				'crop_level': crop_level,
				'dates_found': list(current_dates),
				'new_dates': list(new_dates),
				'confidence': crop_result['structure_analysis'].get('target_lot_confidence', 0)
			})
			
			# Check for stopping conditions
			should_stop, stop_reason = self._detect_discontinuity(
				crop_result, previous_crop_result, all_dates, lot_number
			)
			
			if should_stop and i > 0:  # Don't stop on first crop
				debug_info['stopping_reason'] = stop_reason
				debug_info['final_strategy'] = f'Stopped at crop {crop_level}: {stop_reason}'
				break
			
			# Strategy decision - only add dates under specific conditions
			if new_dates:
				consecutive_no_new_dates = 0
				
				# Only add dates if we have good confidence or this is early in the process
				if (crop_result['structure_analysis'].get('target_lot_confidence', 0) >= 0.6 or 
					crop_level <= CROP_LEVELS[2]):  # More permissive in early crops
					all_dates.update(new_dates)
					debug_info['final_strategy'] = f'Added {len(new_dates)} dates from crop {crop_level}'
				else:
					debug_info['final_strategy'] = f'Rejected {len(new_dates)} dates from crop {crop_level} (low confidence)'
			else:
				consecutive_no_new_dates += 1
			
			previous_crop_result = crop_result
		
		debug_info['total_crops_processed'] = len([r for r in crop_results if r.get('error') is None])
		
		# If we didn't stop early but have consecutive no-new-dates, note it
		if not debug_info.get('stopping_reason') and consecutive_no_new_dates >= 2:
			debug_info['stopping_reason'] = f"Natural stop after {consecutive_no_new_dates} crops with no new dates"
		
		# Final validation and sorting
		final_dates = sorted(list(all_dates))
		
		# Combine raw text from all successful crops for logging
		successful_crops = [r for r in crop_results if r.get('error') is None]
		all_raw_text = "\n---\n".join([r.get('raw_text', '') for r in successful_crops])
		
		return final_dates, all_raw_text, debug_info

# ================================
# --- Main Processing Logic ---
# ================================

def get_hunt_dates_with_ocr(df: pd.DataFrame) -> pd.DataFrame:
	"""Main function to extract hunting dates using OCRFlux with progressive cropping."""
	ensure_directory(TILES_DIR)
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
			success = download_tile(lot_num, row['bbox'])
			if not success:
				print(f"Warning: Failed to download tile for lot {lot_num}")

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
		'tile_load_failed': 0,
		'low_confidence': 0,
		'early_stops': 0
	}

	print("\nStep 2: Running OCRFlux with progressive cropping...")
	with open(log_file, 'w', encoding='utf-8') as log, \
		 open(detailed_log, 'w', encoding='utf-8') as detail_log:

		log.write("OCRFlux Progressive Cropping Log\n" + "="*80 + "\n\n")
		detail_log.write("OCRFlux Progressive Cropping Detailed Log\n" + "="*80 + "\n\n")

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

			# Verify tile is readable
			test_image = load_image_safe(tile_path)
			if test_image is None:
				log.write("Status: Tile exists but cannot be loaded.\n")
				detail_log.write("Status: Tile exists but cannot be loaded.\n")
				all_dates.append([])
				stats['failed_lots'] += 1
				stats['tile_load_failed'] += 1
				continue

			dates, raw_output, debug_info = ocr.recognize_dates_progressive(tile_path, lot_number)

			# Track early stops
			if debug_info.get('stopping_reason'):
				stats['early_stops'] += 1

			# Write to logs
			log.write(f"Final Dates: {dates}\n")
			log.write(f"Strategy: {debug_info.get('final_strategy', 'Unknown')}\n")
			log.write(f"Stopping Reason: {debug_info.get('stopping_reason', 'None')}\n")
			log.write(f"Crops Processed: {debug_info.get('total_crops_processed', 0)}\n")
			log.write("-" * 40 + "\n")

			detail_log.write(f"Season: {debug_info.get('season_info', {})}\n")
			detail_log.write(f"Final Strategy: {debug_info.get('final_strategy', 'Unknown')}\n")
			detail_log.write(f"Stopping Reason: {debug_info.get('stopping_reason', 'None')}\n")
			detail_log.write(f"Final Dates: {dates}\n\n")
			
			# Log progressive results
			for i, crop_result in enumerate(debug_info.get('progressive_results', [])):
				crop_level = CROP_LEVELS[i] if i < len(CROP_LEVELS) else 'Unknown'
				detail_log.write(f"Crop Level {crop_level}:\n")
				detail_log.write(f"  Raw Text: {crop_result.get('raw_text', '')}\n")
				detail_log.write(f"  Dates: {crop_result.get('dates', [])}\n")
				detail_log.write(f"  Confidence: {crop_result.get('structure_analysis', {}).get('target_lot_confidence', 0):.2f}\n")
				detail_log.write(f"  Error: {crop_result.get('error', 'None')}\n")
				detail_log.write("-" * 20 + "\n")
			
			detail_log.write("\n")

			all_dates.append(dates)

			if dates:
				stats['lots_with_dates'] += 1
				# Save successful images
				success_image = load_image_safe(tile_path)
				if success_image is not None:
					cv2.imwrite(
						os.path.join(DEBUG_DIR, f"{lot_number:03d}_success.png"),
						success_image
					)
			else:
				stats['failed_lots'] += 1

	print(f"\nLogs saved:")
	print(f"  Summary: {log_file}")
	print(f"  Detailed: {detailed_log}")
	if os.path.exists(DEBUG_DIR):
		print(f"  Success images: {DEBUG_DIR}")

	print("\n=== Progressive Cropping Summary ===")
	print(f"Total lots processed: {stats['total_lots']}")
	print(f"Lots with dates found: {stats['lots_with_dates']}")
	print(f"Early stops due to discontinuity: {stats['early_stops']}")
	print(f"Failed (no tile): {stats['no_tile']}")
	print(f"Failed (tile load error): {stats['tile_load_failed']}")
	print(f"Failed (other): {stats['failed_lots'] - stats['no_tile'] - stats['tile_load_failed']}")
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
