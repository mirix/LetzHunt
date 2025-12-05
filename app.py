import datetime as dt
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import gpxpy
import gpxpy.gpx
import gradio as gr
import pandas as pd
import folium
from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.ops import unary_union

# =======================
# Data loading & parsing
# =======================

CSV_PATH = "hunting_dates.csv"
df_raw = pd.read_csv(CSV_PATH)

# --- Robust polygon parsing ---

def _normalize_polygon_nested(obj: Any) -> List[List[Tuple[float, float]]]:
    """
    Normalize structure into list of rings (each ring = list[(lon, lat)]).
    """
    def is_point(p):
        return (
            isinstance(p, (list, tuple))
            and len(p) >= 2
            and isinstance(p[0], (int, float))
            and isinstance(p[1], (int, float))
        )

    if isinstance(obj, (list, tuple)) and obj and all(is_point(pt) for pt in obj):
        ring = [(float(pt[0]), float(pt[1])) for pt in obj]
        return [ring]

    if isinstance(obj, (list, tuple)):
        rings: List[List[Tuple[float, float]]] = []
        if obj and all(isinstance(item, (list, tuple)) for item in obj):
            if any(isinstance(item, (list, tuple)) and item and all(is_point(pt) for pt in item) for item in obj):
                for item in obj:
                    if not item:
                        continue
                    if all(is_point(pt) for pt in item):
                        ring = [(float(pt[0]), float(pt[1])) for pt in item]
                        rings.append(ring)
                    else:
                        sub_rings = _normalize_polygon_nested(item)
                        rings.extend(sub_rings)
                return rings

        for child in obj:
            if child is None:
                continue
            sub_rings = _normalize_polygon_nested(child)
            if sub_rings:
                return sub_rings

    raise ValueError(f"Cannot interpret polygon structure: {obj!r}")


def parse_polygon_str(poly_str: str) -> Polygon:
    if not isinstance(poly_str, str):
        raise ValueError(f"Polygon value is not a string: {poly_str!r}")

    s = poly_str.strip()

    parsed = None
    try:
        parsed = json.loads(s)
    except Exception:
        try:
            parsed = eval(s, {"__builtins__": None}, {})
        except Exception as e:
            raise ValueError(f"Failed to parse polygon string: {poly_str!r}") from e

    rings = _normalize_polygon_nested(parsed)
    if not rings:
        raise ValueError(f"No valid ring found in polygon: {poly_str!r}")

    shell = rings[0]
    holes = rings[1:] if len(rings) > 1 else None
    return Polygon(shell, holes)


# --- Parse dates column into list of datetime.date ---

DATE_RE = re.compile(r"(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})")

def parse_dates_field(dates_field: str) -> List[dt.date]:
    if not isinstance(dates_field, str):
        return []
    matches = DATE_RE.findall(dates_field)
    parsed: List[dt.date] = []
    for m in matches:
        if "/" in m:
            d = dt.datetime.strptime(m, "%d/%m/%Y").date()
        else:
            d = dt.datetime.strptime(m, "%Y-%m-%d").date()
        parsed.append(d)
    return sorted(set(parsed))


records: List[Dict] = []
for _, row in df_raw.iterrows():
    lot = int(row["lot"])
    poly = parse_polygon_str(str(row["polygon"]))
    dates = parse_dates_field(str(row.get("dates", "")))
    records.append(
        {
            "lot": lot,
            "polygon": poly,
            "dates": dates,
        }
    )

df = pd.DataFrame(records)

all_dates = sorted({d for dates in df["dates"] for d in dates})
if not all_dates:
    MIN_DATE = MAX_DATE = dt.date.today()
else:
    MIN_DATE = all_dates[0]
    MAX_DATE = all_dates[-1]

lot_dates_map: Dict[int, List[dt.date]] = {
    lot: sorted({d for dates in group["dates"] for d in dates})
    for lot, group in df.groupby("lot")
}

def date_to_str(d: dt.date) -> str:
    return d.strftime("%d/%m/%Y")


# =======================
# GPX parsing / overlap
# =======================

def parse_gpx_file(file_obj) -> List[LineString]:
    """
    Handles both filepath string and file object.
    """
    path = None
    if isinstance(file_obj, str):
        path = file_obj
    elif hasattr(file_obj, "name"):
        path = file_obj.name

    if not path or not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        gpx = gpxpy.parse(f)

    lines: List[LineString] = []

    for track in gpx.tracks:
        for segment in track.segments:
            pts = [(p.longitude, p.latitude) for p in segment.points]
            if len(pts) >= 2:
                lines.append(LineString(pts))

    for route in gpx.routes:
        pts = [(p.longitude, p.latitude) for p in route.points]
        if len(pts) >= 2:
            lines.append(LineString(pts))

    return lines


def split_line_by_polygons(line: LineString, polys: List[Polygon]) -> Tuple[List[LineString], List[LineString]]:
    if not polys:
        return [], [line]

    union_poly = unary_union(polys)
    inter = line.intersection(union_poly)

    intersecting: List[LineString] = []
    non_intersecting: List[LineString] = []

    def flatten_geom(g, target: List[LineString]):
        if g.is_empty:
            return
        if isinstance(g, LineString):
            target.append(g)
        elif isinstance(g, MultiLineString):
             for part in g.geoms:
                 target.append(part)
        else:
            # Handle GeometryCollection or other mixed types if necessary
            try:
                for part in g.geoms:
                    flatten_geom(part, target)
            except AttributeError:
                pass

    flatten_geom(inter, intersecting)

    diff = line.difference(union_poly)
    flatten_geom(diff, non_intersecting)

    return intersecting, non_intersecting


# =======================
# Date helpers
# =======================

def clamp_date(d: dt.date) -> dt.date:
    if d < MIN_DATE:
        return MIN_DATE
    if d > MAX_DATE:
        return MAX_DATE
    return d

def default_date() -> dt.date:
    today = dt.date.today()
    return clamp_date(today)

def date_to_timestamp(d: dt.date) -> float:
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()

def normalize_selected_ts(ts: Any) -> dt.date:
    if ts is None:
        return default_date()
    try:
        t = float(ts)
        d = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).date()
    except Exception:
        return default_date()
    return clamp_date(d)


# =======================
# Map rendering (Folium)
# =======================

LUX_CENTER = [49.8153, 6.13]
LUX_ZOOM = 10

def build_map_html(selected_ts: Any,
                   gpx_lines: Optional[List[LineString]]) -> str:
    d = normalize_selected_ts(selected_ts)

    # Responsive map configuration
    m = folium.Map(
        location=LUX_CENTER,
        zoom_start=LUX_ZOOM,
        width="100%",
        height="100vh",
        tiles="OpenStreetMap",
        attr=" ",
    )
    m.options['attributionControl'] = False

    # 1. Add Hunting Lots (Polygons)
    active_rows = df[df["dates"].apply(lambda ds: d in ds)]
    active_polys: List[Polygon] = []

    for _, row in active_rows.iterrows():
        lot = row["lot"]
        poly: Polygon = row["polygon"]
        if poly.is_empty:
            continue
        active_polys.append(poly)

        lot_str = f"{lot:03d}"
        dates_for_lot = lot_dates_map.get(lot, [])
        date_lines = [f"<b>{date_to_str(x)}</b>" for x in dates_for_lot]
        html_popup = f"<b>{lot_str}</b><br><br>" + "<br>".join(date_lines)

        gj = folium.GeoJson(
            data=poly.__geo_interface__,
            style_function=lambda feat, col="crimson": {
                "fillColor": col,
                "color": col,
                "weight": 2,
                "fillOpacity": 0.45,
            },
        )
        folium.Popup(html_popup, max_width=300).add_to(gj)
        gj.add_to(m)

    # 2. Add GPX Lines (GeoJSON)
    all_line_geoms = [] # For calculating bounds

    if gpx_lines:
        if active_polys:
            # Split lines into overlapping (red) and non-overlapping (blue)
            inter_lines = []
            non_inter_lines = []

            for line in gpx_lines:
                inter, non_inter = split_line_by_polygons(line, active_polys)
                inter_lines.extend(inter)
                non_inter_lines.extend(non_inter)

            if inter_lines:
                folium.GeoJson(
                    data=MultiLineString(inter_lines).__geo_interface__,
                    style_function=lambda x: {"color": "red", "weight": 4, "opacity": 0.9}
                ).add_to(m)
                all_line_geoms.extend(inter_lines)

            if non_inter_lines:
                folium.GeoJson(
                    data=MultiLineString(non_inter_lines).__geo_interface__,
                    style_function=lambda x: {"color": "blue", "weight": 3, "opacity": 0.7}
                ).add_to(m)
                all_line_geoms.extend(non_inter_lines)
        else:
            # No hunting lots active, draw all lines in blue
            if gpx_lines:
                folium.GeoJson(
                    data=MultiLineString(gpx_lines).__geo_interface__,
                    style_function=lambda x: {"color": "blue", "weight": 3, "opacity": 0.7}
                ).add_to(m)
                all_line_geoms.extend(gpx_lines)

    # 3. Auto-zoom to GPX track
    if all_line_geoms:
        # Calculate bounds: (minx, miny, maxx, maxy) -> (min_lon, min_lat, max_lon, max_lat)
        min_x, min_y, max_x, max_y = MultiLineString(all_line_geoms).bounds
        # Folium fit_bounds takes [[min_lat, min_lon], [max_lat, max_lon]]
        m.fit_bounds([[min_y, min_x], [max_y, max_x]])

    return m._repr_html_()


# =======================
# Gradio interface logic
# =======================

def app_fn(selected_ts: Any, gpx_file):
    new_lines: List[LineString] = []
    if gpx_file is not None:
        new_lines = parse_gpx_file(gpx_file)
    map_html = build_map_html(selected_ts, new_lines)
    return map_html

def clear_gpx_fn(selected_ts: Any):
    map_html = build_map_html(selected_ts, gpx_lines=None)
    return map_html, None

# =======================
# Build Gradio UI
# =======================

# Removed css argument to fix crash. Added gr.HTML for styles below.
with gr.Blocks(title="Luxembourg Hunting Lots") as demo:

    # Inject CSS for hiding footer
    gr.HTML("""
    <style>
        footer {visibility: hidden}
        .gradio-container {min-height: 0px !important;}
    </style>
    """)

    gr.Markdown(
        "## Hunting Dates in Luxembourg\n"
        "Choose a date and optionally upload a GPX track to see overlaps "
        "with active hunting lots.\n\n"
        f"Data available from **{MIN_DATE}** to **{MAX_DATE}**."
    )

    with gr.Row():
        date_input = gr.DateTime(
            label="Select date",
            value=date_to_timestamp(default_date()),
            include_time=False,
        )

        gpx_input = gr.File(
            label="Upload GPX track (optional)",
            file_types=[".gpx"],
            interactive=True
        )

        clear_btn = gr.Button("Clear GPX")

    map_output = gr.HTML()

    # Initial map
    init_map_html = build_map_html(date_to_timestamp(default_date()), gpx_lines=None)
    map_output.value = init_map_html

    # Events
    date_input.change(
        fn=app_fn,
        inputs=[date_input, gpx_input],
        outputs=[map_output],
        show_progress=False,
    )

    gpx_input.change(
        fn=app_fn,
        inputs=[date_input, gpx_input],
        outputs=[map_output],
        show_progress=True,
    )

    clear_btn.click(
        fn=clear_gpx_fn,
        inputs=[date_input],
        outputs=[map_output, gpx_input],
        show_progress=False,
    )

    gr.Markdown(
        """
---

[Freedom Luxembourg](https://www.freeletz.lu/freeletz/)

The information provided is offered “as is”, without any guarantees.
The only authoritative source of information is the official [GeoPortail](https://map.geoportail.lu/communes/Luxembourg/anf_dates_battues/?lang=en)
        """
    )

if __name__ == "__main__":
    demo.launch()
