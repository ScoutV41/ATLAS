#!/usr/bin/env python3
"""
ATLAS v3 — Automated Terrain and Luminance Analysis System
Ensemble Edition with Advanced Temporal Brightness Curve Modeling

This pipeline processes an entire image directory to map an overarching multi-day
solar luminance curve, isolating true daylight intervals across your data. By 
evaluating the timeline as a whole, it ignores short-term noise (like passing 
clouds or camera exposure glitches) and drops night frames completely before 
firing up the VLM ensemble pass, saving massive amounts of compute time.

LM Studio setup:
  1. Load your models in LM Studio
  2. Start the local server (default: http://localhost:1234)
  3. Load up to 3 models simultaneously if VRAM allows
  4. Update MODEL_ENDPOINTS below with your loaded model names

Dependencies: pip install Pillow numpy requests
"""

import base64
import csv
import json
import math
import re
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import requests # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGES_DIR = Path.home() / "Desktop" / "images"   # CHANGE THIS to your actual images folder
OUTPUT_CSV        = Path("output_v3.csv")
SUMMARY_TXT       = Path("summary_v3.txt")
SKIPPED_LOG       = Path("skipped_v3.txt")
ENSEMBLE_LOG      = Path("ensemble_log.csv")

# Curve Thresholds (Adopted from GAIA Suite)
DAY_THRESHOLD     = 20.0   # Brightness cross-point marking sunrise/sunset transitions
MIN_DAY_DURATION  = 3600   # Seconds (1 hour) minimum to prevent cloud shadow short-circuiting

# LM Studio server configuration
LMS_BASE_URL = "http://localhost:1234/v1"

# ── Models to use in ensemble ──────────────────────────────────────────────────
# Update these to match the exact model strings showing in LM Studio
MODEL_ENDPOINTS = [
    "qwen2.5-vl-7b-instruct",
    "gemma-4-e4b-it"  # <-- Swapped to the short identifier the server expects
]

# Disagreement threshold — flags if shadow angles differ by more than this
ANGLE_DISAGREEMENT_THRESH = 45.0

PROMPT = """Analyze this trail camera image carefully and return ONLY the following structured data, nothing else:
SHADOW_VISIBLE: yes/no
SHADOW_ANGLE: [0-360, or "none"]
SHADOW_LENGTH: [short/medium/long/none]
PLANT_TYPES: [comma separated list of visible plant descriptions]
VEGETATION_DENSITY: [sparse/moderate/dense]
ROCK_COLORS: [comma separated: black/dark-grey/light-grey/white/brown/red/orange/tan]
ROCK_TEXTURE: [smooth/rough/porous/angular/none]
VOLCANIC_FEATURES: yes/no/uncertain/none
Do not add any explanation or preamble. Return only the structured format above."""

SHADOW_LENGTH_RANK = {"short": 1, "medium": 2, "long": 3, "none": 4}
TIMESTAMP_RE       = re.compile(r"(\d{8}T\d{6}Z)")

CSV_FIELDS = [
    "timestamp", "mean_brightness", "is_day",
    "shadow_angle", "shadow_length",
    "plant_types", "vegetation_density",
    "rock_colors", "rock_texture", "volcanic_features",
    "ensemble_agreement", "models_used",
]

ENSEMBLE_LOG_FIELDS = [
    "timestamp", "model",
    "shadow_angle", "shadow_length", "plant_types",
    "rock_colors", "volcanic_features",
]

# ── Habitat classification rules ───────────────────────────────────────────────
HABITAT_RULES = [
    (
        "Sonoran Desert (SW Arizona / NW Mexico, <1200m, below 34°N)",
        "Brittlebush, saguaro, ocotillo, cholla, or palo verde indicators",
        lambda r: any(kw in r["plant_types"].lower()
                      for kw in ["saguaro", "ocotillo", "cholla", "brittlebush", "palo verde", "desert grass"]),
    ),
    (
        "Chihuahuan Desert (S New Mexico / W Texas, 32-34°N)",
        "Lechuguilla, sotol, creosote, yucca, or agave",
        lambda r: any(kw in r["plant_types"].lower()
                      for kw in ["lechuguilla", "sotol", "creosote",
                                 "yucca", "agave", "desert shrub"]),
    ),
    (
        "Mojave Desert (S Nevada / SE California)",
        "Joshua tree is the definitive Mojave indicator",
        lambda r: any(kw in r["plant_types"].lower()
                      for kw in ["joshua", "mojave yucca"]),
    ),
    (
        "Colorado Plateau / Southern Rockies (SW Colorado / NW New Mexico)",
        "Ponderosa pine + red/orange rock",
        lambda r: (any(kw in r["plant_types"].lower()
                       for kw in ["ponderosa", "pinyon", "juniper"])
                   and any(kw in r["rock_colors"].lower()
                           for kw in ["red", "orange", "tan"])),
    ),
    (
        "Cascades / Northern Rockies (NW)",
        "Douglas fir or hemlock with volcanic rock",
        lambda r: (any(kw in r["plant_types"].lower()
                       for kw in ["douglas fir", "hemlock", "western red cedar"])
                   and r["volcanic_features"].lower() in ("yes", "uncertain")),
    ),
    (
        "Sierra Nevada / Great Basin",
        "Jeffrey/lodgepole pine with light granite rock",
        lambda r: (any(kw in r["plant_types"].lower()
                       for kw in ["jeffrey pine", "lodgepole", "whitebark", "sagebrush"])
                   and any(kw in r["rock_colors"].lower()
                           for kw in ["light-grey", "white"])),
    ),
    (
        "High Desert Shrubland (Great Basin / Columbia Plateau)",
        "Sagebrush-dominant sparse vegetation",
        lambda r: ("sagebrush" in r["plant_types"].lower()
                   and r["vegetation_density"].lower() == "sparse"),
    ),
    (
        "Mixed Conifer Forest (montane, ambiguous region)",
        "Conifers present but insufficient detail",
        lambda r: any(kw in r["plant_types"].lower()
                      for kw in ["pine", "fir", "spruce", "conifer", "cedar"]),
    ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def calculate_brightness(image_path: Path) -> float:
    """Safely calculate mean pixel brightness (0.0 to 255.0)."""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            return float(np.mean(np.array(gray)))
    except Exception:
        return 0.0


def compute_day_intervals(dataset: list[dict]) -> list[tuple[datetime, datetime]]:
    """
    Builds a timeline curve across all extracted frame data, smoothing into 
    5-minute evaluation slots to find robust sunrise and sunset thresholds.
    """
    if not dataset:
        return []

    slots: dict[int, float] = {}
    for row in dataset:
        ts = row["_dt"]
        b = row["mean_brightness"]
        epoch = int(ts.timestamp())
        slot = (epoch // 300) * 300
        if slot not in slots or b > slots[slot]:
            slots[slot] = b

    curve = sorted(slots.items())

    intervals = []
    in_day = False
    day_start = None

    for epoch, b in curve:
        was_day = in_day
        in_day = b >= DAY_THRESHOLD

        if not was_day and in_day:
            day_start = datetime.fromtimestamp(epoch, tz=timezone.utc)
        elif was_day and not in_day and day_start is not None:
            day_end = datetime.fromtimestamp(epoch, tz=timezone.utc)
            duration = (day_end - day_start).total_seconds()
            if duration >= MIN_DAY_DURATION:
                intervals.append((day_start, day_end))
            day_start = None

    if in_day and day_start is not None:
        last_epoch = curve[-1][0]
        day_end = datetime.fromtimestamp(last_epoch, tz=timezone.utc)
        intervals.append((day_start, day_end))

    return intervals


def is_day_for_ts(ts: datetime, intervals: list[tuple[datetime, datetime]]) -> bool:
    """Checks if a frame timestamp falls inside any globally calculated daylight window."""
    for start, end in intervals:
        if start <= ts <= end:
            return True
    return False


def extract_timestamp(filename: str) -> datetime | None:
    m = TIMESTAMP_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_field(text: str, field: str) -> str:
    pattern = rf"^\s*\*?\*?{re.escape(field)}\*?\*?\s*:\s*(.+)$"
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else "unknown"


def extract_numeric_angle(angle_str: str) -> float | None:
    match = re.search(r"[-+]?\d*\.\d+|\d+", angle_str)
    return float(match.group()) if match else None


def angle_to_compass(angle: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(angle / 45) % 8]


def circular_mean(angles: list[float]) -> float:
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360


def circular_std(angles: list[float]) -> float:
    if len(angles) < 2:
        return 0.0
    mean_angle = circular_mean(angles)
    devs = [abs(((a - mean_angle + 180) % 360) - 180) for a in angles]
    return float(sum(devs) / len(devs))


def majority_vote(values: list[str]) -> str:
    clean = [v.lower().strip() for v in values if v.lower().strip() not in ("unknown", "")]
    if not clean:
        return "unknown"
    counts: dict[str, int] = defaultdict(int)
    for v in clean:
        counts[v] += 1
    return max(counts, key=lambda k: counts[k])


def union_plants(plant_lists: list[str]) -> str:
    all_plants: set[str] = set()
    for plant_str in plant_lists:
        if plant_str.lower() not in ("unknown", "none", ""):
            for p in plant_str.split(","):
                p = p.strip().lower()
                if p:
                    all_plants.add(p)
    return ", ".join(sorted(all_plants)) if all_plants else "unknown"


def midpoint_datetime(dt1: datetime, dt2: datetime) -> datetime:
    return dt1 + (dt2 - dt1) / 2


# ── LM Studio API Call ─────────────────────────────────────────────────────────

def query_model(model: str, image_path: Path, result_holder: dict, index: int) -> None:
    """Dispatches a single visual analysis inquiry to the loaded LM Studio model."""
    try:
        # 1. Dynamically determine mime type based on file extension
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        image_b64 = image_to_base64(image_path)
        
        # 2. Structure the payload using standard OpenAI vision format
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        response = requests.post(
            f"{LMS_BASE_URL}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        
        # If it fails, catch the error text before raising
        if response.status_code != 200:
            print(f"    [SERVER ERROR] {model} returned status {response.status_code}: {response.text}", file=sys.stderr)
            
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        result_holder[index] = {
            "model":              model,
            "shadow_angle":       parse_field(text, "SHADOW_ANGLE"),
            "shadow_length":      parse_field(text, "SHADOW_LENGTH"),
            "plant_types":        parse_field(text, "PLANT_TYPES"),
            "vegetation_density": parse_field(text, "VEGETATION_DENSITY"),
            "rock_colors":        parse_field(text, "ROCK_COLORS"),
            "rock_texture":       parse_field(text, "ROCK_TEXTURE"),
            "volcanic_features":  parse_field(text, "VOLCANIC_FEATURES"),
        }
    except Exception as exc:
        print(f"    [WARN] {model} execution error: {exc}", file=sys.stderr)
        result_holder[index] = None


def analyze_image_ensemble(image_path: Path) -> dict:
    """Runs all targets simultaneously using Python threads, aggregating predictions."""
    results: dict[int, dict | None] = {}
    threads = []

    for i, model in enumerate(MODEL_ENDPOINTS):
        t = threading.Thread(target=query_model, args=(model, image_path, results, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    valid = [r for r in results.values() if r is not None]

    if not valid:
        return {f: "unknown" for f in CSV_FIELDS
                if f not in ("timestamp", "ensemble_agreement", "models_used")} | {
            "ensemble_agreement": "all_failed",
            "models_used": "0",
            "_raw_results": [],
        }

    raw_angles = [extract_numeric_angle(r["shadow_angle"]) for r in valid]
    numeric_angles = [a for a in raw_angles if a is not None]

    if numeric_angles:
        merged_angle = str(round(circular_mean(numeric_angles), 1))
        angle_spread = circular_std(numeric_angles)
        angle_agreement = "DISAGREE" if angle_spread > ANGLE_DISAGREEMENT_THRESH else "agree"
    else:
        merged_angle = "none"
        angle_spread = 0.0
        angle_agreement = "no_shadow"

    merged_length     = majority_vote([r["shadow_length"] for r in valid])
    merged_plants     = union_plants([r["plant_types"] for r in valid])
    merged_density    = majority_vote([r["vegetation_density"] for r in valid])
    merged_rocks      = union_plants([r["rock_colors"] for r in valid])
    merged_texture    = majority_vote([r["rock_texture"] for r in valid])
    merged_volcanic   = majority_vote([r["volcanic_features"] for r in valid])

    if len(valid) == 1:
        agreement = "single_model"
    elif angle_agreement == "DISAGREE":
        agreement = f"ANGLE_DISAGREE({angle_spread:.0f}°)"
    else:
        agreement = f"consensus({len(valid)}_models)"

    return {
        "shadow_angle":       merged_angle,
        "shadow_length":      merged_length,
        "plant_types":        merged_plants,
        "vegetation_density": merged_density,
        "rock_colors":        merged_rocks,
        "rock_texture":       merged_texture,
        "volcanic_features":  merged_volcanic,
        "ensemble_agreement": agreement,
        "models_used":        str(len(valid)),
        "_raw_results":       valid,
    }


# ── Habitat + Solar Noon Analytics ─────────────────────────────────────────────

def classify_habitat(rows: list[dict]) -> tuple[str, str]:
    tally: dict[str, int] = defaultdict(int)
    rule_map = {label: note for label, note, _ in HABITAT_RULES}
    for row in rows:
        for label, _, match_fn in HABITAT_RULES:
            try:
                if match_fn(row):
                    tally[label] += 1
            except Exception:
                pass
    if not tally:
        return "Unclassified", "No vegetation or geology indicators matched"
    winner = max(tally, key=lambda k: tally[k])
    total  = sum(tally.values())
    pct    = round(100 * tally[winner] / total)
    return winner, f"{rule_map[winner]} ({pct}% of total rule-matches)"


def compute_solar_noon(rows: list[dict]) -> tuple[str, list[str]]:
    valid_shadows = [
        (SHADOW_LENGTH_RANK[r["shadow_length"].lower().strip()], r["_dt"])
        for r in rows
        if r["shadow_length"].lower().strip() in SHADOW_LENGTH_RANK
    ]
    if not valid_shadows:
        return "N/A", []
    min_rank = min(valid_shadows, key=lambda x: x[0])[0]
    daily: dict[str, list[datetime]] = defaultdict(list)
    for rank, dt in valid_shadows:
        if rank == min_rank:
            daily[dt.strftime("%Y%m%d")].append(dt)
    per_day_noons: list[datetime] = []
    per_day_lines: list[str] = []
    for date_key in sorted(daily):
        frames = sorted(daily[date_key])
        n = len(frames)
        noon_dt = (frames[n // 2] if n % 2 == 1
                   else midpoint_datetime(frames[n // 2 - 1], frames[n // 2]))
        per_day_noons.append(noon_dt)
        per_day_lines.append(
            f"  {date_key}  →  {noon_dt.strftime('%H:%M:%S')} UTC  ({n} frame{'s' if n != 1 else ''})"
        )
    avg_secs = mean(dt.hour * 3600 + dt.minute * 60 + dt.second for dt in per_day_noons)
    h = int(avg_secs // 3600)
    m = int((avg_secs % 3600) // 60)
    s = int(avg_secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d} UTC", per_day_lines


# ── Execution Logic ────────────────────────────────────────────────────────────

def main() -> None:
    if not IMAGES_DIR.exists() or not IMAGES_DIR.is_dir():
        print(f"[ERROR] Specified folder path does not exist: {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    image_files = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        print(f"No targeting frames identified in {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"ATLAS v3 — Curve-Optimized Ensemble Edition")
    print(f"Ensemble Configuration : {', '.join(MODEL_ENDPOINTS)}")
    print(f"Total Frames Identified: {len(image_files)}")
    print(f"LM Studio Server Target: {LMS_BASE_URL}\n")

    # ── Phase 1: Pre-process Timeline Luminance Curve ─────────────────────────
    print("Phase 1: Generating global solar luminance curves...")
    pre_process_data: list[dict] = []
    skipped: list[str] = []

    for img_path in image_files:
        ts = extract_timestamp(img_path.name)
        if ts is None:
            skipped.append(f"NO_TIMESTAMP: {img_path.name}")
            continue
        
        mb = calculate_brightness(img_path)
        pre_process_data.append({
            "_dt": ts,
            "mean_brightness": round(mb, 2),
            "path": img_path
        })

    intervals = compute_day_intervals(pre_process_data)
    print(f"  System mapped {len(intervals)} authentic daylight cycle window(s):")
    for start, end in intervals:
        print(f"    ↳ {start.strftime('%Y-%m-%d %H:%M:%S')} UTC  thru  {end.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Luminance evaluation complete. Commencing VLM Pass.\n")

    # ── Phase 2: Targeted Ensemble Extraction ──────────────────────────────────
    print("Phase 2: Executing Ensemble Inference on Daylight Frames...")
    rows: list[dict] = []
    disagreements = 0

    with (OUTPUT_CSV.open("w", newline="") as csvfile,
          ENSEMBLE_LOG.open("w", newline="") as logfile):

        writer     = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        log_writer = csv.DictWriter(logfile, fieldnames=ENSEMBLE_LOG_FIELDS)
        writer.writeheader()
        log_writer.writeheader()

        for i, item in enumerate(pre_process_data, 1):
            img_path = item["path"]
            ts = item["_dt"]
            ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
            mb = item["mean_brightness"]
            
            is_day = is_day_for_ts(ts, intervals)

            if not is_day:
                print(f"[{i}/{len(pre_process_data)}] [SKIP] Timeline curve classified frame as Night: {img_path.name}")
                skipped.append(f"NIGHT_CURVE_SKIPPED: {img_path.name}")
                
                # Still log the basic structural metrics to the primary CSV output tracker
                writer.writerow({
                    "timestamp": ts_str, "mean_brightness": mb, "is_day": "False",
                    "shadow_angle": "none", "shadow_length": "none", "plant_types": "none",
                    "vegetation_density": "none", "rock_colors": "none", "rock_texture": "none",
                    "volcanic_features": "none", "ensemble_agreement": "skipped_night", "models_used": "0"
                })
                continue

            print(f"[{i}/{len(pre_process_data)}] Analyzing: {img_path.name} (Luminance: {mb})")
            result = analyze_image_ensemble(img_path)
            agreement = result["ensemble_agreement"]

            if "DISAGREE" in agreement:
                disagreements += 1
                print(f"  ⚠️  ENCOUNTERED ENSEMBLE VARIANCE: {agreement}")
            else:
                print(f"  ✅ Resolution: {agreement}")

            row = {
                "timestamp":          ts_str,
                "mean_brightness":    mb,
                "is_day":             "True",
                "shadow_angle":       result["shadow_angle"],
                "shadow_length":      result["shadow_length"],
                "plant_types":        result["plant_types"],
                "vegetation_density": result["vegetation_density"],
                "rock_colors":        result["rock_colors"],
                "rock_texture":       result["rock_texture"],
                "volcanic_features":  result["volcanic_features"],
                "ensemble_agreement": agreement,
                "models_used":        result["models_used"],
            }
            writer.writerow(row)
            rows.append({**row, "_dt": ts})

            for raw in result.get("_raw_results", []):
                log_writer.writerow({
                    "timestamp":         ts_str,
                    "model":             raw["model"],
                    "shadow_angle":      raw["shadow_angle"],
                    "shadow_length":     raw["shadow_length"],
                    "plant_types":       raw["plant_types"],
                    "rock_colors":       raw["rock_colors"],
                    "volcanic_features": raw["volcanic_features"],
                })

    SKIPPED_LOG.write_text("\n".join(skipped))
    print(f"\n{'─'*55}")
    print(f"Active Daylight Profiles Cataloged : {len(rows)} frames")
    print(f"Supressed Timeline Logs            : {len(skipped)}")
    print(f"Ensemble Discordancies Tracked    : {disagreements}")
    print(f"{'─'*55}\n")

    # ── Compilation & Summary Compilation ──────────────────────────────────────
    hour_angles: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        angle = extract_numeric_angle(row["shadow_angle"])
        if angle is not None:
            hour_angles[row["_dt"].hour].append(angle)

    avg_angle_per_hour: dict[int, float] = {}
    for hour, angles in sorted(hour_angles.items()):
        avg_angle_per_hour[hour] = round(circular_mean(angles), 1)

    solar_noon_avg, per_day_lines = compute_solar_noon(rows)
    habitat_label, habitat_note   = classify_habitat(rows)

    lines = [
        "═══════════════════════════════════════════════════════",
        "  ATLAS v3 — Curve-Optimized Ensemble Architecture",
        "═══════════════════════════════════════════════════════",
        "",
        f"  Images processed      : {len(rows)}",
        f"  Images skipped        : {len(skipped)}",
        f"  Disagreements flagged : {disagreements}",
        f"  Models used           : {', '.join(MODEL_ENDPOINTS)}",
        f"  Ensemble log          : {ENSEMBLE_LOG}",
        "",
        "─── Average Shadow Angle by Hour (UTC) ────────────────",
    ]

    if avg_angle_per_hour:
        for hour in sorted(avg_angle_per_hour):
            compass = angle_to_compass(avg_angle_per_hour[hour])
            lines.append(f"  {hour:02d}:00  →  {avg_angle_per_hour[hour]:>6.1f}°  ({compass})")
    else:
        lines.append("  No valid shadow angle data collected during day intervals.")

    lines += ["", "─── Solar Noon Estimates (per day) ────────────────────"]
    lines += per_day_lines if per_day_lines else ["  Insufficient data pairs generated."]
    lines += [
        "",
        f"  Averaged solar noon : {solar_noon_avg}",
        "",
        "─── Habitat Classification ─────────────────────────────",
        f"  Best match : {habitat_label}",
        f"  Basis      : {habitat_note}",
        "",
        "─── Longitude Estimate ─────────────────────────────────",
    ]

    if solar_noon_avg != "N/A":
        try:
            h, m, s = map(int, solar_noon_avg.split(" ")[0].split(":"))
            noon_secs = h * 3600 + m * 60 + s
            if rows:
                day_of_year = rows[len(rows) // 2]["_dt"].timetuple().tm_yday
                b = math.radians((360 / 365) * (day_of_year - 81))
                eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
            else:
                day_of_year, eot = 1, 0.0
            true_noon  = 43200 - (eot * 60)
            delta_min  = (noon_secs - true_noon) / 60
            longitude  = round(-delta_min * 0.25, 2)
            direction  = "W" if longitude < 0 else "E"
            lines += [
                f"  Estimated longitude : {abs(longitude):.2f}° {direction}",
                f"  EoT correction      : Julian Day {day_of_year}, EoT = {eot:.2f} min",
                f"  NOTE: hill/crevice bias may shift true longitude 2-8° further west",
                f"  Suggested corridor  : {abs(longitude):.1f}°–{abs(longitude)+8:.1f}° W",
            ]
        except Exception as e:
            lines.append(f"  [ERROR] Astrometric projection failure: {e}")
    else:
        lines.append("  Insufficient shadow metadata present to calculate longitude.")

    lines += ["", "═══════════════════════════════════════════════════════"]
    summary_text = "\n".join(lines)
    # To this safe version:
    SUMMARY_TXT.write_text(summary_text, encoding="utf-8")
    
    print(summary_text)
    print(f"\nExecution Report Compiled:")
    print(f"  Summary Artifact → {SUMMARY_TXT}")
    print(f"  Merged Metrics   → {OUTPUT_CSV}")
    print(f"  Ensemble Matrix  → {ENSEMBLE_LOG}")


if __name__ == "__main__":
    main()