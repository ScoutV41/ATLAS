#!/usr/bin/env python3
"""
ATLAS v1 — Automated Terrain and Luminance Analysis System (Lightweight)
A fast, shadow-only trail camera analyzer. 
Computes average hourly shadow angles and estimates geographic longitude 
using true solar noon and Equation of Time astrometry.
"""

import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import ollama

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGES_DIR  = Path("/images")
OUTPUT_CSV  = Path("output.csv")
SUMMARY_TXT = Path("summary.txt")
MODEL       = "qwen2.5vl:7b"

# V1 focuses ONLY on shadows for maximum inference speed
PROMPT = """Analyze this trail camera image carefully and return ONLY the following structured data, nothing else:
SHADOW_VISIBLE: yes/no
SHADOW_ANGLE: [0-360, or "none"]
SHADOW_LENGTH: [short/medium/long/none]
Do not add any explanation or preamble."""

SHADOW_LENGTH_RANK = {"short": 1, "medium": 2, "long": 3, "none": 4}
TIMESTAMP_RE       = re.compile(r"(\d{8}T\d{6}Z)")

CSV_FIELDS = ["timestamp", "shadow_angle", "shadow_length"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_timestamp(filename: str) -> datetime | None:
    m = TIMESTAMP_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

def parse_field(text: str, field: str) -> str:
    """Extracts values while ignoring markdown bold symbols injected by the VLM."""
    pattern = rf"^\s*\*?\*?{re.escape(field)}\*?\*?\s*:\s*(.+)$"
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else "unknown"

def extract_numeric_angle(angle_str: str) -> float | None:
    match = re.search(r"[-+]?\d*\.\d+|\d+", angle_str)
    return float(match.group()) if match else None

def angle_to_compass(angle: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(angle / 45) % 8]

def midpoint_datetime(dt1: datetime, dt2: datetime) -> datetime:
    return dt1 + (dt2 - dt1) / 2

def analyze_image(image_path: Path) -> dict:
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [str(image_path)],
            }]
        )
        text = response["message"]["content"]
    except Exception as exc:
        print(f"  [ERROR] Ollama call failed for {image_path.name}: {exc}", file=sys.stderr)
        return {"shadow_angle": "unknown", "shadow_length": "unknown"}

    return {
        "shadow_angle":  parse_field(text, "SHADOW_ANGLE"),
        "shadow_length": parse_field(text, "SHADOW_LENGTH"),
    }

def compute_solar_noon(rows: list[dict]) -> tuple[str, list[str]]:
    valid_shadows = [
        (SHADOW_LENGTH_RANK.get(r["shadow_length"].lower().strip(), 99), r["_dt"])
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
        noon_dt = frames[n // 2] if n % 2 == 1 else midpoint_datetime(frames[n // 2 - 1], frames[n // 2])
        per_day_noons.append(noon_dt)
        per_day_lines.append(
            f"  {date_key}  →  {noon_dt.strftime('%H:%M:%S')} UTC  "
            f"({n} shortest-shadow frame{'s' if n != 1 else ''})"
        )

    avg_secs = mean(dt.hour * 3600 + dt.minute * 60 + dt.second for dt in per_day_noons)
    h = int(avg_secs // 3600)
    m = int((avg_secs % 3600) // 60)
    s = int(avg_secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d} UTC", per_day_lines

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not IMAGES_DIR.exists() or not IMAGES_DIR.is_dir():
        print(f"[ERROR] {IMAGES_DIR} does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    image_files = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        print(f"No .jpg/.png files found in {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"ATLAS v1 — Found {len(image_files)} image(s) in {IMAGES_DIR}\n")

    rows: list[dict] = []

    with OUTPUT_CSV.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for img_path in image_files:
            ts = extract_timestamp(img_path.name)
            if ts is None:
                print(f"  [SKIP] No timestamp in filename: {img_path.name}")
                continue

            print(f"Processing: {img_path.name}")
            result = analyze_image(img_path)
            row = {"timestamp": ts.strftime("%Y%m%dT%H%M%SZ"), **result}
            writer.writerow(row)
            rows.append({**row, "_dt": ts})
            print(f"  → angle={result['shadow_angle']}  length={result['shadow_length']}")

    print(f"\nCSV written → {OUTPUT_CSV} ({len(rows)} rows processed)")

    # ── Shadow angle per hour ──────────────────────────────────────────────────
    hour_angles: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        angle = extract_numeric_angle(row["shadow_angle"])
        if angle is not None:
            hour_angles[row["_dt"].hour].append(angle)

    avg_angle_per_hour: dict[int, float] = {}
    for hour, angles in sorted(hour_angles.items()):
        sin_sum = sum(math.sin(math.radians(a)) for a in angles)
        cos_sum = sum(math.cos(math.radians(a)) for a in angles)
        avg_angle_per_hour[hour] = round(math.degrees(math.atan2(sin_sum, cos_sum)) % 360, 1)

    # ── Solar noon ─────────────────────────────────────────────────────────────
    solar_noon_avg, per_day_lines = compute_solar_noon(rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    lines = [
        "═══════════════════════════════════════════════════════",
        "  ATLAS v1 — Lightweight Astrometry Analysis",
        "═══════════════════════════════════════════════════════",
        "",
        f"  Images processed : {len(rows)}",
        f"  Output CSV       : {OUTPUT_CSV}",
        "",
        "─── Average Shadow Angle by Hour (UTC) ────────────────",
    ]

    if avg_angle_per_hour:
        for hour in sorted(avg_angle_per_hour):
            compass = angle_to_compass(avg_angle_per_hour[hour])
            lines.append(f"  {hour:02d}:00  →  {avg_angle_per_hour[hour]:>6.1f}°  ({compass})")
    else:
        lines.append("  No valid shadow angle data.")

    lines += ["", "─── Solar Noon Estimates (per day) ────────────────────"]
    lines += per_day_lines if per_day_lines else ["  Insufficient data."]
    lines += [
        "",
        f"  Averaged solar noon : {solar_noon_avg}",
        "  (True temporal midpoint of shortest-shadow frames, averaged across days)",
        "",
        "─── Longitude Estimate ─────────────────────────────────",
    ]

    if solar_noon_avg != "N/A":
        try:
            time_part = solar_noon_avg.split(" ")[0]
            h, m, s = map(int, time_part.split(":"))
            noon_secs = h * 3600 + m * 60 + s

            if rows:
                mid_row = rows[len(rows) // 2]
                day_of_year = mid_row["_dt"].timetuple().tm_yday
                b = math.radians((360 / 365) * (day_of_year - 81))
                eot_minutes = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
            else:
                day_of_year, eot_minutes = 1, 0.0

            true_prime_meridian_noon = 43200 - (eot_minutes * 60)
            delta_min  = (noon_secs - true_prime_meridian_noon) / 60
            longitude  = round(-delta_min * 0.25, 2)
            direction  = "W" if longitude < 0 else "E"

            lines += [
                f"  Estimated longitude : {abs(longitude):.2f}° {direction}",
                f"  Equation of Time correction applied (Julian Day {day_of_year}, EoT = {eot_minutes:.2f} min)",
                f"  Suggested search corridor: {abs(longitude):.1f}°–{abs(longitude)+8:.1f}° W",
            ]
        except (ValueError, IndexError, KeyError) as e:
            lines.append(f"  [ERROR] Longitude calculation failed: {e}")
    else:
        lines.append("  Insufficient shadow data for longitude estimate.")

    lines += ["", "═══════════════════════════════════════════════════════"]

    summary_text = "\n".join(lines)
    SUMMARY_TXT.write_text(summary_text)
    print(f"\n{summary_text}")
    print(f"\nSummary written → {SUMMARY_TXT}")

if __name__ == "__main__":
    main()