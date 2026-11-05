#!/usr/bin/env python3
"""
ATLAS v5.2 — Sequential Batch OSINT Engine (VRAM Optimized & Hardened)
Addresses Astrometric framing, timezone offsets, VRAM eviction, and JSON fragility.
"""

import base64
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import requests
from PIL import Image

# ── Global Configuration ──────────────────────────────────────────────────────
IMAGES_DIR  = Path.home() / "Desktop" / "images"
OUTPUT_JSON = Path("atlas_v5_analysis.json")
SUMMARY_TXT = Path("summary_v5_report.txt")

# API Configuration
API_BASE_URL = "http://localhost:1234/v1/chat/completions"
OLLAMA_EVICT_URL = "http://localhost:11434/api/generate" # Used if running Ollama to force VRAM dump

MODELS = {
    "vision_combo": "qwen_qwen2.5-vl-7b-instruct",
    "geology":      "pixtral-12b",
    "synthesis":    "gemma-4-31b",  # Use your local server's exact identifier string
}

# ── Crucial OSINT Variables ──
# 0=North, 90=East, 180=South, 270=West. If unknown, assume 0 for trail cams.
CAMERA_HEADING_DEG = 0.0  
# The timezone the camera clock was set to (e.g., -7 for PDT, -4 for EDT)
CAMERA_UTC_OFFSET_HOURS = -7.0  

PROMPTS = {
    "vision_combo": """Analyze flora and shadows in this image. 
Assume the top of the image is 0 degrees, the right is 90, the bottom is 180, and the left is 270.
Return ONLY a raw JSON object:
{
    "plant_types": ["list", "of", "species"],
    "vegetation_density": "sparse|moderate|dense",
    "shadow_visible": true|false,
    "relative_shadow_angle": numeric_degree_value_or_null
}""",

    "geology": """Analyze the rock formations and geology. 
Return ONLY a raw JSON object:
{
    "rock_colors": ["color1", "color2"],
    "rock_texture": "smooth|rough|porous|angular|layered",
    "terrain_type": "canyon|plateau|montane|alluvial|basin"
}"""
}

# ── Hardened Helper Functions ─────────────────────────────────────────────────

def force_vram_eviction(model_name: str):
    """Attempts to force VRAM eviction via Ollama API. Fails silently on LM Studio."""
    try:
        requests.post(OLLAMA_EVICT_URL, json={"model": model_name, "keep_alive": 0}, timeout=2)
    except Exception:
        pass

def extract_true_utc(filename: str) -> datetime | None:
    """Extracts local timestamp from filename and applies UTC offset."""
    m = re.search(r"(\d{8})[T_](\d{6})", filename)
    if not m:
        return None
    try:
        local_dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
        true_utc = local_dt.replace(tzinfo=timezone.utc) - timedelta(hours=CAMERA_UTC_OFFSET_HOURS)
        return true_utc
    except ValueError:
        return None

def image_to_base64(image_path: Path, max_size=(1536, 1536)) -> str:
    """Retains high OSINT resolution up to 1536px."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def parse_robust_json(raw_text: str) -> dict:
    """Violently extracts JSON from LLM output, ignoring conversational filler."""
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}

def safe_extract_float(val) -> float | None:
    """Safely extracts the first floating point number from an LLM string."""
    if val is None:
        return None
    try:
        nums = re.findall(r'-?\d+\.?\d*', str(val))
        return float(nums[0]) if nums else None
    except Exception:
        return None

def query_model(model_name: str, prompt: str, image_path: Path | None = None) -> dict:
    """Sends request to local server. Uses OpenAI format without forcing JSON schema."""
    content = [{"type": "text", "text": prompt}]
    
    if image_path:
        # Resize to fit within typical 8k context window token limits
        b64_data = image_to_base64(image_path, max_size=(1024, 1024))
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
        })

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1
        # Removed response_format to prevent HTTP 400 vision backend crashes
    }

    try:
        resp = requests.post(API_BASE_URL, json=payload, timeout=300)
        
        # Print detailed error body if status code is not 200 OK
        if resp.status_code != 200:
            print(f"  [!] HTTP {resp.status_code} Error from LM Studio: {resp.text}", file=sys.stderr)
            return {}
            
        raw_text = resp.json()["choices"][0]["message"]["content"]
        return parse_robust_json(raw_text)
    except Exception as exc:
        print(f"  [!] Query Exception on {model_name}: {exc}", file=sys.stderr)
        return {}

# ── Spatial & Topographical Math ───────────────────────────────────────────

def generate_candidate_lop(dt_utc: datetime, true_shadow_azimuth: float, tolerance: float = 2.0) -> list[dict]:
    """Generates a Line of Position (LOP) using vectorized pvlib calculations for speed."""
    target_sun_azimuth = (true_shadow_azimuth + 180.0) % 360.0
    
    # Create grids
    lats = np.arange(30.0, 49.0, 0.5)
    lons = np.arange(-125.0, -65.0, 0.5)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Flatten grids into 1D arrays
    lat_flat = lat_mesh.flatten()
    lon_flat = lon_mesh.flatten()
    
    times = pd.DatetimeIndex([dt_utc])
    
    # Vectorized calculation: Calculate all 4,500+ coordinates at exactly the same time
    solpos = pvlib.solarposition.get_solarposition(times, lat_flat, lon_flat)
    
    # solpos['azimuth'] will have the same shape as your flattened arrays
    calc_azimuths = solpos['azimuth'].values[0] # Get the array of azimuths for this specific time
    
    # Find all indices where the calculated azimuth is within tolerance
    valid_indices = np.where(np.abs(calc_azimuths - target_sun_azimuth) <= tolerance)[0]
    
    candidates = []
    for idx in valid_indices:
        candidates.append({
            "lat": round(float(lat_flat[idx]), 2), 
            "lon": round(float(lon_flat[idx]), 2)
        })
        
    if not candidates:
        return []
        
    step = max(1, len(candidates) // 5)
    return candidates[::step][:5]

# ── Main Execution Flow ───────────────────────────────────────────────────────

def main():
    if not IMAGES_DIR.exists():
        print(f"[Error] Images directory not found: {IMAGES_DIR}")
        sys.exit(1)

    image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_files:
        print("No valid image files found.")
        sys.exit(1)

    print(f"=== ATLAS v5.2 OSINT Engine (Hardened) ===")
    print(f"Frames: {len(image_files)} | Cam Heading: {CAMERA_HEADING_DEG}° | UTC Offset: {CAMERA_UTC_OFFSET_HOURS}h\n")

    pipeline_data = {}

    # ── PASS 1: Qwen 2.5 VL (Flora & Relative Shadow)
    print(f"[Pass 1/4] Loading '{MODELS['vision_combo']}'...")
    for img_path in image_files:
        dt_utc = extract_true_utc(img_path.name)
        res = query_model(MODELS["vision_combo"], PROMPTS["vision_combo"], img_path)
        
        rel_shadow = safe_extract_float(res.get("relative_shadow_angle"))
        true_azimuth = (CAMERA_HEADING_DEG + rel_shadow) % 360 if rel_shadow is not None else None

        pipeline_data[img_path.name] = {
            "timestamp_utc": dt_utc.isoformat() if dt_utc else None,
            "flora": {
                "plant_types": res.get("plant_types", []),
                "vegetation_density": res.get("vegetation_density")
            },
            "shadow": {
                "shadow_visible": res.get("shadow_visible", False),
                "relative_angle": rel_shadow,
                "true_azimuth_deg": true_azimuth
            }
        }

    force_vram_eviction(MODELS["vision_combo"])

    # ── PASS 2: Pixtral-12B (Geology Analysis)
    print(f"\n[Pass 2/4] Loading '{MODELS['geology']}'...")
    for img_path in image_files:
        res = query_model(MODELS["geology"], PROMPTS["geology"], img_path)
        pipeline_data[img_path.name]["geology"] = res

    force_vram_eviction(MODELS["geology"])

    # ── PASS 3: Deterministic GIS & Solar Math
    print(f"\n[Pass 3/4] Running Geographic Mathematics...")
    for img_name, data in pipeline_data.items():
        shadow_info = data.get("shadow", {})
        ts_str = data.get("timestamp_utc")
        true_az = shadow_info.get("true_azimuth_deg")
        
        if shadow_info.get("shadow_visible") and true_az is not None and ts_str:
            dt_utc = datetime.fromisoformat(ts_str)
            candidates = generate_candidate_lop(dt_utc, true_az)
            
            for cand in candidates:
                cand["elevation_meters"] = fetch_usgs_elevation(cand["lat"], cand["lon"])
                
            data["candidates"] = candidates
        else:
            data["candidates"] = []

    OUTPUT_JSON.write_text(json.dumps(pipeline_data, indent=2), encoding="utf-8")
    print(f"  └─ Spatial calculations saved to {OUTPUT_JSON}")

    # ── PASS 4: DeepSeek-R1 (Data Synthesis)
    print(f"\n[Pass 4/4] Loading '{MODELS['synthesis']}'...")
    synthesis_prompt = f"""You are a master OSINT GIS analyst. Analyze the structured pipeline data:
{json.dumps(pipeline_data, indent=2)}

Cross-reference the botanical species, geology, and distributed USGS candidate elevations.
Synthesize a concise final report identifying the single most likely US ecoregion, terrain zone, and public land boundary matching this data."""

    synthesis_payload = {
        "model": MODELS["synthesis"],
        "messages": [{"role": "user", "content": synthesis_prompt}],
        "temperature": 0.2
    }

    try:
        resp = requests.post(API_BASE_URL, json=synthesis_payload, timeout=300)
        resp.raise_for_status()
        report = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        report = f"[!] Synthesis Error: {exc}"

    SUMMARY_TXT.write_text(report, encoding="utf-8")
    print("\n" + report)

if __name__ == "__main__":
    main()