# ATLAS-V2
ATLAS is a implementation of Qwen2.5-VL for GeoSearching on a large scale.


### Automated Terrain and Luminance Analysis System

A local vision AI pipeline for extracting geolocation data from trail camera image sequences. Built for the [Waterfall Hunt](https://waterfallhunt.com) treasure hunt investigation designed to analyze shadow angles, vegetation, and geology across hundreds of timestamped cam frames to estimate longitude, latitude band, and biome.

---

## What it does

ATLAS feeds every image in a folder to a local vision model (Qwen2.5-VL via Ollama), extracts structured terrain data from each frame, and produces:

- **Shadow angle per hour (UTC)** -  circular-mean averaged across all frames
- **Solar noon estimate** - derived from shortest-shadow frames, averaged per day
- **Longitude estimate** - calculated from solar noon with Equation of Time correction
- **Habitat classification** - biome identified from vegetation and geology across all frames
- **Suggested search corridor** - accounts for crevice/hill shadow bias

Dark and night frames are automatically filtered out before analysis.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `qwen2.5vl:7b` pulled
- A folder of trail camera images with UTC timestamps in filenames

```
pip install ollama Pillow numpy
ollama pull qwen2.5vl:7b
```

---

## Usage

**1. Set your images folder**

Open `ATLASV2.py` and edit line 24:
```python
IMAGES_DIR = Path(r"C:\Users\YourName\Desktop\camimages")
```

**2. Run**
```bash
python ATLASV2.py
```

**3. Check outputs**

| File | Contents |
|------|----------|
| `output.csv` | Per-image data: shadow angle/length, plants, rocks, volcanic features |
| `summary.txt` | Solar noon, longitude estimate, habitat classification |
| `skipped.txt` | Frames skipped (too dark or no timestamp) |

---

## Image filename format

Filenames must contain a UTC timestamp in this format:
```
20260518T230318Z.png
```
Anything before or after the timestamp is ignored.

---

## Configuration

All tuneable settings are at the top of the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGES_DIR` | `/images` | Path to your image folder |
| `MODEL` | `qwen2.5vl:7b` | Ollama vision model to use |
| `BRIGHTNESS_THRESH` | `20` | Frames below this brightness (0-255) are skipped |

Raise `BRIGHTNESS_THRESH` to `30` if too many twilight frames slip through the filter.

---

## How the solar noon calculation works

1. Qwen estimates shadow length (short/medium/long) for each frame
2. Frames with the shortest shadows are grouped by calendar date
3. The temporal midpoint of shortest-shadow frames = apparent solar noon per day
4. Per-day noons are averaged across all days in the dataset
5. The **Equation of Time** correction is applied using the Julian day of the dataset midpoint
6. Longitude = `-(solar_noon_UTC_hours - corrected_prime_meridian_noon) × 15`

> Note: Shadow geometry in crevices or near hillsides can bias apparent solar noon earlier than true solar noon, shifting the longitude estimate 2–8° east of the actual location. The summary output includes a corrected search corridor to account for this.

---

## Habitat classification

ATLAS classifies the biome by tallying vegetation and geology indicators across all processed frames. Supported biomes:

- Sonoran Desert (SW Arizona / NW Mexico)
- Chihuahuan Desert (S New Mexico / W Texas)
- Mojave Desert (S Nevada / SE California)
- Colorado Plateau / Southern Rockies
- Cascades / Northern Rockies
- Sierra Nevada / Great Basin
- High Desert Shrubland
- Mixed Conifer Forest

The most frequently matched biome across all frames wins.

---

## Shadow angle output

Shadow angles are averaged using **circular mean** (sin/cos decomposition) rather than arithmetic mean — this prevents wrap-around errors at the 0°/360° boundary and is mathematically correct for angular data.

---

## Performance

Runtime depends on your GPU and Ollama's ROCm/CUDA support:

| Setup | Approx. speed |
|-------|--------------|
| NVIDIA GPU (CUDA) | ~3–8s per image |
| AMD GPU (ROCm) | ~5–15s per image |
| CPU only | ~30–60s per image |

For 600 images on a mid-range GPU expect **1–3 hours** total.

---

## Notes

- Tested with `qwen2.5vl:7b` - other vision models may produce inconsistent structured output
- The markdown bold stripping in `parse_field()` handles models that randomly bold their output (`**FIELD**: value`)
- File handles are explicitly closed after brightness checking to prevent handle leaks on Windows

---

## Origin

Built during a 5-day investigation into the [Waterfall Hunt](https://waterfallhunt.com) - a real-world treasure hunt involving $14k+ in gold coins and USDC hidden somewhere in the American Southwest. ATLAS was developed to systematically analyze trail camera feeds using local AI rather than manual frame-by-frame inspection.

