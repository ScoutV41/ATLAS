
---

# ATLAS

## [MAJOR WIP: EXPECT MISCALCULATIONS AND ERRORS - DO NOT RELY ENTIRELY ON INFO OUTPUT]

ATLAS (Automated Terrain and Luminance Analysis System) is an implementation of Qwen2.5-VL (And various other VLMs) 

A local vision AI pipeline for extracting geolocation data from landscape camera sequences. Originally built for the [Waterfall Hunt](https://waterfallhunt.com) treasure hunt investigation, it is designed to analyze shadow angles, vegetation, and geology across hundreds of timestamped camera frames to estimate longitude, latitude band, and biome.
(V1-V2 use Ollama, Will **not** work with LMS)

| Version | Backend | Notes |
| ---     | ---     | ---   |
| V1-V2   | Ollama  | Single model, shadow geometry only |
| V3.x    | LM Studio | Ensemble system, luminance filtering |

---

## What it does

ATLAS feeds every image in a folder to a local vision model (Tested on Qwen2.5-VL via LM Studio), extracts structured terrain data from each frame, and produces:

* **Shadow angle per hour (UTC)** - circular-mean averaged across all frames.
* **Solar noon estimate** - derived from shortest-shadow frames, averaged per day.
* **Longitude estimate** - calculated from solar noon with Equation of Time correction.
* **Habitat classification** - biome identified from vegetation and geology across all frames.
* **Suggested search corridor** - accounts for crevice/hill shadow bias.

Dark and night frames are automatically filtered out before analysis to prevent model hallucinations and save processing time.
V3.x introduces a multi-model ensemble system and global luminance profiling to filter night frames before inference. 

---

## Requirements

* Python 3.10+
* [LM Studio](https://lmstudio.ai/) OR [Ollama](https://ollama.com/) running locally with a compatible vision model loaded (`Model used:` `qwen2.5vl:7b`).
* A folder of landscape images with UTC timestamps in the filenames (e.g., `20260518T230318Z.png`).
* 
[For V1-2] Install [Ollama](https://ollama.com/download/windows)
```bash
pip install Pillow numpy ollama

```
[For V3+]
Install [LMS](https://lmstudio.ai/download)
```bash
pip install Pillow numpy
```
---

## Usage

**1. Set your images folder**

Open your ATLAS script (e.g., `ATLASV1.py`) and edit the configuration section at the top:

```python
IMAGES_DIR = Path(r"C:\Users\YourName\Desktop\images")

```
**Change models in:**
```python
MODEL_ENDPOINTS = [
    "qwen2.5-vl-7b-instruct",
    "gemma-4-e4b-it"  # <-- Swapped to the short identifier the server expects
]
```
to local models on your machine


**2. Run**

```bash
python atlas.py

```

**3. Check outputs**

| File | Contents |
| --- | --- |
| `output.csv` | Per-image data: shadow angle/length, plants, rocks, volcanic features. |
| `summary.txt` | Solar noon, longitude estimate, habitat classification. |
| `skipped.txt` | Frames skipped (too dark or no timestamp). |

---

## Image filename format

Filenames must contain a UTC timestamp in this exact format:

```text
20260518T230318Z.png

```

Anything before or after the timestamp in the filename is ignored.

---

## Configuration

All tuneable settings are located at the top of the script:

| Variable | Default | Description |
| --- | --- | --- |
| `IMAGES_DIR` | `/images` | Path to your image folder. |
| `MODEL` | `qwen2.5vl:7b` | The local vision model to use via Ollama/LM Studio. |
| `BRIGHTNESS_THRESH` | `20` | Frames below this brightness average (0-255) are skipped. |

> **Tip:** Raise `BRIGHTNESS_THRESH` to `30` if too many twilight/dusk frames slip through the filter and confuse the model.
> **[for V3.3]** There is no brightness threshold you need to manually set, everything is done automatically via luminance curve.


---

## How the solar noon calculation works

1. The vision model estimates shadow length (short/medium/long) for each frame.
2. Frames with the shortest shadows are grouped by calendar date.
3. The temporal midpoint of shortest-shadow frames = apparent solar noon per day.
4. Per-day noons are averaged across all days in the dataset.
5. The **Equation of Time** correction is applied using the Julian day of the dataset midpoint.
6. Longitude = `-(solar_noon_UTC_hours - corrected_prime_meridian_noon) × 15`

> **Note:** Shadow geometry in crevices or near hillsides can bias apparent solar noon earlier than true solar noon, shifting the longitude estimate 2–8° east of the actual location. The summary output includes a corrected search corridor to account for this terrain bias.

---

## Habitat classification

ATLAS classifies the biome by tallying vegetation and geology indicators across all processed frames. Supported biomes currently include:

* Sonoran Desert (SW Arizona / NW Mexico)
* Chihuahuan Desert (S New Mexico / W Texas)
* Mojave Desert (S Nevada / SE California)
* Colorado Plateau / Southern Rockies
* Cascades / Northern Rockies
* Sierra Nevada / Great Basin
* High Desert Shrubland
* Mixed Conifer Forest

The most frequently matched biome across all analyzed frames wins the classification.

---

## Shadow angle output

Shadow angles are averaged using **circular mean** (sin/cos decomposition) rather than an arithmetic mean. This prevents wrap-around errors at the 0°/360° boundary (where 359° and 1° correctly average to 0°, not 180°) and is mathematically required for accurate angular data tracking.

---

## Performance

For a dataset of 600 images on a mid-range GPU, expect the script to take **15-25 minutes** to complete.
*(Tested on an RX 9070 XT: Approximate runtime for a standard batch was 6 minutes and 38 seconds).*

**NOTE: ATLAS V1-V2 will ALWAYS yield significantly faster results due to being limited to Shadow Geometry**

---

## Notes

* **Model Compatibility:** Tested primarily with `qwen2.5vl:7b`. Other vision models may produce inconsistent structured outputs and break the regex parsing.
* **Markdown Resilience:** The `parse_field()` function actively strips markdown bolding, handling models that spontaneously format their outputs (e.g., returning `FIELD: value` instead of `FIELD: value`).
* **Memory Safety:** File handles are explicitly closed via context managers after brightness checking to prevent OS-level handle leaks on Windows during large directory scans.

---

## Origin

Built during a 5-day investigation into the [Waterfall Hunt](https://waterfallhunt.com) a real-world treasure hunt involving over $15k in gold coins and USDC hidden somewhere in the American Southwest. ATLAS was developed to systematically analyze trail camera feeds using local AI rather than relying on manual, frame-by-frame human inspection.

## WHY A normal demo isnt possible
ATLAS Requires running AI Models locally in order to work, I do not have the compute to keep the models running on my machine for a web API.
See a video demonstration [here](https://vimeo.com/1199611593?share=copy&fl=sv&fe=ci)

##  License & Attribution

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. 

### What this means for you:
* **Patent Protection:** Anyone contributing code automatically grants a royalty-free patent license to all users.
* **Copyleft Requirement:** If you fork, modify, or distribute this software locally, you **must** open-source your modifications under this exact same GPLv3 license.
* **No Commercial Lockdown:** You cannot take this ensemble logic and wrap it into a closed-source commercial application.

See the [LICENSE](LICENSE) file for the full legal text.
