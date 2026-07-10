---

# ATLAS

## [MAJOR WIP: EXPECT MISCALCULATIONS AND ERRORS - DO NOT RELY ENTIRELY ON OUTPUT]

ATLAS (Automated Terrain and Luminance Analysis System) is a local vision AI pipeline that estimates geolocation data - longitude, latitude band, and biome - from a sequence of timestamped landscape camera images. It analyzes shadow angles, vegetation, and geology across hundreds of frames to derive a best-guess location.

Originally built for the [Waterfall Hunt](https://waterfallhunt.com) treasure hunt investigation.

| Version | Backend | Notes |
| --- | --- | --- |
| V1–V2 | Ollama | Single model, shadow geometry only (legacy, no longer the recommended path) |
| V3.x | LM Studio | Ensemble system, luminance filtering |
| **V3.3 (Lite)** | **LM Studio** | **Current version - see setup below** |

---

## ⚠️ Before you run this, read this first

**ATLAS requires a vision-language model running locally.** It will not work, and will not fail with a clear error, if no model backend is active. Running it without a VLM loaded will produce meaningless output (random/default coordinates, often landing in the ocean) from the DEMO mode fallback (this lets you simulate ATLAS on weaker hardware).

**You need, in this order:**

1. Install [LM Studio](https://lmstudio.ai/)
2. Download and load a vision-capable model - `qwen2.5-vl-7b-instruct` is what ATLAS is tested against
3. Start the local server in LM Studio (Developer tab → confirm it shows "Running")
4. Only then run `python ATLASV3.3"(Lite)".py`

If you skip step 2 or 3, ATLAS has nothing to extract shadow/terrain data from and the output is not representative of how the tool works.

If your hardware cannot run a local vision model, you can run the **DEMO** version instead, which uses randomly generated coordinates. Any images used in DEMO mode are for interface testing only - the output is not real.

---

## What it actually needs to produce a real estimate

ATLAS cannot guess a location from random or unrelated photos. It needs:

| Requirement | Why |
| --- | --- |
| **40+ images minimum** | Enough data points to average shadow angle and reduce noise |
| **One single physical location** | All images must be the same place - different locations confuse the averaging |
| **Spanning sunrise to sunset (ideally a full day or more)** | Solar noon is derived from the shortest-shadow frames - without daylight range, there's no signal |
| **Consistent shadow-casting objects in frame** (rocks, trees, posts, structures) | The model needs something to estimate shadow angle/length from |
| **Filenames with UTC timestamps**: `YYYYMMDDTHHMMSSZ` (e.g. `20260518T230318Z.png`) | This is how ATLAS derives time-of-day; without it, frames are skipped entirely |

**What will NOT work:**
- A handful of random photos from different places
- Images without the timestamp filename format
- Synthetic/mock data with no real shadow geometry - this will produce nonsense coordinates, which is expected, not a bug

---

## Quick test image-set

[SAMPLE SET HERE](https://drive.google.com/drive/folders/1fp-efQHF-mpdb3wr4wQy-DNNl0TbnCjm?usp=drive_link)

Test with a known-good dataset first before trying your own images, so you can confirm ATLAS is working correctly before troubleshooting your own data.

---

## Setup

**Requirements:**
- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running locally with `qwen2.5-vl-7b-instruct` (or similar vision model) loaded
- A folder of landscape images meeting the requirements above

```bash
pip install Pillow numpy requests
```

**1. Set your images folder**

Open the script and edit the config section at the top:

```python
IMAGES_DIR = Path(r"C:\Users\YourName\Desktop\images")
```

**2. Set your model endpoint**

```python
MODEL_ENDPOINTS = [
    "qwen2.5-vl-7b-instruct"
]
```

> For the absolute lightest possible model, run using MoonDream 2 - the released executable is already set to use MD2. If it returns an error, report an issue.

**3. Run**

```bash
python ATLASV3.3"(Lite)".py
```

**4. Check outputs**

| File | Contents |
| --- | --- |
| `output.csv` | Per-image data: shadow angle/length, plants, rocks, volcanic features |
| `summary.txt` | Solar noon, longitude estimate, habitat classification |
| `skipped.txt` | Frames skipped (too dark or missing timestamp) |

---

## Image filename format

Filenames must contain a UTC timestamp in this exact format:

```
20260518T230318Z.png
```

Anything before or after the timestamp is ignored. Images without a valid timestamp are automatically skipped and logged to `skipped.txt`.

---

## How the estimate is derived (for context, not required reading to test)

1. The vision model estimates shadow length (short/medium/long) per frame.
2. Frames with the shortest shadows are grouped by calendar date - this approximates solar noon.
3. Per-day solar noon values are averaged across the full dataset.
4. An Equation of Time correction is applied using the Julian day of the dataset midpoint.
5. Longitude is calculated from the corrected solar noon offset.
6. Habitat/biome is classified by tallying vegetation and geology indicators across all frames.

Shadow angles use a **circular mean** (sin/cos decomposition, not an arithmetic mean) to avoid wraparound errors at the 0°/360° boundary - where 359° and 1° correctly average to 0°, not 180°.

> **Note:** Shadow geometry in crevices or near hillsides can bias apparent solar noon earlier than true solar noon, shifting the longitude estimate 2–8° east of the actual location. The summary output includes a corrected search corridor to account for this terrain bias.

---

## Habitat classification

ATLAS classifies the biome by tallying vegetation and geology indicators across all processed frames. Supported biomes currently include:

- Sonoran Desert (SW Arizona / NW Mexico)
- Chihuahuan Desert (S New Mexico / W Texas)
- Mojave Desert (S Nevada / SE California)
- Colorado Plateau / Southern Rockies
- Cascades / Northern Rockies
- Sierra Nevada / Great Basin
- High Desert Shrubland
- Mixed Conifer Forest

The most frequently matched biome across all analyzed frames wins the classification.

---

## Performance

For a dataset of 600 images on a mid-range GPU, expect the script to take **15–25 minutes** to complete.

*(V3.3 tested on an RX 9070 XT: approximate runtime for a standard batch was 13 minutes 32 seconds.)*

**Note:** ATLAS V1–V2 will always yield faster results due to being limited to shadow geometry only - but V3.3 is the recommended version going forward.

---

## Known limitations (current WIP state)

- Coordinate accuracy is currently inconsistent - multiple tests were off target by several thousand km due to a known boundary/geometry bug in the longitude derivation, not the inference pipeline itself.
- Shadow-angle estimation can return `unknown` for certain ambiguous vectors.
- Model compatibility is currently strongest with `qwen2.5-vl-7b-instruct` - other vision models may produce inconsistent structured output and break parsing.
- This is a research tool, not a production-ready geolocation system. Treat all output as a rough estimate requiring human verification.

---

## Notes

- **Model Compatibility:** Tested primarily with `qwen2.5-vl-7b-instruct`. Other vision models may produce inconsistent structured outputs and break the regex parsing.
- **Markdown Resilience:** The `parse_field()` function actively strips markdown bolding, handling models that spontaneously format their outputs (e.g., returning bolded `FIELD: value` instead of plain text).
- **Memory Safety:** File handles are explicitly closed via context managers after brightness checking to prevent OS-level handle leaks on Windows during large directory scans.

---

## Questions?

If something isn't working as described above, the most likely cause is either:
1. No model backend running (see the warning at the top)
2. Test images not meeting the dataset requirements above

If you've confirmed both of those and it's still not working, please reach out - happy to help debug.

---

## Origin

Built during a 5-day investigation into the [Waterfall Hunt](https://waterfallhunt.com), a real-world treasure hunt involving over $15k in gold coins and USDC hidden somewhere in the American Southwest. ATLAS was developed to systematically analyze trail camera feeds using local AI rather than relying on manual, frame-by-frame human inspection.

## License & Attribution

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

- **Patent Protection:** Anyone contributing code automatically grants a royalty-free patent license to all users.
- **Copyleft Requirement:** If you fork, modify, or distribute this software, you must open-source your modifications under this exact same GPLv3 license.
- **No Commercial Lockdown:** You cannot take this ensemble logic and wrap it into a closed-source commercial application.

See the [LICENSE](https://github.com/ScoutV41/ATLAS/blob/main/LICENSE.txt) file for the full legal text.
