# CYCLONE-X: Phase 4 — Satellite Image Analysis, Cyclone Detection & Center Localization

## 1. Executive Summary & Objective

**Phase 4** implements the computer-vision and meteorological feature-extraction foundation for the **CYCLONE-X** platform. It provides automated detection, bounding box estimation, radial symmetry center localization, morphological intensity classification, and explainable visual attribution overlays for geostationary and polar meteorological satellite imagery.

### Zero-Fabrication Scientific Integrity Guarantees
1. **No Simulated Accuracy**: The baseline detector is transparently labeled as an *experimental computer-vision baseline — not operational meteorological guidance*.
2. **No Invented Intensity**: In the absence of an empirically validated and trained neural network, sustained wind speed and central pressure are **not** simulated; the system truthfully reports *"Intensity estimation unavailable — trained intensity model required."*
3. **No Invented Coordinates**: When an uploaded raster lacks spatial calibration bounds, the system returns pixel coordinates $(x, y)$ and states *"Geographic center unavailable — image georeferencing required."*
4. **Qualitative Confidence Tiers**: Confidence is categorized into **HIGH**, **MEDIUM**, and **LOW** based on deterministic geometric convergence metrics rather than uncalibrated floating-point percentages.

---

## 2. Computer-Vision Pipeline Architecture

```
                       [ RAW SATELLITE IMAGE ]
                                  │
                                  ▼
                     [ 1. Image Preprocessing ]
                     • Format normalization to RGB / Grayscale
                     • Contrast enhancement (Histogram Equalization)
                     • Laplacian variance sharpness metric
                     • Dynamic range & SNR estimation (IQA 0-100)
                                  │
                                  ▼
                 [ 2. Cyclone Candidate Detection ]
                 • Convective cloud shield adaptive thresholding
                 • Morphological closing & bounding box calculation
                 • Tangential spiral gradient curvature index
                                  │
                                  ▼
               [ 3. Circulation Center Localization ]
               • Fast Radial Symmetry Transform (FRST) accumulator
               • Convective core centroid fallback
               • Core radius estimation (pixels)
               • Geodetic projection (if bounds present)
                                  │
                                  ▼
                 [ 4. Structural Classification ]
                 • Compactness, symmetry & spiral curvature analysis
                 • 7 IMD/WMO structural categories
                 • Mandatory experimental research disclaimer
                                  │
                                  ▼
                    [ 5. Intensity Verification ]
                    • Checks ModelRegistry for trained checkpoints
                    • Returns honest "Unavailable" status
                                  │
                                  ▼
               [ 6. Explainable AI & Visual Overlays ]
               • Generates transparent RGBA annotation overlay
               • Draws bounding box, center crosshair, core circle
               • Generates deterministic natural-language reasoning
                                  │
                                  ▼
                [ 7. Provenance & Audit Registration ]
                • SHA-256 cryptographic image hashing
                • Logs run in ProvenanceTracker with UTC timestamps
```

---

## 3. Algorithm Specifications

### A. Image Quality Assessment (IQA)
The sharpness metric evaluates high-frequency energy across convective edges using the discrete 2D Laplacian operator:
$$\Delta I(x, y) = I(x+1, y) + I(x-1, y) + I(x, y+1) + I(x, y-1) - 4 I(x, y)$$
$$\text{Sharpness} = \text{Var}(\Delta I)$$

### B. Spiral Curvature Metric
Measures cyclonic rotational alignment between spatial intensity gradients $\vec{\nabla} I = (g_x, g_y)$ and theoretical counter-clockwise tangential unit vectors $\vec{t} = \left(-\frac{y - c_y}{r}, \frac{x - c_x}{r}\right)$:
$$\text{Spiral Index} = \frac{1}{N} \sum_{i \in \text{Core}} \left| \frac{\vec{\nabla} I_i}{\|\vec{\nabla} I_i\|} \cdot \vec{t}_i \right|$$
A value $\ge 0.65$ indicates coherent spiral banding characteristic of an organized tropical cyclone.

### C. Fast Radial Symmetry Center Detection
For radius set $\mathcal{R} = \{15, 25, 35, 50, 75\}$, gradient vectors project inward along the surface normal:
$$\vec{p} = \vec{x} - \frac{\vec{\nabla} I}{\|\vec{\nabla} I\|} \cdot r$$
Accumulation peaks identify circulation focal points, further regularized by a Gaussian spatial prior centered on the primary convective cloud centroid.

---

## 4. Model Registry Architecture

The platform catalogs registered computer vision algorithms via `ModelRegistry` (`backend/app/ml/model_registry.py`):

| Model Name | Version | Framework | Training Status | Description |
| :--- | :---: | :---: | :---: | :--- |
| **`baseline_cv`** | 0.4.0 | NumPy / Pillow | **ACTIVE** | Morphological cloud canopy segmentation with radial symmetry center localization. |
| **`cyclone_cnn`** | 1.0.0-alpha | PyTorch 2.14 | `NOT_TRAINED` | Deep 2D Convolutional Neural Network for automated eye heatmap regression. |
| **`cyclone_vit`** | 1.0.0-alpha | PyTorch 2.14 | `NOT_TRAINED` | Vision Transformer utilizing global self-attention across wide-area satellite granules. |
| **`intensity_dvorak_net`** | 1.0.0-alpha | PyTorch 2.14 | `NOT_TRAINED` | Objective Dvorak T-number deep regression network. |

---

## 5. API Reference

- `POST /api/analysis/satellite`: Uploads raster, executes the 7-stage computer vision pipeline, and returns structured analysis results.
- `GET /api/analysis/{analysis_id}`: Retrieves cached analysis record and metrics.
- `GET /api/analysis/{analysis_id}/explanation`: Returns XAI reasoning and salient coordinates.
- `GET /api/analysis/{analysis_id}/overlay`: Streams transparent PNG detection overlay.
- `GET /api/analysis/models`: Returns catalog of active and untrained model architectures.

