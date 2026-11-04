# CYCLONE-X: Meteorological Dataset Preparation & Training Guidelines

## 1. Overview & Data Directory Structure

To prepare **CYCLONE-X** for future deep neural network training (e.g. `cyclone_cnn`, `cyclone_vit`, `intensity_dvorak_net`) in Phase 5 and Phase 6, the repository organizes data into the following standard layout:

```
data/
├── raw/            # Untouched original satellite granules (HDF5, NetCDF4, GeoTIFF)
├── processed/      # Normalized, cropped, and calibrated tensors (NumPy .npy, PyTorch .pt)
├── samples/        # Deterministic benchmark samples for testing and reproducible demos
└── labels/         # Ground-truth track coordinates, Dvorak T-numbers, and IMD best tracks
```

---

## 2. Expected Image Formats & Channels

1. **Resolution & Crop Dimensions**:
   - Geostationary imagery: $512 \times 512$ pixel storm-centered crops at $4 \text{ km}$ nominal sub-satellite resolution.
   - Polar / Low Earth Orbit imagery: $0.1^\circ \times 0.1^\circ$ resolution.
2. **Standard Satellite Channels**:
   - **TIR1 (Thermal Infrared 10.8 µm)**: Cloud-top brightness temperatures in Kelvin ($180 \text{ K}$ to $310 \text{ K}$).
   - **WV (Water Vapor 6.7 µm)**: Mid-to-upper tropospheric moisture dynamics ($200 \text{ K}$ to $260 \text{ K}$).
   - **VIS (Visible 0.65 µm)**: Daytime albedo ($0.0$ to $1.0$).
3. **Data Encoding**:
   - Storage format: Single or multi-channel Float32 arrays scaled to $[-1.0, 1.0]$ or $[0.0, 1.0]$.
   - Image formats for ingestion: PNG, JPEG, TIFF.

---

## 3. Ground Truth Label Format

Annotations in `data/labels/` must adhere to standard JSON schema matching `CycloneObservation`:

```json
{
  "storm_id": "BOB-01-2020",
  "storm_name": "AMPHAN",
  "timestamp_utc": "2020-05-18T12:00:00Z",
  "center_latitude": 13.9,
  "center_longitude": 86.5,
  "center_pixel_x": 256.0,
  "center_pixel_y": 256.0,
  "eye_diameter_km": 28.0,
  "classification": "Super Cyclonic Storm",
  "sustained_wind_knots": 140.0,
  "central_pressure_hpa": 907.0,
  "source_bulletin": "IMD Best Track Archive (RSMC New Delhi)",
  "georeferencing_bounds": {
    "lat_min": 5.0,
    "lat_max": 25.0,
    "lon_min": 75.0,
    "lon_max": 95.0
  }
}
```

---

## 4. Preventing Data Leakage & Temporal Splits

### The Spatial-Temporal Autocorrelation Problem
In tropical cyclone meteorology, frames taken 30 minutes or 1 hour apart from the same cyclone are heavily autocorrelated. Standard random $K$-fold cross-validation or uniform random train/test splits causes severe **data leakage**, resulting in artificially inflated validation scores that fail in operational settings.

### Mandatory Partitioning Protocol:
1. **Storm-Independent Splitting**: All frames and granules belonging to a specific cyclone (e.g. Cyclone Amphan or Cyclone Fani) **must** reside entirely within one split (Train, Validation, or Test). Never divide a single cyclone's timeline across train and test sets.
2. **Temporal Out-of-Sample Holdout**:
   - **Training Set**: Historical cyclones from 2000 to 2020 (e.g. ~80% of historical events).
   - **Validation Set**: Cyclones from 2021 to 2022 (e.g. ~10% for hyperparameter tuning and early stopping).
   - **Test Set**: Cyclones from 2023 onwards (e.g. Cyclone Biparjoy, Cyclone Michaung, Cyclone Remal) kept strictly untouched until final verification.

---

## 5. Georeferencing & Spatial Alignment

- Satellite imagery must retain affine transformation parameters or corner geodetic coordinates (`lat_min`, `lat_max`, `lon_min`, `lon_max`).
- Pixel coordinates $(x, y)$ map to $(\text{latitude}, \text{longitude})$ via equirectangular projection:
  $$\text{lat} = \text{lat}_{\max} - \left(\frac{y}{H}\right)(\text{lat}_{\max} - \text{lat}_{\min})$$
  $$\text{lon} = \text{lon}_{\min} + \left(\frac{x}{W}\right)(\text{lon}_{\max} - \text{lon}_{\min})$$
- When georeferencing coordinates are not available, models must operate in pure pixel space and label geographic coordinates as unavailable.

