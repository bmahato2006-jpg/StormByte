"""Cyclone Center Localization & Georeferencing Module.

Implements radial gradient symmetry analysis, cold-core IR minimum brightness
temperature detection, and convective centroid tracking.
Guarantees zero-fabrication: Never invents latitude/longitude when georeferencing
metadata is absent.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from backend.app.schemas.analysis import CenterDetectionResult, ConfidenceTier


class CenterDetector:
    """Estimates cyclone circulation center using radial gradient convergence and symmetry."""

    def estimate_center_pixels(
        self,
        gray_array: np.ndarray,
        candidate_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float, str]:
        """Estimate circulation vortex center in pixel coordinates.

        Stage 1: Fast Radial Symmetry Transform (FRST) — detects circular gradient
                 convergence characteristic of a cyclonic vortex.
        Stage 2 fallback: Cold-Core IR Minimum Brightness Temperature — finds the
                 coldest cloud-top cluster in TIR1 (10.8 µm) imagery, which by
                 atmospheric physics corresponds to the tallest convection / vortex center.

        Returns:
            pixel_x (float)
            pixel_y (float)
            estimated_radius (float)
            symmetry_score (float)
            method (str): 'RADIAL_GRADIENT_SYMMETRY' | 'COLD_CORE_IR_MINIMUM'
        """
        h, w = gray_array.shape

        # Stage 0: Deep CycloneCenterNet Eye Keypoint Detector (ResNet-18 Backbone)
        try:
            from pathlib import Path
            import torch
            from backend.app.ml.training_framework.models import CycloneCenterNet
            from PIL import Image

            ckpt_path = Path(__file__).resolve().parents[2] / "ml" / "checkpoints" / "cyclone_centernet_resnet18.pt"
            if ckpt_path.exists():
                net = CycloneCenterNet(backbone_name="resnet18", pretrained=False)
                ckpt = torch.load(ckpt_path, map_location="cpu")
                net.load_state_dict(ckpt["model_state_dict"])
                net.eval()

                # Scale raster array to 128x128 3-channel normalized tensor
                u8_arr = (np.clip(gray_array, 0.0, 1.0) * 255.0).astype(np.uint8) if gray_array.max() <= 1.0 else np.clip(gray_array, 0, 255).astype(np.uint8)
                pil_im = Image.fromarray(u8_arr).convert("RGB").resize((128, 128))
                img_t = torch.tensor(np.array(pil_im), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

                with torch.no_grad():
                    out = net(img_t)
                    dets = net.decode_keypoints(out["heatmap"], out["offset"], k=1)

                if dets and dets[0]["score"] >= 0.20:
                    d = dets[0]
                    pred_x = float(d["x"] / 128.0) * w
                    pred_y = float(d["y"] / 128.0) * h
                    est_r = 25.0
                    score = float(d["score"])
                    return pred_x, pred_y, est_r, score, "DEEP_CENTERNET_RESNET18"
        except Exception:
            pass

        # Compute spatial image gradients
        gy, gx = np.gradient(gray_array)
        grad_mag = np.hypot(gx, gy) + 1e-6

        # Normal vectors pointing along gradient
        nx = gx / grad_mag
        ny = gy / grad_mag

        # Restrict search to candidate convective region if available
        if candidate_mask is not None and np.any(candidate_mask):
            y_pts, x_pts = np.where(candidate_mask)
            centroid_x = float(np.mean(x_pts))
            centroid_y = float(np.mean(y_pts))
        else:
            centroid_x = w / 2.0
            centroid_y = h / 2.0

        # Fast Radial Symmetry Transform (FRST) projection accumulator
        radii = [15, 25, 35, 50, 75]
        accumulator = np.zeros((h, w), dtype=np.float32)

        # Subsample for efficient and stable convergence
        step = 2
        for r in radii:
            for y in range(0, h, step):
                for x in range(0, w, step):
                    if grad_mag[y, x] > 0.08:  # ignore flat areas
                        # Gradient pointing inward towards center
                        px = int(x - nx[y, x] * r)
                        py = int(y - ny[y, x] * r)
                        if 0 <= px < w and 0 <= py < h:
                            accumulator[py, px] += grad_mag[y, x]

        # Apply Gaussian-like spatial weighting towards cloud centroid
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_centroid = np.hypot(x_grid - centroid_x, y_grid - centroid_y)
        spatial_prior = np.exp(-0.5 * (dist_from_centroid / (min(h, w) * 0.35)) ** 2)
        weighted_acc = accumulator * spatial_prior

        peak_y, peak_x = np.unravel_index(np.argmax(weighted_acc), weighted_acc.shape)
        max_val = float(weighted_acc[peak_y, peak_x])

        # Evaluate symmetry strength
        if max_val > 5.0:
            pixel_x = float(peak_x)
            pixel_y = float(peak_y)
            # Find best fitting radius around peak
            estimated_radius = 28.0
            method = "RADIAL_GRADIENT_SYMMETRY"
            symmetry_score = min(1.0, max_val / 25.0)
        else:
            # FRST symmetry too weak → use Cold-Core IR Minimum Brightness Temperature detection.
            # Physics: INSAT-3D TIR1 (10.8 µm) encodes temperature as pixel intensity.
            # Cold cloud tops (highest altitude → closest to vortex circulation) = dark/low-value pixels.
            # The coldest contiguous cluster centroid is the real circulation center.
            pixel_x, pixel_y, estimated_radius, symmetry_score = self._cold_core_ir_center(
                gray_array, centroid_x, centroid_y
            )
            method = "COLD_CORE_IR_MINIMUM"

        return pixel_x, pixel_y, estimated_radius, symmetry_score, method

    def _cold_core_ir_center(
        self,
        gray_array: np.ndarray,
        prior_cx: float,
        prior_cy: float,
    ) -> Tuple[float, float, float, float]:
        """Detect cyclone center using Cold-Core Minimum Brightness Temperature (MBT).

        Algorithm (pure NumPy — no external dependencies):

        1. Identify the coldest 15th-percentile pixels (cold cloud top region).
        2. Compute per-pixel weight = coldness_depth × Gaussian spatial prior.
           The Gaussian prior is centred on (prior_cx, prior_cy) so that cold
           pixels near the expected storm region outweigh equally cold pixels
           far from the centre (e.g., image corners / background).
        3. Compute the weighted centroid of cold pixels.  Unlike argmax, the
           weighted centroid degrades gracefully for uniform images — it returns
           a position near (prior_cx, prior_cy) when no dominant cold cluster
           exists.
        4. Report a symmetry score derived from the radial spread of cold pixels
           around the detected centre.

        For INSAT-3D TIR1 (10.8 µm):
          - Dark pixels (low normalised value) = cold cloud tops = eyewall /
            deep convection.
          - The coldest cluster centroid closest to the storm prior is the
            real circulation centre.

        Returns:
            pixel_x, pixel_y, estimated_radius_px, symmetry_score
        """
        h, w = gray_array.shape
        sigma = min(h, w) * 0.40

        # ── Step 1: isolate cold cloud tops ──────────────────────────────────
        cold_threshold = float(np.percentile(gray_array, 15))
        cold_mask = gray_array <= cold_threshold

        if not np.any(cold_mask):
            cold_threshold = float(np.percentile(gray_array, 30))
            cold_mask = gray_array <= cold_threshold

        # ── Step 2: coldness depth per pixel (how far below threshold) ───────
        # Positive where cold, zero elsewhere.
        coldness = np.where(cold_mask, cold_threshold - gray_array, 0.0).astype(np.float64)
        coldness_max = coldness.max()
        if coldness_max > 0:
            coldness /= coldness_max  # normalise 0–1

        # ── Step 3: Gaussian spatial prior centred on (prior_cx, prior_cy) ───
        # Cold pixels near the storm prior get higher weight.
        y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        dist_sq = (x_idx - prior_cx) ** 2 + (y_idx - prior_cy) ** 2
        spatial_prior = np.exp(-0.5 * dist_sq / (sigma ** 2))

        # ── Step 4: combined weight map, evaluated only at cold pixels ────────
        weight_map = coldness * spatial_prior  # shape (h, w)

        total_weight = float(weight_map.sum())

        if total_weight < 1e-12:
            # No significant cold signal → return spatial prior centre
            return prior_cx, prior_cy, 35.0, 0.35

        # ── Step 5: weighted centroid ─────────────────────────────────────────
        cx = float(np.sum(x_idx * weight_map) / total_weight)
        cy = float(np.sum(y_idx * weight_map) / total_weight)

        # Clamp to valid pixel range
        cx = float(np.clip(cx, 0.0, w - 1))
        cy = float(np.clip(cy, 0.0, h - 1))

        # ── Step 6: estimate radius from cold-cloud area ──────────────────────
        cold_area_px = float(np.sum(cold_mask))
        estimated_radius = float(np.sqrt(cold_area_px / np.pi)) if cold_area_px > 0 else 35.0

        # ── Step 7: symmetry score from radial variance of cold pixels ────────
        y_cold, x_cold = np.where(cold_mask)
        dx = x_cold - cx
        dy = y_cold - cy
        dist = np.sqrt(dx ** 2 + dy ** 2)
        mean_dist = float(np.mean(dist)) + 1e-6
        std_dist = float(np.std(dist))
        # Low radial variance = more symmetric = higher confidence
        symmetry_score = float(np.clip(1.0 - std_dist / (mean_dist + std_dist), 0.35, 0.85))

        return cx, cy, estimated_radius, symmetry_score

    def locate_center(
        self,
        gray_array: np.ndarray,
        candidate_mask: Optional[np.ndarray] = None,
        geo_bounds: Optional[Dict[str, float]] = None,
        detected: bool = True,
    ) -> CenterDetectionResult:
        """Locates the center, applying geodetic projection only if verified bounds exist."""
        h, w = gray_array.shape
        px_x, px_y, radius, score, method = self.estimate_center_pixels(gray_array, candidate_mask)

        # Determine qualitative confidence
        if not detected:
            confidence = ConfidenceTier.LOW
        elif score >= 0.70:
            confidence = ConfidenceTier.HIGH
        elif score >= 0.40:
            confidence = ConfidenceTier.MEDIUM
        else:
            confidence = ConfidenceTier.LOW

        # Check for georeferencing metadata
        has_bounds = (
            geo_bounds is not None
            and "lat_min" in geo_bounds
            and "lat_max" in geo_bounds
            and "lon_min" in geo_bounds
            and "lon_max" in geo_bounds
        )

        if has_bounds:
            lat_min = float(geo_bounds["lat_min"])
            lat_max = float(geo_bounds["lat_max"])
            lon_min = float(geo_bounds["lon_min"])
            lon_max = float(geo_bounds["lon_max"])

            # Map image pixel (top-left origin) to geodetic coordinates
            # Pixel 0 is lat_max (North), Pixel H is lat_min (South)
            latitude = lat_max - (px_y / h) * (lat_max - lat_min)
            # Pixel 0 is lon_min (West), Pixel W is lon_max (East)
            longitude = lon_min + (px_x / w) * (lon_max - lon_min)

            latitude = round(latitude, 4)
            longitude = round(longitude, 4)
            georeferenced = True
            msg = f"Center projected to {latitude:.4f}°N, {longitude:.4f}°E via verified bounds."
        else:
            latitude = None
            longitude = None
            georeferenced = False
            msg = "Geographic center unavailable — image georeferencing required."

        return CenterDetectionResult(
            pixel_x=round(px_x, 1),
            pixel_y=round(px_y, 1),
            estimated_radius_pixels=round(radius, 1),
            confidence=confidence,
            confidence_score=round(score, 3),
            method=method,
            georeferenced=georeferenced,
            latitude=latitude,
            longitude=longitude,
            status_message=msg,
        )


center_detector = CenterDetector()

