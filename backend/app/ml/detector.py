"""Cyclone Candidate Detection Module.

Segments convective cloud masses, evaluates spatial compactness,
and measures spiral vector alignment.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from backend.app.schemas.analysis import ConfidenceTier


class CycloneDetector:
    """Baseline morphological and gradient-based cyclone candidate detector."""

    def __init__(self, min_area_ratio: float = 0.05, max_area_ratio: float = 0.95):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def segment_convective_cloud(self, gray_array: np.ndarray) -> np.ndarray:
        """Adaptive thresholding to segment prominent cloud canopy."""
        threshold = max(0.40, float(np.mean(gray_array) + 0.35 * np.std(gray_array)))
        mask = gray_array > threshold
        return mask

    def find_primary_cloud_mass(
        self, mask: np.ndarray
    ) -> Tuple[Optional[Dict[str, int]], float]:
        """Find the bounding box and area ratio of the primary connected convective region."""
        h, w = mask.shape
        total_pixels = h * w
        active_pixels = int(np.sum(mask))

        if active_pixels == 0:
            return None, 0.0

        # Compute bounding box of active convective mass
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

        area_ratio = active_pixels / total_pixels
        bbox = {
            "x_min": cmin,
            "y_min": rmin,
            "x_max": cmax,
            "y_max": rmax,
            "width": cmax - cmin + 1,
            "height": rmax - rmin + 1,
        }
        return bbox, round(area_ratio, 4)

    def calculate_spiral_curvature_metric(self, gray_array: np.ndarray, center: Tuple[int, int]) -> float:
        """Measures gradient tangential alignment with cyclonic spiral flow."""
        h, w = gray_array.shape
        cy, cx = center

        # Spatial gradients
        gy, gx = np.gradient(gray_array)
        grad_mag = np.hypot(gx, gy) + 1e-6

        # Distance grid from center
        y_grid, x_grid = np.ogrid[:h, :w]
        dx = x_grid - cx
        dy = y_grid - cy
        dist = np.hypot(dx, dy) + 1e-6

        # Tangential unit vectors (counter-clockwise cyclonic circulation)
        tx = -dy / dist
        ty = dx / dist

        # Normalized gradient dot product with tangential vector
        tangential_alignment = np.abs((gx / grad_mag) * tx + (gy / grad_mag) * ty)

        # Focus metric within core radius (15 to 150 pixels)
        core_mask = (dist >= 15) & (dist <= min(h, w) // 3)
        if np.any(core_mask):
            score = float(np.mean(tangential_alignment[core_mask]))
        else:
            score = 0.5

        return round(float(np.clip(score, 0.0, 1.0)), 3)

    def detect(self, gray_array: np.ndarray) -> Dict[str, Any]:
        """Execute detection pipeline on normalized grayscale imagery."""
        h, w = gray_array.shape
        cloud_mask = self.segment_convective_cloud(gray_array)
        bbox, area_ratio = self.find_primary_cloud_mass(cloud_mask)

        if not bbox or area_ratio < self.min_area_ratio:
            return {
                "detected": False,
                "confidence": ConfidenceTier.LOW,
                "confidence_score": 0.15,
                "bounding_box": None,
                "convective_area_ratio": area_ratio,
                "cloud_mask": cloud_mask,
                "spiral_index": 0.0,
                "reason": "Convective cloud area below minimum cyclonic threshold.",
            }

        # Approximate centroid for gradient curvature evaluation
        y_indices, x_indices = np.where(cloud_mask)
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))

        spiral_score = self.calculate_spiral_curvature_metric(gray_array, (center_y, center_x))

        # Determine detection confidence
        # Strict zero-fake-positive rule: Ordinary cloud cover or random terrain gradients
        # must never trigger a cyclone detection. Mathematical expectation of random gradients is ~0.637.
        # Genuine detection requires both significant spiral organization (>0.65) and consolidated convective mass.
        if spiral_score >= 0.72 and area_ratio >= 0.15:
            confidence = ConfidenceTier.HIGH
            detected = True
            reason = (
                f"Identified intense organized cyclonic vortex covering {area_ratio * 100:.1f}% "
                f"of frame with high spiral curvature index of {spiral_score:.2f}."
            )
        elif spiral_score >= 0.65 and area_ratio >= 0.12:
            confidence = ConfidenceTier.MEDIUM
            detected = True
            reason = (
                f"Identified developing cyclonic circulation covering {area_ratio * 100:.1f}% "
                f"of frame with spiral curvature index of {spiral_score:.2f}."
            )
        else:
            confidence = ConfidenceTier.LOW
            detected = False
            reason = (
                f"Cloud structures lack coherent rotational vortex symmetry (spiral index: {spiral_score:.2f}, "
                f"convective area: {area_ratio * 100:.1f}%). No active cyclone identified in frame."
            )

        return {
            "detected": detected,
            "confidence": confidence,
            "confidence_score": spiral_score if detected else (1.0 - spiral_score),
            "bounding_box": bbox if detected else None,
            "convective_area_ratio": area_ratio,
            "cloud_mask": cloud_mask,
            "spiral_index": spiral_score,
            "reason": reason,
        }


cyclone_detector = CycloneDetector()

