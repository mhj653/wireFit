"""Frame-to-frame ID tracking for detected wires."""

import numpy as np


class WireTracker:
    def __init__(self):
        self._next_id = 1
        self._previous = {}

    def reset(self):
        self._next_id = 1
        self._previous.clear()

    def update(self, wires, max_distance_px=80.0):
        previous = dict(self._previous)
        used = set()
        for wire in wires:
            point = np.asarray(wire.representative_uv, dtype=np.float32)
            best_id = None
            best_distance = None
            for track_id, old_point in previous.items():
                if track_id in used:
                    continue
                distance = float(np.linalg.norm(point - old_point))
                if distance <= max_distance_px and (
                    best_distance is None or distance < best_distance
                ):
                    best_id = track_id
                    best_distance = distance
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            wire.tracking_id = best_id
            used.add(best_id)
        self._previous = {
            wire.tracking_id: np.asarray(wire.representative_uv, dtype=np.float32)
            for wire in wires
            if wire.tracking_id is not None
        }
        return wires
