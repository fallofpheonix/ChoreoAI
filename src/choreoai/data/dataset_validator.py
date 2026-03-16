"""
dataset_validator.py — Tools for verifying dataset integrity.
"""

import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def validate_poses(poses_path: Path, expected_joints: int = 17):
    """Check for corruption and shape consistency."""
    try:
        arr = np.load(poses_path)
        if not np.isfinite(arr).all():
            return False, "Contains non-finite values (NaN/Inf)"
        if arr.ndim != 3 or arr.shape[1] != expected_joints or arr.shape[2] != 3:
            return False, f"Invalid shape: {arr.shape}, expected (T, {expected_joints}, 3)"
        return True, ""
    except Exception as e:
        return False, str(e)

def run_full_validation(manifest_path: Path):
    """Validate all entries in the manifest."""
    with open(manifest_path) as f:
        entries = json.load(f)
    
    report = {
        "total": len(entries),
        "valid": 0,
        "invalid": 0,
        "details": []
    }
    
    base_dir = manifest_path.parent
    for entry in entries:
        poses_path = base_dir / entry["path"] / "poses.npy"
        is_valid, msg = validate_poses(poses_path)
        if is_valid:
            report["valid"] += 1
        else:
            report["invalid"] += 1
            report["details"].append({"id": entry["path"], "error": msg})
            
    return report

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        res = run_full_validation(Path(sys.argv[1]))
        print(json.dumps(res, indent=2))
