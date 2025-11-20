#!/usr/bin/env python3
"""
merge_zero_certs.py

Merge multiple zero-certificate JSON files (1..1000) into a single file.

Features:
- Handles different JSON formats:
    { "metadata": {...}, "zeros": [ ... ] }
    { "zeros": [ ... ] }
    [ ... ]  (raw list of zeros)
- For overlapping zero_index values, **later files override earlier** ones.
- Outputs a unified file in the standard "metadata + zeros" format.
- Detects gaps in zero indices
- Computes overall statistics

Usage:
    python merge_zero_certs.py zeros_*.json
    python merge_zero_certs.py --output merged.json zeros_1_to_100.json zeros_101_to_200.json
    python merge_zero_certs.py zeros_*.json --check-gaps
"""

import argparse
import json
import os
from typing import Any, Dict, List


def load_zero_list(path: str) -> List[Dict[str, Any]]:
    """
    Load a certificate JSON file and return the list of zero dicts.

    Supports:
      - {"metadata": {...}, "zeros": [ ... ]}
      - {"zeros": [ ... ]}
      - [ ... ]  (raw list of zeros)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "zeros" in data:
        zeros = data["zeros"]
    elif isinstance(data, list):
        zeros = data
    else:
        raise ValueError(f"{path}: Unrecognized JSON structure (no 'zeros' list)")

    if not isinstance(zeros, list):
        raise ValueError(f"{path}: 'zeros' is not a list")

    return zeros


def collect_zeros(file_paths: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Load zeros from all files and merge into a single dict keyed by zero_index.

    Later files override earlier ones for the same zero_index.
    """
    merged: Dict[int, Dict[str, Any]] = {}
    file_stats = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"⚠️  [WARN] File not found, skipping: {path}")
            continue

        print(f"📂 [LOAD] {path}")
        zeros = load_zero_list(path)
        
        # Get index range for this file
        if zeros:
            indices = [int(z.get("zero_index", 0)) for z in zeros]
            file_range = f"{min(indices)}-{max(indices)}"
        else:
            file_range = "empty"
        
        print(f"       {len(zeros)} zeros (indices: {file_range})")
        file_stats.append({
            'file': os.path.basename(path),
            'count': len(zeros),
            'range': file_range,
        })

        overrides = 0
        for z in zeros:
            idx = z.get("zero_index")
            if idx is None:
                raise ValueError(f"{path}: zero entry missing 'zero_index': {z}")
            try:
                idx_int = int(idx)
            except Exception as e:
                raise ValueError(f"{path}: invalid zero_index {idx!r}: {e}")

            if idx_int in merged:
                overrides += 1
            merged[idx_int] = z
        
        if overrides > 0:
            print(f"       ⚠️  {overrides} overrides (duplicates replaced)")

    return merged, file_stats


def compute_metadata(zeros: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute metadata summary from the merged zeros.
    """
    if not zeros:
        return {
            "start_index": None,
            "end_index": None,
            "planned_end_index": None,
            "certified_count": 0,
            "krawczyk_passed": 0,
            "winding_ok": 0,
            "fully_certified": 0,
            "precision": "mixed",
            "mode": "merged",
            "t_range": [None, None],
        }

    indices = [int(z["zero_index"]) for z in zeros]
    t_vals = [float(z["approx_zero"]["t"]) for z in zeros]

    start_index = min(indices)
    end_index = max(indices)

    k_pass = 0
    w_ok = 0
    fully_cert = 0
    
    for z in zeros:
        w = z.get("winding_numbers", {})
        k = z.get("krawczyk", {})
        
        wA_int = int(round(w.get("wA_int", 0)))
        wB_int = int(round(w.get("wB_int", 0)))
        success = bool(k.get("success", False))
        beta = k.get("beta", float("inf"))
        rho = k.get("rho", float("inf"))
        r_box = k.get("r_box", 0.035)

        if success:
            k_pass += 1
        if wA_int == 1 and wB_int == 1:
            w_ok += 1
        
        # Check if fully certified (strict criteria)
        if (wA_int == 1 and wB_int == 1 and success and 
            beta < 1.0 and beta <= 0.90 and rho <= r_box):
            fully_cert += 1

    meta = {
        "start_index": start_index,
        "end_index": end_index,
        "planned_end_index": end_index,
        "certified_count": len(zeros),
        "krawczyk_passed": k_pass,
        "winding_ok": w_ok,
        "fully_certified": fully_cert,
        "precision": "mixed",  # Different files may have different precisions
        "mode": "merged",
        "t_range": [min(t_vals), max(t_vals)],
    }
    return meta


def check_gaps(indices: List[int]) -> List[int]:
    """Find missing indices in a sorted list."""
    if not indices:
        return []
    
    missing = []
    for i in range(indices[0], indices[-1] + 1):
        if i not in indices:
            missing.append(i)
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple zero-certificate JSONs into one unified file."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input JSON certificate files (e.g. zeros_*.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="zeros_1_to_1000_merged.json",
        help="Output JSON filename (default: zeros_1_to_1000_merged.json)",
    )
    parser.add_argument(
        "--check-gaps",
        action="store_true",
        help="Report missing zero indices in detail",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MERGING ZERO CERTIFICATES")
    print("=" * 70)
    print(f"Input files: {len(args.files)}")
    print(f"Output: {args.output}\n")

    # Collect all zeros
    merged_dict, file_stats = collect_zeros(args.files)
    
    if not merged_dict:
        print("\n❌ [ERROR] No zeros loaded. Check file paths.")
        return

    # Sort by zero_index
    all_indices = sorted(merged_dict.keys())
    zeros_sorted = [merged_dict[i] for i in all_indices]

    # Check for gaps
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"Total zeros merged:     {len(zeros_sorted)}")
    print(f"Index range:            {all_indices[0]} → {all_indices[-1]}")
    print(f"Expected zeros in range:{all_indices[-1] - all_indices[0] + 1}")
    
    missing = check_gaps(all_indices)
    
    if missing:
        print(f"\n⚠️  WARNING: {len(missing)} missing zero indices")
        if args.check_gaps:
            print(f"Missing indices: {missing}")
        else:
            print(f"First 10 missing: {missing[:10]}")
            if len(missing) > 10:
                print(f"... and {len(missing) - 10} more")
            print("(Use --check-gaps to see all)")
    else:
        print("\n✓ No gaps! All indices present in range.")

    # Compute metadata
    metadata = compute_metadata(zeros_sorted)

    # File statistics
    print("\n" + "=" * 70)
    print("SOURCE FILE STATISTICS")
    print("=" * 70)
    for stat in file_stats:
        print(f"  {stat['file']:40s} {stat['count']:4d} zeros  ({stat['range']})")

    # Quality statistics
    print("\n" + "=" * 70)
    print("QUALITY STATISTICS")
    print("=" * 70)
    print(f"Krawczyk passed:    {metadata['krawczyk_passed']:4d} / {metadata['certified_count']:4d}  "
          f"({100*metadata['krawczyk_passed']/metadata['certified_count']:.1f}%)")
    print(f"Winding OK:         {metadata['winding_ok']:4d} / {metadata['certified_count']:4d}  "
          f"({100*metadata['winding_ok']/metadata['certified_count']:.1f}%)")
    print(f"Fully certified:    {metadata['fully_certified']:4d} / {metadata['certified_count']:4d}  "
          f"({100*metadata['fully_certified']/metadata['certified_count']:.1f}%)")
    print(f"t-value range:      [{metadata['t_range'][0]:.3f}, {metadata['t_range'][1]:.3f}]")

    # Build output
    output_obj = {
        "metadata": metadata,
        "zeros": zeros_sorted,
    }

    # Save
    print("\n" + "=" * 70)
    print("SAVING MERGED FILE")
    print("=" * 70)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)
    
    # Calculate file size
    file_size = os.path.getsize(args.output)
    size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
    
    print(f"✓ Saved: {args.output}")
    print(f"  Size: {size_str}")
    print(f"  Zeros: {len(zeros_sorted)}")
    print(f"  Index range: {metadata['start_index']}–{metadata['end_index']}")
    
    if missing:
        print(f"\n⚠️  Note: {len(missing)} indices missing from merged file")
        print("  Consider certifying these gaps to complete the set")
    else:
        print(f"\n✓ Complete! All zeros from {metadata['start_index']} to {metadata['end_index']} present")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
