"""
zero_analysis_and_scaling.py

Next-step tooling for your zeta zero certifier:

1. Load and analyze existing zero certificates (JSON files).
2. Drive batch certification for higher ranges via unified_zeta_framework_v2.4_final.
3. Compute spacing statistics and compare to GUE Wigner surmise.

Dependencies:
    - numpy
    - matplotlib
    
Usage:
    python zero_analysis_and_scaling.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np

# Optional matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")

# -------------------------------------------------------------------
# 1. Data structures and loaders
# -------------------------------------------------------------------

@dataclass
class ZeroCert:
    index: int
    t: float
    min_mod: float
    beta: float
    rho: float
    r_box: float
    k_success: bool


def load_zero_certificates(file_path: str) -> List[ZeroCert]:
    """
    Load zero certificates from a JSON file produced by your certifier.

    Assumes JSON structure like:
    {
      "metadata": {...},
      "zeros": [
        {
          "zero_index": ...,
          "approx_zero": {"sigma": 0.5, "t": ...},
          "modulus_bounds": {"min_abs_zeta_on_contour": ...},
          "krawczyk": {"beta": ..., "alpha": ..., "rho": ..., "r_box": ..., "success": true},
          ...
        },
        ...
      ]
    }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zeros_raw = data.get("zeros", [])
    out: List[ZeroCert] = []
    for z in zeros_raw:
        idx = z["zero_index"]
        t = z["approx_zero"]["t"]
        min_mod = z["modulus_bounds"]["min_abs_zeta_on_contour"]
        k = z["krawczyk"]
        beta = k["beta"]
        rho = k["rho"]
        r_box = k["r_box"]
        success = bool(k["success"])
        out.append(ZeroCert(
            index=idx,
            t=t,
            min_mod=min_mod,
            beta=beta,
            rho=rho,
            r_box=r_box,
            k_success=success,
        ))
    return out


def load_multiple_cert_files(file_paths: List[str]) -> List[ZeroCert]:
    """
    Load and concatenate ZeroCert entries from multiple JSON files.
    """
    all_zeros: List[ZeroCert] = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"WARNING: file not found: {path}")
            continue
        print(f"Loading: {path}")
        certs = load_zero_certificates(path)
        all_zeros.extend(certs)
        print(f"  Loaded {len(certs)} zeros")
    
    # sort by index for consistency
    all_zeros.sort(key=lambda z: z.index)
    return all_zeros


# -------------------------------------------------------------------
# 2. Analysis of method performance / stability
# -------------------------------------------------------------------

def summarize_certificates(zeros: List[ZeroCert]) -> Dict[str, Any]:
    """
    Compute summary statistics over a set of certified zeros.
    """
    if not zeros:
        return {}

    min_mods = np.array([z.min_mod for z in zeros], dtype=float)
    betas = np.array([z.beta for z in zeros], dtype=float)
    rhos = np.array([z.rho for z in zeros], dtype=float)
    r_boxes = np.array([z.r_box for z in zeros], dtype=float)
    ks = np.array([1.0 if z.k_success else 0.0 for z in zeros], dtype=float)

    # Filter out infinities for better statistics
    finite_betas = betas[np.isfinite(betas)]
    finite_rhos = rhos[np.isfinite(rhos)]

    stats = {
        "count": len(zeros),
        "min_mod": {
            "min": float(min_mods.min()),
            "max": float(min_mods.max()),
            "mean": float(min_mods.mean()),
            "median": float(np.median(min_mods)),
            "std": float(min_mods.std()),
        },
        "beta": {
            "min": float(finite_betas.min()) if len(finite_betas) > 0 else float('inf'),
            "max": float(finite_betas.max()) if len(finite_betas) > 0 else float('inf'),
            "mean": float(finite_betas.mean()) if len(finite_betas) > 0 else float('inf'),
            "median": float(np.median(finite_betas)) if len(finite_betas) > 0 else float('inf'),
            "finite_count": len(finite_betas),
        },
        "rho_over_r": {
            "min": float((finite_rhos / r_boxes[:len(finite_rhos)]).min()) if len(finite_rhos) > 0 else float('inf'),
            "max": float((finite_rhos / r_boxes[:len(finite_rhos)]).max()) if len(finite_rhos) > 0 else float('inf'),
            "mean": float((finite_rhos / r_boxes[:len(finite_rhos)]).mean()) if len(finite_rhos) > 0 else float('inf'),
        },
        "krawczyk_success_rate": float(ks.mean()),
        "index_range": (zeros[0].index, zeros[-1].index),
        "t_range": (float(min(z.t for z in zeros)), float(max(z.t for z in zeros))),
    }
    return stats


def print_summary(stats: Dict[str, Any]) -> None:
    """
    Pretty-print the summary statistics.
    """
    if not stats:
        print("No zeros to summarize.")
        return

    print("=" * 70)
    print("CERTIFICATE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Zero count:          {stats['count']}")
    print(f"Index range:         {stats['index_range'][0]}..{stats['index_range'][1]}")
    print(f"t range:             {stats['t_range'][0]:.3f} .. {stats['t_range'][1]:.3f}")
    print("-" * 70)
    print("min |ζ| on contour:")
    mm = stats["min_mod"]
    print(f"    min:   {mm['min']:.3e}")
    print(f"    max:   {mm['max']:.3e}")
    print(f"    mean:  {mm['mean']:.3e}")
    print(f"    median:{mm['median']:.3e}")
    print(f"    std:   {mm['std']:.3e}")
    print("-" * 70)
    print("Krawczyk β:")
    b = stats["beta"]
    print(f"    min:   {b['min']:.3e}")
    print(f"    max:   {b['max']:.3e}")
    print(f"    mean:  {b['mean']:.3e}")
    print(f"    median:{b['median']:.3e}")
    print(f"    finite:{b['finite_count']}/{stats['count']}")
    print("-" * 70)
    print("ρ / r_box:")
    rratio = stats["rho_over_r"]
    print(f"    min:   {rratio['min']:.3e}")
    print(f"    max:   {rratio['max']:.3e}")
    print(f"    mean:  {rratio['mean']:.3e}")
    print("-" * 70)
    print(f"Krawczyk success rate: {stats['krawczyk_success_rate']*100.0:.1f}%")
    print("=" * 70)


# -------------------------------------------------------------------
# 3. Spacing analysis and GUE comparison
# -------------------------------------------------------------------

def extract_t_values(zeros: List[ZeroCert]) -> np.ndarray:
    """
    Extract t-values sorted by index.
    """
    zeros_sorted = sorted(zeros, key=lambda z: z.index)
    return np.array([z.t for z in zeros_sorted], dtype=float)


def compute_normalized_spacings(t_vals: np.ndarray) -> np.ndarray:
    """
    Compute normalized nearest-neighbor spacings:

        s_i = (t_{i+1} - t_i) / mean_spacing

    for i = 0 .. N-2.
    """
    if t_vals.size < 2:
        return np.array([], dtype=float)

    diffs = np.diff(t_vals)
    mean_spacing = diffs.mean()
    if mean_spacing <= 0:
        return np.array([], dtype=float)
    return diffs / mean_spacing


def wigner_surmise_gue(s: np.ndarray) -> np.ndarray:
    """
    Wigner surmise PDF for GUE spacing (approximate):

        p(s) = (32 / π^2) s^2 exp(-4 s^2 / π)
    """
    return (32.0 / (math.pi**2)) * s**2 * np.exp(-4.0 * s**2 / math.pi)


def plot_spacing_histogram(spacings: np.ndarray,
                           bins: int = 50,
                           x_max: float = 4.0,
                           show: bool = True,
                           save_path: str | None = None) -> None:
    """
    Plot normalized spacing histogram and overlay GUE Wigner surmise.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - cannot plot")
        return
    
    if spacings.size == 0:
        print("No spacings to plot.")
        return

    s = spacings[(spacings >= 0) & (spacings <= x_max)]
    if s.size == 0:
        print("No spacings in the chosen range.")
        return

    # Histogram
    counts, bin_edges = np.histogram(s, bins=bins, range=(0, x_max), density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # GUE curve
    s_grid = np.linspace(0, x_max, 400)
    p_gue = wigner_surmise_gue(s_grid)

    plt.figure(figsize=(10, 6))
    plt.bar(centers, counts, width=(x_max / bins), alpha=0.6, label="Empirical spacings", color='steelblue')
    plt.plot(s_grid, p_gue, "r-", linewidth=2.5, label="GUE Wigner surmise")
    plt.xlabel("Normalized spacing s", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Nearest-neighbor spacing distribution vs GUE", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved spacing histogram to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compute_spacing_statistics(spacings: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical measures of the spacing distribution.
    """
    if spacings.size == 0:
        return {}
    
    return {
        "mean": float(spacings.mean()),
        "std": float(spacings.std()),
        "min": float(spacings.min()),
        "max": float(spacings.max()),
        "median": float(np.median(spacings)),
        "count": len(spacings),
    }


# -------------------------------------------------------------------
# 4. Scaling driver – hook into unified_zeta_framework
# -------------------------------------------------------------------

def batch_certify_with_unified(start_n: int, 
                               end_n: int, 
                               script_path: str = "unified_zeta_framework_v2.4_final") -> Dict[str, Any]:
    """
    Call certify_zero_range(start_n, end_n) from the unified framework.

    Usage example:
        from zero_analysis_and_scaling import batch_certify_with_unified
        batch_certify_with_unified(201, 300)

    Args:
        start_n: First zero to certify
        end_n: Last zero to certify
        script_path: Path to framework module (without .py)
        
    Returns:
        Summary dictionary from certification
    """
    try:
        # Try to import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "unified_framework", 
            f"{script_path}.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {script_path}.py")
        
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        if not hasattr(mod, "certify_zero_range"):
            raise AttributeError(
                f"Module {script_path} has no function 'certify_zero_range'"
            )

        print(f"[Batch driver] Certifying zeros {start_n} to {end_n}...")
        print(f"[Batch driver] Using {script_path}.certify_zero_range")
        
        result = mod.certify_zero_range(start_n, end_n)
        
        print("[Batch driver] Certification complete!")
        return result
        
    except Exception as e:
        print(f"[Batch driver] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}


# -------------------------------------------------------------------
# 5. Main analysis routine
# -------------------------------------------------------------------

def analyze_certificates(cert_files: List[str], 
                        plot_spacing: bool = True,
                        save_plots: bool = False) -> Dict[str, Any]:
    """
    Complete analysis pipeline for certificate files.
    
    Args:
        cert_files: List of JSON certificate file paths
        plot_spacing: Whether to generate spacing plots
        save_plots: Whether to save plots to disk
        
    Returns:
        Dictionary with all analysis results
    """
    print("\n" + "=" * 70)
    print("ZERO CERTIFICATE ANALYSIS")
    print("=" * 70 + "\n")
    
    # Load certificates
    print("Loading certificates...")
    zeros_all = load_multiple_cert_files(cert_files)
    
    if not zeros_all:
        print("No certificates loaded!")
        return {}
    
    print(f"\nTotal zeros loaded: {len(zeros_all)}")
    print(f"Index range: {zeros_all[0].index} - {zeros_all[-1].index}")
    
    # Compute and print summary statistics
    print("\n" + "=" * 70)
    print("PERFORMANCE STATISTICS")
    print("=" * 70 + "\n")
    stats = summarize_certificates(zeros_all)
    print_summary(stats)
    
    # Spacing analysis
    print("\n" + "=" * 70)
    print("SPACING ANALYSIS")
    print("=" * 70 + "\n")
    t_vals = extract_t_values(zeros_all)
    spacings = compute_normalized_spacings(t_vals)
    
    spacing_stats = compute_spacing_statistics(spacings)
    print(f"Number of spacings: {spacing_stats.get('count', 0)}")
    print(f"Mean spacing (normalized): {spacing_stats.get('mean', 0):.4f}")
    print(f"Std deviation: {spacing_stats.get('std', 0):.4f}")
    print(f"Min spacing: {spacing_stats.get('min', 0):.4f}")
    print(f"Max spacing: {spacing_stats.get('max', 0):.4f}")
    
    # Plot if requested
    if plot_spacing and HAS_MATPLOTLIB:
        print("\nGenerating spacing histogram...")
        save_path = "spacing_histogram.png" if save_plots else None
        plot_spacing_histogram(spacings, bins=50, x_max=4.0, 
                             show=True, save_path=save_path)
    
    return {
        "summary": stats,
        "spacing_stats": spacing_stats,
        "zeros": zeros_all,
        "t_values": t_vals,
        "spacings": spacings,
    }


# -------------------------------------------------------------------
# 6. Command-line interface
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze zero certificates and drive batch certification"
    )
    parser.add_argument(
        "--analyze",
        nargs="+",
        help="Certificate JSON files to analyze"
    )
    parser.add_argument(
        "--certify",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Certify zeros START to END"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to disk"
    )
    
    args = parser.parse_args()
    
    # Default: analyze example files if they exist
    if not args.analyze and not args.certify:
        # Look for common certificate files
        example_files = [
            "zero_certificates_100.json",
            "zeros_101_to_200.json",
            "zeros_201_to_300.json",
        ]
        existing_files = [f for f in example_files if os.path.exists(f)]
        
        if existing_files:
            print("Found certificate files:")
            for f in existing_files:
                print(f"  - {f}")
            print("\nAnalyzing...")
            analyze_certificates(
                existing_files,
                plot_spacing=not args.no_plot,
                save_plots=args.save_plots
            )
        else:
            print("No certificate files found.")
            print("Usage:")
            print("  python zero_analysis_and_scaling.py --analyze file1.json file2.json")
            print("  python zero_analysis_and_scaling.py --certify 201 300")
    
    # Analyze specified files
    if args.analyze:
        analyze_certificates(
            args.analyze,
            plot_spacing=not args.no_plot,
            save_plots=args.save_plots
        )
    
    # Certify new range
    if args.certify:
        start_n, end_n = args.certify
        batch_certify_with_unified(start_n, end_n)
