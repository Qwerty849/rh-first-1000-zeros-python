# Certified First 1,000 Zeros of the Riemann Zeta Function

This repository contains the final certified dataset of the first **1,000 nontrivial zeros**
of the Riemann zeta function  
\[
\zeta\!\left(\tfrac12 + it\right),
\]
produced using a **dual-evaluator method**, a **hexagonal argument-principle contour**, strict
**Krawczyk uniqueness isolation**, and an **automatic refinement pipeline** that corrects any
multi-zero contours or weak contraction regions.

The goal is to provide a clean, reproducible, and verifiable reference dataset for research,
analysis, numerical experiments, or independent verification.

---

## Contents
data/
zeros_1_to_1000_final.json # Final certified dataset

scripts/
unified_zeta_framework_v2.5.py # Full certification engine
zero_analysis_and_scaling.py # Spacing analysis + stability metrics
merge_zero_certs.py # Utility to merge per-range JSONs
These are the only files needed to reproduce the dataset from scratch.

---

## Method Summary

### **1. Dual ζ Evaluators (Consistency Check)**
Each contour evaluation uses two independent ζ functions:

- `mpmath.zeta(s)`
- Dirichlet η-series partial summation

The maximum disagreement (`max_evaluator_diff_on_contour`) confirms numerical stability.

---

### **2. Hexagonal Contour + Argument Principle**
Each zero is enclosed inside a hexagonal contour.  
Winding numbers are computed for both evaluators:

- `wA_int = 1`
- `wB_int = 1`

Any contour that encloses more than one zero (`w = 2`) automatically triggers refinement.

---

### **3. Wavelength-Limited Sampling**
Contour sampling is governed by a Nyquist-style bound using the local phase speed of  
\[
\frac{\zeta'}{\zeta},
\]
ensuring correct resolution of phase jumps and preventing aliasing.

---

### **4. Krawczyk Uniqueness Test**
Each zero is validated with a 2D Krawczyk operator, verifying:

- `β < 1` (contraction)  
- `ρ ≤ r_box` (isolation)  
- exactly **one** zero exists in the box  

If any condition fails, refinement is automatically applied.

---

### **5. Automatic Refinement**
For any zero where the contour or Krawczyk test fails:

- contour radius is reduced  
- Krawczyk box is reduced  
- evaluator agreement rechecked  
- the full certification cycle repeats  

This continues until the zero satisfies:

- `wA_int = wB_int = 1`
- `β` safely below 1
- `ρ ≤ r_box`
- evaluator agreement is stable

---

## Dataset

The file:
data/zeros_1_to_1000_final.json

contains, for each zero:

- `zero_index`
- `approx_zero.t` (the height)
- modulus bounds (`min_abs_zeta_on_contour`)
- evaluator agreement (`max_evaluator_diff_on_contour`)
- winding numbers (`wA`, `wB`, `wA_int`, `wB_int`)
- full Krawczyk isolation fields (`beta`, `rho`, `r_box`, `success`)
- contour geometry parameters

This dataset is ready for:

- visualization  
- GUE spacing experiments  
- analytic number theory research  
- replication or extension  

---

## Usage

### Certify a new range:

```bash
python scripts/unified_zeta_framework_v2.5.py --range 101 150

Analyze spacing statistics:
python scripts/zero_analysis_and_scaling.py --analyze data/zeros_1_to_1000_final.json
Merge multiple certificate files:
python scripts/merge_zero_certs.py --output merged.json zeros_*.json


License

MIT License — free for academic, commercial, and independent research use.
