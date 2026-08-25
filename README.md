# Dual-Listing Spread Dynamics

### Institutional Arbitrage Mechanisms and the Law of One Price in Chinese Cross-Listed Equities, 2014–2026

This project studies why the law of one price behaves differently across two forms of Chinese cross-listing:

- **A–H pairs:** Shanghai/Shenzhen and Hong Kong dual-listed shares
- **H–ADR pairs:** Hong Kong-listed shares and US-listed ADRs

The central question is whether the presence of an institutional conversion mechanism affects the persistence of cross-market price differences.

## Key Results

| | A–H | H–ADR |
|---|---:|---:|
| Verified pairs | 63 | 20 |
| Mean premium | **+77.95%** | **−0.15%** |
| ADF rejects unit root (FDR) | 0 / 63 | 20 / 20 |
| Engle–Granger cointegrated (FDR) | 0 / 63 | 20 / 20 |
| Median AR(1) coefficient | 0.9936 | 0.0972 |
| Median half-life | 107.4 days | 0.30 days |

The A–H and H–ADR samples use pair-specific histories but share a common end date of **August 20, 2026**.

The results suggest that a low-friction institutional conversion mechanism is an important determinant of price convergence. H–ADR pairs, where a depositary mechanism provides two-way conversion between the receipt and underlying Hong Kong shares, exhibit tight cointegration. A–H pairs have no equivalent conversion mechanism and exhibit persistent price differences.

## Methodology

The analysis includes:

- A–H and H–ADR premium construction
- H–ADR conversion-ratio estimation and verification
- Augmented Dickey–Fuller, Phillips–Perron and KPSS stationarity tests
- Engle–Granger cointegration testing
- Johansen cointegration diagnostics
- Benjamini–Hochberg false-discovery-rate correction
- AR(1) persistence and half-life estimation
- Zivot–Andrews structural-break analysis
- Price staleness and jump diagnostics
- Adjusted versus unadjusted price sensitivity analysis

## Data

The final sample contains:

- **63 A–H pairs**
- **20 verified H–ADR pairs**

Three attempted A–H pairs were excluded because of delisting/data availability, while H–ADR candidates were filtered using conversion-ratio and stability diagnostics.

The two groups use pair-specific sample histories and a common end date of August 20, 2026.

## Repository Structure

```text
dual-listing-spread-dynamics/
│
├── data/
│   └── processed/
│       ├── breaks.csv
│       ├── ah_quality_flags.csv
│       └── johansen_diagnostic.csv
│
├── results/
│   └── figures/
│       └── ah_vs_adr_comparison.png
│
├── src/
│   ├── build_all_pairs.py
│   ├── build_adr_pairs.py
│   ├── breaks.py
│   ├── final_comparison_chart.py
│   ├── johansen_diagnostic.py
│   └── ...
│
├── report/
│   └── Report.pdf
│
├── requirements.txt
└── README.md

```

## Reproducibility

Environment
Python 3.13.6
Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

The analysis scripts in src/ can then be run to reproduce the corresponding processed data, diagnostics, and figures.

## Report

The full research report is available in [report/Report.pdf](report/Report.pdf).
The report provides the complete methodology, statistical results, robustness checks, limitations, and discussion of the institutional mechanisms underlying the A–H and H–ADR comparison.

## Limitations

The analysis has several important limitations, including pair-specific sample histories, survivorship selection in the A–H universe, non-synchronous Hong Kong/New York closing times for H–ADR pairs, and the use of unadjusted closing prices in the final specification.
A matched post-2021 robustness analysis and a full transaction-cost-aware trading backtest are identified as natural extensions.

## Citation

If you use this project or its results, please cite the accompanying research report:

> **Dual-Listing Spread Dynamics: Institutional Arbitrage Mechanisms and the Law of One Price in Chinese Cross-Listed Equities, 2014–2026**  
> Rishabh Verma  
> *Indian Statistical Institute, Kolkata (2026)*

```bibtex
@article{verma2026duallisting,
  title={Dual-Listing Spread Dynamics: Institutional Arbitrage Mechanisms and the Law of One Price in Chinese Cross-Listed Equities, 2014--2026},
  author={Verma, Rishabh},
  institution={Indian Statistical Institute, Kolkata},
  year={2026}
}