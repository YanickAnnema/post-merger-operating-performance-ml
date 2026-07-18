# Extension and robustness studies

The thesis also included a bounded extension layer beyond the baseline LSEG/Worldscope pipeline. Those artifacts are intentionally not copied into this public repository as a folder, because the original `outputs/extension-suite`, `outputs/orbis-feasibility-audit`, and `outputs/culture-mapping` directories contain a mix of safe aggregate notes and non-public derived data.

## What was tested

- Alternative operating-performance targets, including unwinsorised CFROA, raw CFROA change, EBIT/assets, asset turnover, operating margin, and capex-intensity reduction.
- Pompe-Bilderbeek-inspired distress ratios as an additional feature family.
- Country-level culture-distance features as an exploratory robustness input.
- ORBIS SME field coverage as a feasibility audit for transferring the public-firm model structure to private SME settings.
- Cross-model SHAP summaries for the extension models.

## Main findings

The extension suite supported the thesis interpretation but did not replace the baseline model.

| Extension | Result | Public interpretation |
|---|---:|---|
| Baseline Healy CFROA model | R2 0.0221, Spearman 0.1609 | Weak but positive rank-order signal |
| Pompe-Bilderbeek add-on | R2 0.0296, Spearman 0.1975 | Small positive robustness result; confidence intervals overlap the baseline |
| Culture-distance add-on | R2 0.0120, Spearman 0.1318 | Weak or null addition for the Healy target |
| Pompe + culture combination | R2 0.0304, Spearman 0.2035 | Best Healy variant, but only marginally above Pompe alone |
| Asset-turnover target | R2 0.1167, Spearman 0.2761 | Stronger target-family signal, bounded by mean-reversion concerns |
| Asset-turnover without level features | R2 0.0254, Spearman 0.2327 | Diagnostic evidence that much of the raw R2 comes from pre-deal level features |

The practical reading is conservative: Pompe-style distress ratios add the most consistent incremental information, culture proxies are weak or mixed, and asset turnover is more predictable but should not be promoted as a clean replacement for the Healy CFROA target.

## Why the extension folders are not included

The local extension folders include files that should stay private:

- Row-level LSEG/Worldscope-derived datasets.
- SHAP value matrices at test-observation level.
- ORBIS workbook exports and private-company coverage material.
- Cached model outputs, experiment folders, and local machine paths.
- Culture score CSV/JSON files whose source recorded non-commercial or non-profit usage restrictions.

For that reason, this repository documents the extension logic and headline aggregate outcomes, while excluding the underlying data and generated matrices. This keeps the portfolio version useful without redistributing licensed or usage-restricted material.

## Public-safe next step

A later public update could add a sanitized extension runner that works only on synthetic data and reproduces the same experiment structure. That should be a clean implementation, not a direct dump of the thesis `outputs/` folders.
