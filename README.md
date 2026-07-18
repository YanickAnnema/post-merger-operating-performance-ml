# Predicting post-merger operating performance with interpretable ML

[![Thesis](https://img.shields.io/badge/MSc%20thesis-University%20of%20Twente%20(2026)-1f6feb)](https://purl.utwente.nl/essays/110659)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Model](https://img.shields.io/badge/model-XGBoost%20%2B%20SHAP-orange)](src/ml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Code companion for my MSc thesis, *Predicting Post-Merger Operating Performance Across Alternative M&A Outcome Measures*, University of Twente, 2026.

The thesis studies whether pre-deal firm and transaction information can predict post-merger operating performance. The empirical pipeline builds a deal-level dataset from LSEG/Worldscope, constructs pre-deal synergy proxies, trains XGBoost models under chronological validation, and explains feature and channel contributions with SHAP.

**Full thesis:** https://purl.utwente.nl/essays/110659

## Headline result

Predicting post-merger operating performance from pre-deal information is hard — and the thesis quantifies exactly how hard, instead of overclaiming.

| Model (test set 2019–2022) | R² | Spearman ρ |
|---|---:|---:|
| Baseline Healy CFROA model | 0.0221 | 0.1609 |
| + Pompe-Bilderbeek distress ratios | 0.0296 | 0.1975 |
| Asset-turnover target (alternative outcome) | 0.1167 | 0.2761 |

The honest takeaway: pre-deal data carries a weak but real rank-ordering signal for post-merger performance, and SHAP shows it is concentrated in operational and financial channels rather than classic revenue-synergy stories. See `docs/extensions.md` for the full robustness table.

## The thesis in 46 seconds

A short explainer video of the research question, the model, and the screening app built on top of it: [`assets/thesis-video.mp4`](assets/thesis-video.mp4)

<!-- To embed an inline video player: edit this README on github.com, drag
     assets/thesis-video.mp4 into the editor just below this comment, and GitHub
     will insert a user-attachments URL that renders as a player. -->

## Defense deck

Six key slides from the interactive defense presentation — pipeline, sample construction, chronological validation, the central result, and the SHAP story:

[![Defense deck preview](assets/readme/deck-preview.gif)](presentation/)

The full deck lives in [`presentation/`](presentation/) — open `presentation/index.html` in any browser (arrow keys to navigate, `a` for the appendix). All figures show aggregate results only; row-level data and SHAP matrices are not redistributed.

## Code map

How the pipeline above maps to the source folders in this repository:

```mermaid
flowchart LR
    subgraph DAQ["Data acquisition — src/daq/"]
        A["LSEG deal screener<br/>M&amp;A deals 1995–2022"] --> B["Worldscope financials<br/>acquiror + target, t−1 … t+3"]
        B --> C["Healy (1992) target:<br/>industry-adjusted ΔCFROA"]
        C --> D["Pre-deal synergy proxies<br/>cost / revenue / operational / financial"]
        D --> E["Macro merge<br/>rates, spreads, market conditions"]
    end
    subgraph ML["Modeling — src/ml/"]
        E --> F["Chronological splits<br/>train ≤2015 · val 2016–18 · test 2019–22"]
        F --> G["XGBoost regression<br/>tuned on inner validation"]
    end
    subgraph XAI["Explanation — src/shap/"]
        G --> H["SHAP values"]
        H --> I["Channel-level attribution<br/>operational · financial · cost · macro · revenue"]
    end
```

## What is included

- `src/daq/`: data acquisition, feature engineering, macro merge, and diagnostic scripts.
- `src/ml/`: XGBoost training, validation, evaluation, and model visualisation.
- `src/shap/`: SHAP feature and channel attribution pipeline.
- `src/synthetic/`: small synthetic-data helper for local demos without licensed data.
- `presentation/`: the interactive defense deck (single HTML file, 28 slides + Q&A appendix).
- `docs/data_policy.md`: what can and cannot be redistributed.
- `docs/extensions.md`: public-safe summary of the extension and robustness studies.
- `docs/visual_companion.md`: why selected visuals are included but full presentation/dashboard folders are not.

## What is not included

The underlying LSEG/Worldscope data, ORBIS exports, derived deal-level datasets, caches, trained model artifacts, and SHAP matrices are not redistributed. They are proprietary inputs, licensed-derived files, or usage-restricted materials. The repository is intended to share the research code, aggregate visuals, and reproducibility structure, not the vendor data.

## Repository layout

```text
post-merger-operating-performance-ml/
  assets/
    readme/
  presentation/
    figs/
  src/
    daq/
    ml/
    shap/
    synthetic/
  data/       ignored, except .gitkeep
  outputs/    ignored, except .gitkeep
  config/     ignored, except .gitkeep
  docs/
  requirements.txt
  LICENSE
```

## Reproducing with licensed data

1. Install dependencies: `pip install -r requirements.txt`.
2. Configure your own LSEG/Workspace access locally. Do not commit config files.
3. Run the DAQ scripts in `src/daq/` to create the intermediate files expected by the ML scripts.
4. Run `src/ml/model_training.py`, then `src/ml/model_evaluation.py` and `src/shap/shap_analysis.py`.

Paths in the original thesis scripts assume the local thesis project layout. Before making this repo fully reproducible, replace those paths with a small project-root config layer.

## Data note

The public repository should only contain code, metadata, documentation, and synthetic examples. LSEG/Worldscope data and any licensed-derived data remain private. See `docs/data_policy.md`.

## License

Code is released under the MIT License. Data is not included.
