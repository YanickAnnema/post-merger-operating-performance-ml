# Data policy

This repository is code-only.

## Not redistributed

The following must not be committed or published:

- Raw LSEG/Worldscope exports.
- LSEG/Refinitiv/Workspace configuration files.
- Deal-level datasets such as `full_deal_level.csv`, `full_deal_level_features.csv`, `ml_ready.csv`, or `ml_ready_nowinsor.csv`.
- Cache files such as `deals_*.pkl` and `financials_*.pkl`.
- Trained models and split artifacts such as `xgboost_model_final.pkl` and `training_splits.pkl`.
- Raw SHAP matrices or row-level output files derived from licensed data.
- Company branding or private employer materials.

## Allowed

The repository may include:

- Source code.
- Documentation.
- Field lists and schemas where they do not reveal licensed row-level data.
- Aggregate thesis results already disclosed in the public thesis.
- Synthetic demo data generated from scratch.

## Rationale

The thesis used LSEG/Worldscope data under institutional access. The data and row-level derived artifacts are not open data. The public repo therefore follows the rule: share the method, not the proprietary dataset.

