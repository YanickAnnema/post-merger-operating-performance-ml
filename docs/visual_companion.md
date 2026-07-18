# Visual companion

The thesis defense included a selective web presentation and an interactive code-workflow dashboard. The presentation is included in this repository in a cleaned public form (`presentation/`): the deck itself with its illustrative figures, but without the speaker script, presenter-view build, or rehearsal material. The dashboard is not included, because its content layer embeds internal process notes and employer styling.

The README's visual layer consists of:

- `assets/readme/deck-preview.gif` — an animated preview cycling six key deck slides (pipeline concept, sample funnel, chronological validation, target-family results, SHAP intuition, channel attribution), all aggregate thesis-public content.
- `assets/thesis-video.mp4` — the 46-second explainer video also used for the LinkedIn announcement of the thesis.
- `presentation/` — the full public deck itself.

## What stays out

- Presenter-view builds and speaker scripts.
- Dashboard builds, `node_modules`, and presentation-specific React app files.
- Unused or personal figure files from the original deck folder.
- Any chart or app state that exposes proprietary row-level data, local paths, or private notes.

## Future public demo option

A sanitized version of the code-workflow dashboard (pipeline stages, model evaluation flow, SHAP explanation concept, synthetic data only) would make a good GitHub Pages microsite. The `presentation/` deck already covers the narrative side publicly.
