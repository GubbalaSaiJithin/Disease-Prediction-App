# Symptom Classification Demo

An educational Streamlit application for classifying symptom sets. **It is not a diagnostic tool.** The bundled data are small, repetitive and do not establish clinical validity.

## Run locally

Tested on Windows with Python 3.10. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true
```

Open the local URL shown by Streamlit. Run training once before starting the app; it creates the model in artifacts/. Model and data paths are resolved relative to this project, so app startup does not depend on the terminal's current folder.

## September 2026 repairs

- Canonicalise each row as a set of symptoms. Selection order, case, underscores and duplicate selections no longer change the input.
- Keep absent symptoms absent. Removed mode imputation, numeric symptom IDs treated as ordered features, hard-coded ten-feature padding and silent truncation.
- Use binary symptom features fitted inside each cross-validation fold.
- Remove exact duplicate symptom sets before the train/test split. The original 4,920 rows reduce to 304 distinct sets across 41 labels; 4,616 duplicates are excluded.
- Compare logistic regression and random forest using training-only three-fold cross-validation. Evaluate the selected model on a separate stratified test split.
- Replace hard-coded accuracy/precision/recall/F1 labels with results loaded from the actual training run.
- Remove the runtime dependency on the old XGBoost pickle and pin the tested runtime dependencies.

The old Pickle files/ artifacts and final_dataset.csv remain only for historical reference. The repaired app never loads them. The notebooks now use the same corrected modules as the app.

## Evaluation

The fixed split contains 228 training and 76 test symptom sets, with zero exact-set overlap. Both candidate models achieved macro F1 1.0 in the three development folds; logistic regression was selected by the deterministic tie rule. Its test accuracy and macro F1 were also 1.0 on these 76 examples. The majority-class baseline accuracy was 0.0263.

These high values mainly demonstrate how separable this small template-like dataset is. Deduplication removes exact leakage but does not turn it into an independent patient cohort. Do not present these scores as diagnostic performance. Model scores shown in the UI are not calibrated probabilities of illness.

Full metrics, source checksum and split manifest are in artifacts/. Four regression/interaction tests passed, including reordered input, invalid input and a real Streamlit button interaction. A separate live HTTP startup check also passed.

## Files

- symptom_model.py: normalisation, binary features and inference.
- train.py: deduplication, split, model comparison, training and evaluation.
- app.py: Streamlit interface.
- test_project.py: regression and app interaction checks.

The repairs were made with AI assistance. Explain both the original design and the corrections when discussing this project. Original source data and assets were preserved from this repository; their provenance/license needs to be documented before reusing them in another public dataset release.
