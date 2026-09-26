"""Order-independent symptom-set classification for an educational demo."""
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parent

def normalize_symptom(value):
    if not isinstance(value, str):
        raise ValueError('Each symptom must be a string.')
    return re.sub(r'\s+', ' ', value.replace('_', ' ').strip().lower())

def canonical_symptoms(values):
    return tuple(sorted({normalize_symptom(v) for v in values if isinstance(v, str) and v.strip()}))

def prepare_dataset(path=ROOT / 'Datasets/dataset.csv'):
    raw = pd.read_csv(path)
    columns = [name for name in raw.columns if name.startswith('Symptom_')]
    if 'Disease' not in raw or not columns:
        raise ValueError('Dataset needs Disease and Symptom_* columns.')
    if raw.Disease.isna().any():
        raise ValueError('Missing class labels found.')
    frame = pd.DataFrame({'symptoms': raw[columns].apply(lambda row: canonical_symptoms(row.values), axis=1),
                          'label': raw.Disease.astype(str).str.strip()})
    if frame.symptoms.map(len).eq(0).any():
        raise ValueError('Rows without symptoms cannot be trained.')
    if (frame.groupby('symptoms').label.nunique() > 1).any():
        raise ValueError('Identical symptom sets have conflicting labels; inspect the data.')
    unique = frame.drop_duplicates(subset=['symptoms']).reset_index(drop=True)
    return unique, {'original_rows': len(raw), 'unique_symptom_sets': len(unique),
                    'duplicate_rows_removed': len(raw) - len(unique), 'classes': int(unique.label.nunique())}

class SymptomEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.binarizer_ = MultiLabelBinarizer().fit(X)
        self.known_ = set(self.binarizer_.classes_)
        return self

    def transform(self, X):
        # Evaluation records unknown features; the app rejects unknown user input.
        return self.binarizer_.transform([tuple(v for v in values if v in self.known_) for values in X])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.binarizer_.classes_, dtype=object)

def load_artifact(path=ROOT / 'artifacts/symptom_model.joblib'):
    if not Path(path).is_file():
        raise FileNotFoundError('Run python train.py once to create artifacts/symptom_model.joblib.')
    return joblib.load(path)

def predict_symptoms(values, artifact):
    if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
        raise ValueError('Provide a list of symptom names.')
    symptoms = canonical_symptoms(values)
    if not symptoms:
        raise ValueError('Select at least one symptom.')
    model = artifact['model']
    unknown = sorted(set(symptoms) - model.named_steps['features'].known_)
    if unknown:
        raise ValueError('Unknown symptoms: ' + ', '.join(unknown))
    probabilities = model.predict_proba([symptoms])[0]
    ranking = np.argsort(-probabilities)[:3]
    return [{'label': str(model.classes_[i]), 'model_score': float(probabilities[i])} for i in ranking]
