"""Deduplicate symptom sets before splitting; fit preprocessing inside CV."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from symptom_model import ROOT, SymptomEncoder, prepare_dataset

def train(data=ROOT / 'Datasets/dataset.csv', output=ROOT / 'artifacts'):
    data, output = Path(data), Path(output)
    frame, statistics = prepare_dataset(data)
    train_indices, test_indices = train_test_split(np.arange(len(frame)), test_size=.25,
                          random_state=42, stratify=frame.label)
    development, test = frame.iloc[train_indices], frame.iloc[test_indices]
    assert set(development.symptoms).isdisjoint(test.symptoms)
    X, y = development.symptoms.tolist(), development.label.to_numpy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    candidates = {
        'logistic_regression': LogisticRegression(max_iter=2000, C=3., random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    }
    summaries, pipelines = {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([('features', SymptomEncoder()), ('classifier', estimator)])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
        summaries[name] = {'cv_macro_f1_mean': float(scores.mean()), 'cv_macro_f1_std': float(scores.std())}
        pipelines[name] = pipeline
    selected = max(summaries, key=lambda name: summaries[name]['cv_macro_f1_mean'])
    model = pipelines[selected].fit(X, y)
    actual = test.label.to_numpy()
    predicted = model.predict(test.symptoms.tolist())
    baseline = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(y), 1)), y)
    baseline_predicted = baseline.predict(np.zeros((len(actual), 1)))
    unknown_test = sorted(set(s for row in test.symptoms for s in row) - model.named_steps['features'].known_)
    metrics = {**statistics, 'train_rows': len(development), 'test_rows': len(test),
        'split': 'Deduplicated symptom sets, stratified 75/25 split, random_state=42; 3-fold CV on training only.',
        'overlapping_symptom_sets': 0, 'selected_model': selected, 'candidates': summaries,
        'test_accuracy': float(accuracy_score(actual, predicted)), 'test_macro_f1': float(f1_score(actual, predicted, average='macro', zero_division=0)),
        'baseline_accuracy': float(accuracy_score(actual, baseline_predicted)),
        'unknown_test_symptoms': unknown_test,
        'classification_report': classification_report(actual, predicted, zero_division=0, output_dict=True),
        'dataset_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
        'limitation': 'Small repetitive educational dataset; performance is NOT clinical validation. Model scores are not disease probabilities.'}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'metrics': metrics}, output / 'symptom_model.joblib')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    split = {'train': [list(row) for row in development.symptoms], 'test': [list(row) for row in test.symptoms]}
    (output / 'split_manifest.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    print(json.dumps({key:value for key,value in metrics.items() if key != 'classification_report'}, indent=2))
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'Datasets/dataset.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    train(args.data, args.output)
