import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ffd_clustering import FFDClustering, Settings
from ffd_clustering.grouping import group_count_diagnostics


def load_data(train_path: Path, test_path: Path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if train.empty or test.empty or list(train.columns) != list(test.columns):
        raise ValueError('Train and test CSVs require identical numeric columns and nonempty rows.')
    if train.columns.duplicated().any():
        raise ValueError('Duplicate sensor names are not supported.')
    return train.to_numpy(dtype=np.float32), test.to_numpy(dtype=np.float32), train.columns.tolist()


def main():
    parser = argparse.ArgumentParser(description='FFD-Clustering')
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/run'))
    parser.add_argument('--groups', type=int, default=6)
    parser.add_argument('--clusters', type=int, default=4)
    parser.add_argument('--contamination', type=float, required=True)
    parser.add_argument('--fen-epochs', type=int, default=50)
    parser.add_argument('--dmn-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--scan-groups', action='store_true')
    args = parser.parse_args()

    x_train, x_test, columns = load_data(args.train_csv, args.test_csv)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError('Input CSVs must contain finite numeric sensor values.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(groups=args.groups, clusters=args.clusters,
                        contamination=args.contamination,
                        fen_epochs=args.fen_epochs, dmn_epochs=args.dmn_epochs,
                        batch_size=args.batch_size, seed=args.seed)

    if args.scan_groups:
        normalized = (x_train - x_train.mean(0)) / x_train.std(0).clip(min=1e-6)
        candidates = range(2, min(10, x_train.shape[1]) + 1)
        pd.DataFrame(group_count_diagnostics(normalized, candidates, seed=args.seed)).to_csv(
            args.output_dir / 'group_diagnostics.csv', index=False)

    model = FFDClustering(settings).fit(x_train)
    mapped = model.transform(x_test)
    predictions = pd.DataFrame({
        'anomaly_score': model.decision_function(x_test),
        'prediction': model.predict(x_test),
    })
    for group in range(mapped.shape[1]):
        predictions[f'feature_{group + 1}'] = mapped[:, group]
    predictions.to_csv(args.output_dir / 'predictions.csv', index=False)

    pd.DataFrame(model.grouping_.memberships, index=columns,
                 columns=[f'Group_{i + 1}' for i in range(settings.groups)]).to_csv(
        args.output_dir / 'fuzzy_membership.csv', index_label='variable')

    report = {
        'partition_coefficient': model.grouping_.partition_coefficient,
        'classification_entropy': model.grouping_.classification_entropy,
        'threshold': model.threshold_,
        'detected_anomalies': int(predictions['prediction'].sum()),
        'groups': [
            {'group': i + 1, 'variables': [columns[j] for j in indices]}
            for i, indices in enumerate(model.grouping_.groups)
        ],
        'anomaly_group_ranking': [
            {'group': g['group'], 'variables': [columns[j] for j in g['variables']],
             'reconstruction_error': g['reconstruction_error']}
            for g in model.locate_groups(x_test)
        ],
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    joblib.dump(model, args.output_dir / 'model.joblib')
    print(f'Results: {args.output_dir.resolve()}')
    print(f'Detected anomalies: {report["detected_anomalies"]}/{len(x_test)}')


if __name__ == '__main__':
    main()
