import joblib
import numpy as np
import torch

from ffd_clustering import FFDClustering, Settings
from ffd_clustering.grouping import fuzzy_variable_groups
from ffd_clustering.networks import DiscriminativeMappingNetwork, FusionNetwork


def test_group_memberships():
    samples = np.random.default_rng(8).normal(size=(64, 8))
    grouping = fuzzy_variable_groups(samples, 3, seed=8)
    assert grouping.memberships.shape == (8, 3)
    assert np.allclose(grouping.memberships.sum(1), 1)
    assert all(len(group) for group in grouping.groups)


def test_network_shapes():
    decoder_output, middle, latent, key_middle, key_latent = FusionNetwork(4)(torch.randn(6, 4))
    assert decoder_output.shape == (6, 4)
    assert middle.shape == key_middle.shape == (6, 16)
    assert latent.shape == key_latent.shape == (6, 1)
    mapped, reconstructed = DiscriminativeMappingNetwork(3)(torch.randn(6, 3))
    assert mapped.shape == reconstructed.shape == (6, 3)


def test_centroid_distance_matches_kmeans(tmp_path):
    rng = np.random.default_rng(23)
    train = rng.normal(size=(96, 9)).astype('float32')
    test = rng.normal(size=(24, 9)).astype('float32')
    model = FFDClustering(Settings(groups=3, clusters=3, fen_epochs=2,
                                    dmn_epochs=2, batch_size=32, seed=23,
                                    contamination=0.1)).fit(train)
    mapped = model.transform(test)
    expected = model.detector_.transform(mapped).min(axis=1)
    assert np.allclose(model.decision_function(test), expected)
    assert np.isclose(model.threshold_, np.quantile(model.train_scores_, 0.9))
    assert np.array_equal(model.predict(test), (expected >= model.threshold_).astype('int64'))
    assert len(model.locate_groups(test, detected_only=False)) == 3
    target = tmp_path / 'model.joblib'
    joblib.dump(model, target)
    assert np.allclose(joblib.load(target).decision_function(test), expected)


def test_input_validation():
    train = np.random.default_rng(4).normal(size=(64, 5)).astype('float32')
    model = FFDClustering(Settings(groups=3, clusters=3, fen_epochs=1, dmn_epochs=1)).fit(train)
    try:
        model.predict(np.ones((2, 4), dtype='float32'))
    except ValueError:
        pass
    else:
        raise AssertionError('Feature count mismatch must raise ValueError')
