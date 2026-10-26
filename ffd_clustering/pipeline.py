"""FFD-Clustering: fuzzy grouping, FEN, DMN, and final K-means."""
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans

from .grouping import fuzzy_variable_groups, FuzzyGroups
from .networks import FusionNetwork, DiscriminativeMappingNetwork


@dataclass
class Settings:
    groups: int = 4
    clusters: int = 4
    fen_epochs: int = 50
    dmn_epochs: int = 100
    learning_rate: float = 0.1
    contrast_weight: float = 0.8
    center_weight: float = 0.0005
    key_momentum: float = 0.9
    batch_size: int = 128
    contamination: float = 0.08
    seed: int = 17


class FFDClustering:
    """Fuzzy grouping, feature fusion, discriminative mapping and K-means."""
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.fitted = False

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32)

    @staticmethod
    def adaptive_momentum(previous_momentum: float, old_grad, new_grad, n_epochs: int):
        """Gradient-direction-aware key update from paper Eq. (14).

        A numerical clamp is applied when gradient ratios make the raw update
        invalid as a convex-combination parameter.
        """
        if old_grad is None or new_grad is None:
            return previous_momentum
        old_norm = torch.linalg.vector_norm(old_grad).item()
        new_norm = torch.linalg.vector_norm(new_grad).item()
        if old_norm < 1e-12 or new_norm < 1e-12:
            return previous_momentum
        cosine = (torch.dot(old_grad, new_grad) / (old_norm * new_norm)).item()
        ratio = new_norm / old_norm
        raw = previous_momentum * ((1.0 - 1.0 / n_epochs) - ratio * cosine / n_epochs)
        return float(np.clip(raw, 0.0, 0.999))

    def _iter_batches(self, tensor):
        indices = torch.randperm(len(tensor))
        for subset in indices.split(self.settings.batch_size):
            yield tensor[subset]

    def fit(self, data: np.ndarray):
        x = np.asarray(data, dtype=np.float32)
        if x.ndim != 2 or len(x) < 20 or not np.isfinite(x).all():
            raise ValueError('fit requires finite [samples, variables] data')
        if not (0 < self.settings.contamination < 1):
            raise ValueError('contamination must lie in (0, 1)')
        if self.settings.groups > x.shape[1] or self.settings.groups < 2:
            raise ValueError('groups must be between 2 and the number of variables')
        if not (2 <= self.settings.clusters <= len(x)):
            raise ValueError('invalid number of clusters')
        if min(self.settings.fen_epochs, self.settings.dmn_epochs, self.settings.batch_size) < 1:
            raise ValueError('epochs and batch size must be positive')
        torch.manual_seed(self.settings.seed)
        np.random.seed(self.settings.seed)
        # Only training observations determine preprocessing statistics.
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0).clip(min=1e-6)
        normalized = (x - self.mean_) / self.std_
        self.grouping_: FuzzyGroups = fuzzy_variable_groups(
            normalized, self.settings.groups, seed=self.settings.seed)
        self.fens_ = []
        self.fen_history_ = []
        for indices in self.grouping_.groups:
            fen, history = self._train_fen(self._tensor(normalized[:, indices]), len(indices))
            self.fens_.append(fen)
            self.fen_history_.append(history)
        features = self._features_from_normalized(normalized)
        self.dmn_, self.dmn_history_ = self._train_dmn(self._tensor(features))
        mapped = self._map(features)
        # Paper Section 2.3: K-means in the unified feature space, K=4 by default.
        self.detector_ = KMeans(n_clusters=self.settings.clusters,
                                n_init=10, random_state=self.settings.seed).fit(mapped)
        cluster_counts = np.bincount(self.detector_.labels_, minlength=self.settings.clusters)
        self.cluster_counts_ = cluster_counts
        self.train_scores_ = self.detector_.transform(mapped).min(axis=1)
        self.threshold_ = float(np.quantile(self.train_scores_, 1.0 - self.settings.contamination))
        self.fitted = True
        return self

    def _train_fen(self, data, n_variables):
        s = self.settings
        fen = FusionNetwork(n_variables)
        optimizer = torch.optim.SGD(list(fen.query.parameters()) + list(fen.decoder.parameters()),
                                    lr=s.learning_rate)
        momentum, previous_grad = s.key_momentum, None
        # Eq. (14) uses the total number of parameter-update iterations.
        n_updates = s.fen_epochs * int(np.ceil(len(data) / s.batch_size))
        history = []
        for epoch in range(s.fen_epochs):
            total = 0.0
            for xb in self._iter_batches(data):
                reconstructed, q_mid, q_z, k_mid, k_z = fen(xb)
                # Paper Eq. (9), (11), (15), using the mean over group variables.
                reconstruct = (reconstructed - xb).square().mean()
                contrast = ((q_mid - k_mid).square().sum(dim=1).mean()
                            + (q_z - k_z).square().sum(dim=1).mean())
                loss = reconstruct + s.contrast_weight * contrast
                optimizer.zero_grad()
                loss.backward()
                grad = torch.cat([p.grad.detach().flatten() for p in fen.query.parameters()])
                optimizer.step()
                momentum = self.adaptive_momentum(momentum, previous_grad, grad, n_updates)
                fen.update_key(momentum)
                previous_grad = grad.clone()
                total += loss.item() * len(xb)
            history.append(total / len(data))
        fen.eval()
        return fen, history

    def _features_from_normalized(self, x):
        features = []
        with torch.no_grad():
            for fen, indices in zip(self.fens_, self.grouping_.groups):
                _, latent = fen.query(self._tensor(x[:, indices]))
                features.append(latent.numpy().reshape(-1))
        return np.column_stack(features).astype('float32')

    def _train_dmn(self, data):
        s = self.settings
        dmn = DiscriminativeMappingNetwork(s.groups)
        optimizer = torch.optim.SGD(dmn.parameters(), lr=s.learning_rate)
        history = []
        k = min(s.clusters, len(data) // 4)
        centers = None
        for epoch in range(s.dmn_epochs):
            # Step 1/2: map, reconstruct and use each sample's reconstruction
            # MSE as the pseudo-label (paper Eqs. 17--20).
            with torch.no_grad():
                mapped, restored = dmn(data)
                errors = (restored - data).square().mean(dim=1).cpu().numpy()
                z_np = mapped.cpu().numpy()
            pseudo = KMeans(n_clusters=k, n_init=3,
                            random_state=s.seed + epoch).fit_predict(errors[:, None])
            if centers is None:
                centers = torch.tensor(z_np.mean(axis=0), dtype=torch.float32).repeat(k, 1)
            # Step 3: dynamically update clustering centers, Eqs. (21)--(22).
            beta = 0.9 * float(epoch + 1) / s.dmn_epochs
            for j in range(k):
                if np.any(pseudo == j):
                    new = torch.tensor(z_np[pseudo == j].mean(axis=0), dtype=torch.float32)
                    centers[j] = beta * new + (1.0 - beta) * centers[j]
            # Step 4: reconstruction plus nearest-center distance, Eq. (23)--(24).
            epoch_loss = 0.0
            for xb in self._iter_batches(data):
                z, restored = dmn(xb)
                reconstruction = (restored - xb).square().mean()
                center_distance = torch.cdist(z, centers.detach()).square().min(dim=1).values.mean()
                loss = reconstruction + s.center_weight * center_distance
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            history.append(epoch_loss / len(data))
        dmn.eval()
        return dmn, history

    def _map(self, features):
        with torch.no_grad():
            z, _ = self.dmn_(self._tensor(features))
            return z.numpy()

    def transform(self, data):
        if not self.fitted:
            raise RuntimeError('fit the model first')
        x = np.asarray(data, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.mean_) or not np.isfinite(x).all():
            raise ValueError('data must be finite with the training feature count')
        return self._map(self._features_from_normalized((x - self.mean_) / self.std_))

    def decision_function(self, data):
        mapped = self.transform(data)
        return self.detector_.transform(mapped).min(axis=1)

    def predict(self, data):
        return (self.decision_function(data) >= self.threshold_).astype('int64')

    def locate_groups(self, data, *, detected_only: bool = True):
        """Rank associated-variable groups by reconstruction error."""
        x = np.asarray(data, dtype=np.float32)
        mask = self.predict(x).astype(bool) if detected_only else np.ones(len(x), dtype=bool)
        if not mask.any():
            return []
        norm = (x[mask] - self.mean_) / self.std_
        reports = []
        with torch.no_grad():
            for j, (fen, indices) in enumerate(zip(self.fens_, self.grouping_.groups)):
                tensor = self._tensor(norm[:, indices])
                reconstructed, *_ = fen(tensor)
                error = float((reconstructed - tensor).square().mean().item())
                reports.append({'group': j + 1, 'variables': indices.tolist(),
                                'reconstruction_error': error})
        return sorted(reports, key=lambda r: r['reconstruction_error'])
