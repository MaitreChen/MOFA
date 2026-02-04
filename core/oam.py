import torch
import torch.nn as nn
import torch.nn.functional as F


class OAM(nn.Module):
    """
    Orthogonal Alignment Module (OAM)

    This module enforces:
    (1) Intra-modality decorrelation: reducing redundancy within each modality.
    (2) Inter-modality alignment control: preventing excessive similarity across modalities.

    The design balances statistical rigor (clear regularization objectives)
    and engineering practicality (stable normalization and scalable implementation).
    """

    def __init__(
            self,
            feature_dim: int = 256,
            num_modalities: int = 3,
            target_similarity: float = 0.5,
            noise_std: float = 0.1,
    ):
        super().__init__()

        # Dimensionality of each modality feature vector
        self.feature_dim = feature_dim

        # Number of modalities involved in alignment
        self.num_modalities = num_modalities

        # Target inter-modality similarity threshold (tau)
        self.target_similarity = target_similarity

        # Weights for intra- and inter-modality regularization terms
        self.intra_loss_weight = 1.0
        self.inter_loss_weight = 1.0

        # Standard deviation of Gaussian noise for robustness regularization
        self.noise_std = noise_std

        # Inter-modality interaction mode: {'symmetric', 'asymmetric'}
        self.interaction_mode = 'symmetric'

        # Intra-modality constraint mode: {'strict', 'relaxed'}
        self.intra_constraint_mode = 'relaxed'

    def _add_gaussian_noise(self, features: torch.Tensor, training: bool) -> torch.Tensor:
        """
        Inject Gaussian noise during training to improve robustness and avoid degenerate solutions.
        """
        if training and self.noise_std > 0:
            return features + torch.randn_like(features) * self.noise_std
        return features

    def _intra_modality_decorrelation(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Intra-modality decorrelation loss.

        Each modality is encouraged to have a near-orthogonal feature covariance structure.
        Two modes are supported:
        - strict  : target is an identity matrix (full decorrelation).
        - relaxed : diagonal dominance with mild off-diagonal tolerance.
        """
        total_loss = 0.0

        for features in modality_features:
            # L2 normalization to stabilize correlation estimation
            features = F.normalize(features, p=2, dim=1)

            # Empirical correlation matrix: (D x D)
            correlation_matrix = torch.mm(features.T, features) / features.shape[0]

            # Construct target correlation matrix
            if self.intra_constraint_mode == 'strict':
                target_matrix = torch.eye(self.feature_dim, device=features.device)
            else:
                # Relaxed target: strong diagonal, weak global correlation
                target_matrix = (
                        0.9 * torch.eye(self.feature_dim, device=features.device)
                        + 0.1 * torch.ones_like(correlation_matrix)
                )

            # Mean squared error between empirical and target correlation
            total_loss += F.mse_loss(correlation_matrix, target_matrix)

        return self.intra_loss_weight * total_loss / self.num_modalities

    def _inter_modality_alignment(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Inter-modality alignment loss.

        This term controls similarity across modalities to avoid information collapse.
        By default, symmetric cosine similarity is used.
        """
        total_loss = 0.0
        pair_counter = 0

        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                if self.interaction_mode == 'symmetric':
                    # Symmetric cosine similarity between modality i and j
                    similarity = F.cosine_similarity(
                        modality_features[i], modality_features[j], dim=1
                    ).mean()
                    loss = (similarity - self.target_similarity) ** 2
                else:
                    # Placeholder for asymmetric interaction (e.g., cross-attention)
                    raise NotImplementedError("Asymmetric interaction is not implemented.")

                total_loss += loss
                pair_counter += 1

        return self.inter_loss_weight * total_loss / pair_counter

    def forward(
            self,
            modality_features: list[torch.Tensor],
            training: bool = True,
    ):
        """
        Forward pass.

        Args:
            modality_features: list of tensors, each with shape (B, D)
            training: whether the module is in training mode

        Returns:
            normalized_features: L2-normalized modality features
            regularization_loss: combined intra- and inter-modality loss
        """
        # Noise injection and normalization
        noisy_features = [
            self._add_gaussian_noise(f, training) for f in modality_features
        ]
        normalized_features = [
            F.normalize(f, p=2, dim=1) for f in noisy_features
        ]

        # Regularization objectives
        intra_loss = self._intra_modality_decorrelation(normalized_features)
        inter_loss = self._inter_modality_alignment(normalized_features)

        # Balanced scaling (empirically stabilizes optimization)
        regularization_loss = 0.5 * intra_loss + 0.5 * inter_loss

        return normalized_features, regularization_loss


if __name__ == '__main__':
    batch_size = 32
    feature_dim = 256
    num_modalities = 3

    oam = OAM(
        feature_dim=feature_dim,
        num_modalities=num_modalities,
        target_similarity=0.5,
        noise_std=0.1,
    )

    # Fake modality features
    modality_features = [
        torch.randn(batch_size, feature_dim)
        for _ in range(num_modalities)
    ]

    normalized_features, reg_loss = oam(
        modality_features,
        training=True,
    )

    print("OAM forward test passed.")
    print(f"Reg loss: {reg_loss.item():.5f}")
    print(f"Input feature[0] shape: {modality_features[0].shape}")
    print(f"Output feature[0] shape: {normalized_features[0].shape}")
