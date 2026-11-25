"""Shared fragment-pair scoring and protein-pair aggregation."""
import torch
from torch import nn

class PairwiseScorerMLPSymmetry(nn.Module):
    """Symmetric pairwise MLP with log-sum-exp MIL aggregation."""

    def __init__(self, temperature: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.temperature = float(temperature)

    def project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project contextual word embeddings into the model's 256-D space."""

        return self.proj(embeddings)

    @staticmethod
    def symmetric_pair_features(
        projected_a: torch.Tensor,
        projected_b: torch.Tensor,
    ) -> torch.Tensor:
        """Build the symmetric pair feature ``[A+B, A*B, |A-B|]``."""

        if projected_a.shape != projected_b.shape:
            raise ValueError(
                "Projected endpoint tensors must have identical shapes: "
                f"{projected_a.shape} != {projected_b.shape}"
            )
        return torch.cat(
            (
                projected_a + projected_b,
                projected_a * projected_b,
                torch.abs(projected_a - projected_b),
            ),
            dim=-1,
        )

    def score_pair_features(
        self,
        pair_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pair scores and the pre-dropout 256-D hidden representation."""

        hidden = self.pair_mlp[1](self.pair_mlp[0](pair_features))
        score = self.pair_mlp[3](self.pair_mlp[2](hidden)).squeeze(-1)
        return score, hidden

    def forward(
        self,
        batch_emb_a: torch.Tensor,
        batch_emb_b: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_a, _ = batch_emb_a.shape
        _, max_b, _ = batch_emb_b.shape

        emb_a = self.project_embeddings(batch_emb_a)
        emb_b = self.project_embeddings(batch_emb_b)
        a_exp = emb_a.unsqueeze(2).expand(batch_size, max_a, max_b, -1)
        b_exp = emb_b.unsqueeze(1).expand(batch_size, max_a, max_b, -1)
        pair_feat = self.symmetric_pair_features(a_exp, b_exp)
        pair_score, _hidden = self.score_pair_features(pair_feat)

        if mask_a is not None and mask_b is not None:
            pair_mask = mask_a.unsqueeze(2) & mask_b.unsqueeze(1)
            pair_score = pair_score.masked_fill(~pair_mask, float("-inf"))

        scores = torch.logsumexp(pair_score / self.temperature, dim=(1, 2))
        return scores, pair_score
