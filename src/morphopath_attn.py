"""
MorphoPath attention core: multiplicative masking (saliency x morphology).

Key innovation: decouple attention into two orthogonal components:
  1. morph_attn = softmax(loc_sim / tau_loc, dim=0)  → pure cosine, interpretable
  2. saliency = sigmoid(gated_network(h))             → learned diagnostic mask [N, 1]
  3. final_attn = morph_attn * saliency → renormalized

Gated network = "is this patch diagnostic?" (like learned normal detection)
Cosine = "what concept does this patch belong to?" (pure interpretability)

No lambda_loc_bias needed. No normal concept needed in W_loc.
Normal anchor in W_score is optional (can use class-balanced centroid or normal concept).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.morphopath_backbone import MorphoPathBackbone


class MorphoPathAttn(MorphoPathBackbone):
    """Multiplicative masking + normal anchor + feature-level balance."""

    def __init__(self, tau_init: float = 5.0, n_diagnostic: int = None, use_normal_anchor: bool = True, **kwargs):
        super().__init__(**kwargs)

        # n_diagnostic: number of diagnostic concepts (excl normal in W_score)
        if n_diagnostic is None:
            n_diagnostic = self.n_concepts  # no normal concept
        self.n_diagnostic = n_diagnostic
        self.use_normal_anchor = use_normal_anchor

        # Feature-level balance (diagnostic only)
        n_astro = n_diagnostic - self.n_oligo
        scale = torch.ones(n_diagnostic)
        scale[self.n_oligo:] = self.n_oligo / n_astro
        self.register_buffer("concept_scale", scale)

        # Per-concept tau (diagnostic only)
        self.tau = nn.Parameter(torch.ones(n_diagnostic) * tau_init)

        # Saliency network: gated attention → [N, 1] diagnostic mask
        # Reuse attn_V and attn_U from parent, but new output layer
        self.saliency_W = nn.Linear(self.attn_V.out_features, 1)

        # Predictor for diagnostic concepts only
        profile_dim = 3 * n_diagnostic
        self.predictor = nn.Linear(profile_dim, 1)
        self.grade_head = nn.Linear(profile_dim, 1)
        self._init_predictor_weights()

        # Exclusion mask for diagnostic concepts
        mask = torch.zeros(n_diagnostic, n_diagnostic)
        for i in range(n_diagnostic):
            for j in range(n_diagnostic):
                if (i < self.n_oligo) != (j < self.n_oligo):
                    mask[i, j] = 1.0
        self.register_buffer("exclusion_mask", mask)

    def forward(self, features, coords=None):
        if coords is not None and coords.numel() > 0:
            coords = coords.float()
            coord_range = coords.max(dim=0).values - coords.min(dim=0).values
            coord_range = coord_range.clamp(min=1.0)
            coords = (coords - coords.min(dim=0).values) / coord_range

        h = self.projection(features)
        h_norm = F.normalize(h, dim=-1)

        # ── 1. Morphological attention: pure cosine (interpretable) ──
        w_loc_norm = F.normalize(self.W_loc[:self.n_diagnostic], dim=-1)  # [K_diag, dim]
        loc_sim = h_norm @ w_loc_norm.T                                    # [N, K_diag]
        morph_attn = F.softmax(loc_sim / self.tau_loc, dim=0)              # [N, K_diag]

        # ── 2. Saliency mask: gated network (learned diagnostic detector) ──
        a = torch.tanh(self.attn_V(h))
        b = torch.sigmoid(self.attn_U(h))
        saliency_logits = self.saliency_W(a * b)           # [N, 1]
        saliency_mask = torch.sigmoid(saliency_logits)      # [N, 1], 0=background, 1=diagnostic

        # ── 3. Multiplicative fusion + renormalize ──
        raw_attn = morph_attn * saliency_mask               # [N, K_diag]
        attn_weights = raw_attn / (raw_attn.sum(dim=0, keepdim=True) + 1e-8)  # [N, K_diag]

        # ── 4. Concept scoring with anchor ──
        w_score_norm = F.normalize(self.W_score, dim=-1)
        score_sim = h_norm @ w_score_norm.T                 # [N, K_total]

        if self.use_normal_anchor and self.n_concepts > self.n_diagnostic:
            # Normal concept as anchor
            normal_sim = score_sim[:, self.n_diagnostic:].mean(dim=1, keepdim=True)
            diagnostic_sim = score_sim[:, :self.n_diagnostic]
            calibrated_sim = diagnostic_sim - normal_sim
        else:
            # Class-balanced centroid anchor (v12 style)
            diag_score_norm = w_score_norm[:self.n_diagnostic]
            oligo_center = diag_score_norm[:self.n_oligo].mean(dim=0)
            astro_center = diag_score_norm[self.n_oligo:self.n_diagnostic].mean(dim=0)
            anchor = F.normalize(((oligo_center + astro_center) / 2).unsqueeze(0), dim=-1)
            anchor_sim = h_norm @ anchor.T
            calibrated_sim = score_sim[:, :self.n_diagnostic] - anchor_sim

        concept_scores = torch.sigmoid(calibrated_sim * self.tau)  # [N, K_diag]

        # ── 5. Aggregation + Prediction ──
        slide_profile = self._compute_profile(concept_scores, attn_weights, coords)

        if self.sign_constraint:
            logit = self._sign_constrained_predict_diag(slide_profile)
        else:
            logit = self.predictor(slide_profile).squeeze()
        logit_grade = self.grade_head(slide_profile).squeeze()

        # Store saliency for loss computation
        self._last_saliency = saliency_mask
        self._last_concept_scores = concept_scores

        return logit, logit_grade, concept_scores, attn_weights, slide_profile

    def loss_saliency(self):
        """Pseudo-label self-distillation: W_score teaches saliency where to look."""
        target = self._last_concept_scores.max(dim=-1)[0].detach()  # [N]
        pred = self._last_saliency.squeeze(-1)                       # [N]
        return F.binary_cross_entropy(pred, target)

    def _compute_profile(self, concept_scores, attn_weights, coords):
        weighted_score = (attn_weights * concept_scores).sum(dim=0) * self.concept_scale
        peak = concept_scores.max(dim=0).values * self.concept_scale
        if coords is not None:
            centroid = attn_weights.T @ coords.float()
            diffs = coords.float().unsqueeze(1) - centroid.unsqueeze(0)
            dists = diffs.norm(dim=-1)
            spread = (attn_weights * dists).sum(dim=0) * self.concept_scale
        else:
            spread = torch.zeros(self.n_diagnostic, device=attn_weights.device)
        return torch.cat([weighted_score, peak, spread])

    def _sign_constrained_predict_diag(self, slide_profile):
        w = self.predictor.weight
        K = self.n_diagnostic; n_o = self.n_oligo
        blocks = []
        for s in [0, K, 2 * K]:
            blocks.append(torch.cat([F.softplus(w[0, s:s+n_o]), -F.softplus(-w[0, s+n_o:s+K])]))
        w_c = torch.cat(blocks).unsqueeze(0)
        return F.linear(slide_profile, w_c, self.predictor.bias).squeeze()

    def loss_attn_div(self, attn_weights):
        C = attn_weights.T @ attn_weights
        eye = torch.eye(self.n_diagnostic, device=C.device)
        diff = (C - eye) ** 2
        return (self.exclusion_mask.to(C.device) * diff).sum()

    def get_saliency(self, features):
        """Return saliency mask for visualization."""
        with torch.no_grad():
            h = self.projection(features)
            a = torch.tanh(self.attn_V(h))
            b = torch.sigmoid(self.attn_U(h))
            saliency = torch.sigmoid(self.saliency_W(a * b)).squeeze(-1)
        return saliency
