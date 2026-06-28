"""
MorphoPath: concept-anchored MIL with a residual score adapter.

Adds a lightweight residual adapter before concept scoring to transform
features into a space better suited for concept discrimination.
Attention (morph_attn) still uses raw cosine for interpretability.

For CONCH (same space): adapter learns to break VL alignment, restoring concept discrimination.
For UNI (cross space): adapter learns ~0 via residual, no harm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.morphopath_attn import MorphoPathAttn


class MorphoPath(MorphoPathAttn):
    """MorphoPath: residual score adapter over the attention core for cross-encoder transfer."""

    def __init__(self, **kwargs):
        # Architectural-ablation switches (pop before super().__init__ so the
        # base class is unaffected).
        self.ablate_dual_dict     = bool(kwargs.pop("ablate_dual_dict", False))
        self.ablate_saliency_gate = bool(kwargs.pop("ablate_saliency_gate", False))
        super().__init__(**kwargs)

        proj_dim = self.projection[0].out_features
        self.score_adapter = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
        )
        # Init near zero so residual starts as identity
        nn.init.zeros_(self.score_adapter[0].bias)
        nn.init.normal_(self.score_adapter[0].weight, std=0.01)

        # -dual-dict ablation: tie W_score to W_loc so a single concept dictionary
        # is used for both attention and scoring (ConcepPath-style). Both heads
        # now share the same nn.Parameter and updates apply to one tensor.
        if self.ablate_dual_dict:
            with torch.no_grad():
                self.W_score.copy_(self.W_loc.data)
            self.W_score = self.W_loc

        # The -saliency-gate ablation is enforced inside forward(); the
        # saliency_W/attn_V/attn_U layers remain (they are unused but kept so
        # state-dict shapes stay compatible with downstream eval scripts).

    def forward(self, features, coords=None):
        if coords is not None and coords.numel() > 0:
            coords = coords.float()
            coord_range = coords.max(dim=0).values - coords.min(dim=0).values
            coord_range = coord_range.clamp(min=1.0)
            coords = (coords - coords.min(dim=0).values) / coord_range

        h = self.projection(features)
        h_norm = F.normalize(h, dim=-1)

        # ── 1. Morphological attention: pure cosine (unchanged, interpretable) ──
        w_loc_norm = F.normalize(self.W_loc[:self.n_diagnostic], dim=-1)
        loc_sim = h_norm @ w_loc_norm.T
        morph_attn = F.softmax(loc_sim / self.tau_loc, dim=0)

        # ── 2. Saliency mask ──
        a = torch.tanh(self.attn_V(h))
        b = torch.sigmoid(self.attn_U(h))
        saliency_logits = self.saliency_W(a * b)
        saliency_mask = torch.sigmoid(saliency_logits)

        # ── 3. Multiplicative fusion + renormalize (or skip for -saliency-gate ablation)
        if self.ablate_saliency_gate:
            # No gate: attention is the renormalised morph_attn alone.
            attn_weights = morph_attn / (morph_attn.sum(dim=0, keepdim=True) + 1e-8)
        else:
            raw_attn = morph_attn * saliency_mask
            attn_weights = raw_attn / (raw_attn.sum(dim=0, keepdim=True) + 1e-8)

        # ── 4. Concept scoring with ADAPTER ──
        h_adapted = h + self.score_adapter(h)  # residual
        h_adapted_norm = F.normalize(h_adapted, dim=-1)

        w_score_norm = F.normalize(self.W_score, dim=-1)
        score_sim = h_adapted_norm @ w_score_norm.T

        if self.use_normal_anchor and self.n_concepts > self.n_diagnostic:
            normal_sim = score_sim[:, self.n_diagnostic:].mean(dim=1, keepdim=True)
            diagnostic_sim = score_sim[:, :self.n_diagnostic]
            calibrated_sim = diagnostic_sim - normal_sim
        else:
            diag_score_norm = w_score_norm[:self.n_diagnostic]
            oligo_center = diag_score_norm[:self.n_oligo].mean(dim=0)
            astro_center = diag_score_norm[self.n_oligo:self.n_diagnostic].mean(dim=0)
            anchor = F.normalize(((oligo_center + astro_center) / 2).unsqueeze(0), dim=-1)
            anchor_sim = h_adapted_norm @ anchor.T
            calibrated_sim = score_sim[:, :self.n_diagnostic] - anchor_sim

        concept_scores = torch.sigmoid(calibrated_sim * self.tau)

        # ── 5. Aggregation + Prediction ──
        slide_profile = self._compute_profile(concept_scores, attn_weights, coords)

        if self.sign_constraint:
            logit = self._sign_constrained_predict_diag(slide_profile)
        else:
            logit = self.predictor(slide_profile).squeeze()
        logit_grade = self.grade_head(slide_profile).squeeze()

        self._last_saliency = saliency_mask
        self._last_concept_scores = concept_scores

        return logit, logit_grade, concept_scores, attn_weights, slide_profile
