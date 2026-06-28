"""
MorphoPath backbone: dual-dictionary decoupled concept-attention MIL.

Key innovation: Two separate CONCH-initialized concept vector sets:
  W_loc  — pure morphology prompts → drives attention (WHERE to look)
  W_score — label-anchored prompts → drives concept scoring (WHAT direction)

Pipeline:
  UNI patches [N, 1024]
    → Projection [N, 512]
    → W_loc cosine sim → Gated Attention bias → attn_weights [N, K]
    → W_score cosine sim → concept_scores [N, K]
    → Attended aggregation: attn × score → slide_profile [3K]
    → Sign-constrained linear predictor → logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphoPathBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        proj_dim: int = 512,
        attn_dim: int = 256,
        n_concepts: int = 8,
        n_oligo: int = 4,
        conch_loc_path: str = None,
        conch_score_path: str = None,
        dropout: float = 0.1,
        tau_score: float = 5.0,
        lambda_loc_bias: float = 1.0,
        sign_constraint: bool = True,
        attn_mode: str = "gated_bias",
        tau_loc: float = 1.0,
    ):
        """
        Args:
            conch_loc_path:   Path to pure-morphology CONCH embeddings (W_loc init)
            conch_score_path: Path to label-anchored CONCH embeddings (W_score init)
            tau_score:        Sigmoid temperature for concept scoring (lower = less saturated)
            lambda_loc_bias:  Weight of W_loc cosine sim as attention bias (mode=gated_bias)
            attn_mode:        "gated_bias" (W_loc + gated network) or "cosine" (pure W_loc cosine sim)
            tau_loc:          Temperature for cosine attention (mode=cosine only)
        """
        super().__init__()
        self.n_concepts = n_concepts
        self.n_oligo = n_oligo
        self.proj_dim = proj_dim
        self.tau_score = tau_score
        self.lambda_loc_bias = lambda_loc_bias
        self.sign_constraint = sign_constraint
        self.attn_mode = attn_mode
        self.tau_loc = tau_loc

        # --- Projection ---
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

        # --- W_loc: pure morphology vectors (drives attention) ---
        loc_init = self._load_conch(conch_loc_path, n_concepts, proj_dim, "W_loc")
        self.W_loc = nn.Parameter(loc_init)
        self.register_buffer("conch_loc_anchor", loc_init.clone())

        # --- W_score: label-anchored vectors (drives scoring) ---
        score_init = self._load_conch(conch_score_path, n_concepts, proj_dim, "W_score")
        self.W_score = nn.Parameter(score_init)
        self.register_buffer("conch_score_anchor", score_init.clone())

        # --- Gated Attention Network ---
        self.attn_V = nn.Linear(proj_dim, attn_dim)
        self.attn_U = nn.Linear(proj_dim, attn_dim)
        self.attn_W = nn.Linear(attn_dim, n_concepts)

        # --- Sign-constrained predictor ---
        profile_dim = 3 * n_concepts
        self.predictor = nn.Linear(profile_dim, 1)
        self.grade_head = nn.Linear(profile_dim, 1)

        # --- Exclusion mask (dynamic based on n_oligo) ---
        mask = torch.zeros(n_concepts, n_concepts)
        for i in range(n_concepts):
            for j in range(n_concepts):
                if (i < n_oligo) != (j < n_oligo):
                    mask[i, j] = 1.0
        self.register_buffer("exclusion_mask", mask)

        self._init_predictor_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_predictor_weights(self):
        with torch.no_grad():
            w = self.predictor.weight
            K = self.n_concepts
            n_o = self.n_oligo
            w[0, :n_o] = 0.1
            w[0, n_o:K] = -0.1
            w[0, K:K+n_o] = 0.05
            w[0, K+n_o:2*K] = -0.05
            w[0, 2*K:] = 0.0
            self.predictor.bias.zero_()

    @staticmethod
    def _load_conch(path, n_concepts, proj_dim, name=""):
        if path is not None and path.upper() != "NONE":
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                emb = ckpt["embeddings"]
                if emb.shape[0] >= n_concepts and emb.shape[1] == proj_dim:
                    emb = emb[:n_concepts]
                    print(f"[MorphoPathBackbone] Loaded {name} from {path} "
                          f"({n_concepts}/{ckpt['embeddings'].shape[0]} concepts)", flush=True)
                    return emb.float()
                raise ValueError(f"Shape mismatch for {name}")
            except Exception as e:
                raise RuntimeError(
                    f"[MorphoPathBackbone] Failed to load {name} from {path}: {e}"
                )
        print(f"[MorphoPathBackbone] {name}: random init", flush=True)
        w = torch.randn(n_concepts, proj_dim)
        return F.normalize(w, dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, features, coords=None):
        """
        Returns:
            logit, logit_grade, concept_scores, attn_weights, slide_profile
        """
        # Normalize coords
        if coords is not None and coords.numel() > 0:
            coords = coords.float()
            coord_range = coords.max(dim=0).values - coords.min(dim=0).values
            coord_range = coord_range.clamp(min=1.0)
            coords = (coords - coords.min(dim=0).values) / coord_range

        # 1. Projection
        h = self.projection(features)                             # [N, proj_dim]
        h_norm = F.normalize(h, dim=-1)

        # 2. W_loc cosine similarity
        w_loc_norm = F.normalize(self.W_loc, dim=-1)              # [K, proj_dim]
        loc_sim = h_norm @ w_loc_norm.T                           # [N, K]

        # 3. Attention (three modes)
        if self.attn_mode == "cosine":
            # Mode A: pure W_loc cosine similarity → attention
            attn_weights = F.softmax(loc_sim / self.tau_loc, dim=0)   # [N, K]
        elif self.attn_mode == "residual":
            # Mode A+G: W_loc as base + gated as small residual
            a = torch.tanh(self.attn_V(h))                        # [N, attn_dim]
            b = torch.sigmoid(self.attn_U(h))                     # [N, attn_dim]
            gated_output = self.attn_W(a * b)                     # [N, K]
            attn_logits = loc_sim / self.tau_loc + self.lambda_loc_bias * gated_output
            attn_weights = F.softmax(attn_logits, dim=0)          # [N, K]
        else:
            # Mode B: Gated Attention + loc_sim bias (gated dominates)
            a = torch.tanh(self.attn_V(h))                        # [N, attn_dim]
            b = torch.sigmoid(self.attn_U(h))                     # [N, attn_dim]
            attn_logits = self.attn_W(a * b) + self.lambda_loc_bias * loc_sim
            attn_weights = F.softmax(attn_logits, dim=0)          # [N, K]

        # 4. W_score concept scoring
        w_score_norm = F.normalize(self.W_score, dim=-1)          # [K, proj_dim]
        score_sim = h_norm @ w_score_norm.T                       # [N, K]
        concept_scores = torch.sigmoid(score_sim * self.tau_score) # [N, K]

        # 5. Attended aggregation
        slide_profile = self._compute_attended_profile(
            concept_scores, attn_weights, coords
        )

        # 6. Prediction
        if self.sign_constraint:
            logit = self._sign_constrained_predict(slide_profile)
        else:
            logit = self.predictor(slide_profile).squeeze()
        logit_grade = self.grade_head(slide_profile).squeeze()

        return logit, logit_grade, concept_scores, attn_weights, slide_profile

    def _compute_attended_profile(self, concept_scores, attn_weights, coords):
        weighted_score = (attn_weights * concept_scores).sum(dim=0)
        peak = concept_scores.max(dim=0).values
        spread = self._attended_spatial_spread(attn_weights, coords)
        return torch.cat([weighted_score, peak, spread])

    def _attended_spatial_spread(self, attn_weights, coords):
        if coords is None:
            return torch.zeros(self.n_concepts, device=attn_weights.device)
        centroid = attn_weights.T @ coords.float()
        diffs = coords.float().unsqueeze(1) - centroid.unsqueeze(0)
        dists = diffs.norm(dim=-1)
        spread = (attn_weights * dists).sum(dim=0)
        return spread

    def _sign_constrained_predict(self, slide_profile):
        w = self.predictor.weight
        K = self.n_concepts
        n_o = self.n_oligo
        constrained_blocks = []
        for block_start in [0, K, 2 * K]:
            block = torch.cat([
                F.softplus(w[0, block_start:block_start + n_o]),
                -F.softplus(-w[0, block_start + n_o:block_start + K]),
            ])
            constrained_blocks.append(block)
        w_constrained = torch.cat(constrained_blocks).unsqueeze(0)
        logit = F.linear(slide_profile, w_constrained, self.predictor.bias)
        return logit.squeeze()

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------
    def loss_align_loc(self):
        """L_align for W_loc."""
        w = F.normalize(self.W_loc, dim=-1)
        a = F.normalize(self.conch_loc_anchor, dim=-1)
        return -(w * a).sum(dim=-1).mean()

    def loss_align_score(self):
        """L_align for W_score."""
        w = F.normalize(self.W_score, dim=-1)
        a = F.normalize(self.conch_score_anchor, dim=-1)
        return -(w * a).sum(dim=-1).mean()

    def loss_attn_div(self, attn_weights):
        """Prior-masked attention diversity."""
        C = attn_weights.T @ attn_weights
        eye = torch.eye(self.n_concepts, device=C.device)
        diff = (C - eye) ** 2
        masked_diff = self.exclusion_mask.to(C.device) * diff
        return masked_diff.sum()

    def concept_diagnostics(self):
        """Per-concept cos_sim for both W_loc and W_score."""
        with torch.no_grad():
            w_loc = F.normalize(self.W_loc, dim=-1)
            a_loc = F.normalize(self.conch_loc_anchor, dim=-1)
            cos_loc = (w_loc * a_loc).sum(dim=-1).cpu()

            w_score = F.normalize(self.W_score, dim=-1)
            a_score = F.normalize(self.conch_score_anchor, dim=-1)
            cos_score = (w_score * a_score).sum(dim=-1).cpu()

        return cos_loc, cos_score
