"""RQ-Transformer for efficient latent code prediction.

Implements the two-stage transformer from the RQ-VAE paper:
1. Spatial Transformer: Processes T positions with O(T^2) complexity
2. Depth Transformer: Predicts D codes per position with O(D^2) per position

Combined complexity: O(T^2 + T*D^2) instead of O((T*D)^2) for naive approach.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get positional encoding for given sequence length."""
        return self.pe[:seq_len]


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotations to query and key vectors based on position.
    Better extrapolation properties than additive positional embeddings.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute cos and sin for efficiency
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, dim/2)

        # Duplicate for both sin and cos parts
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim)
        """
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_heads, seq_len, head_dim)
        cos: Cosine of rotation angles (seq_len, head_dim) or (batch, seq_len, head_dim)
        sin: Sine of rotation angles (seq_len, head_dim) or (batch, seq_len, head_dim)
        position_ids: Optional position indices for gathering from cos/sin cache

    Returns:
        Rotated query and key tensors
    """
    if position_ids is not None:
        # Gather cos/sin for specific positions
        cos = cos[position_ids]  # (batch, seq_len, dim)
        sin = sin[position_ids]

    # Expand for broadcasting with heads dimension
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, dim)
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len, rope_base)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with RoPE.

        Args:
            x: Input tensor (batch, seq_len, dim)
            attn_mask: Optional attention mask
            is_causal: Whether to apply causal masking
            position_ids: Optional position indices for RoPE

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE to Q and K
        cos, sin = self.rope(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Compute attention with scaled dot product
        # Use PyTorch's efficient implementation when available
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal and attn_mask is None,
            )
        else:
            # Manual implementation fallback
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if is_causal and attn_mask is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_weights.masked_fill_(causal_mask, float("-inf"))
            elif attn_mask is not None:
                attn_weights = attn_weights + attn_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm.

    Supports both standard attention and RoPE attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        causal: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.causal = causal
        self.use_rope = use_rope

        self.norm1 = nn.LayerNorm(dim)

        if use_rope:
            self.attn = RoPEAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)

        if self.use_rope:
            # RoPE attention handles causal masking internally
            attn_out = self.attn(
                normed,
                attn_mask=attn_mask,
                is_causal=self.causal,
                position_ids=position_ids,
            )
        else:
            # Standard attention
            # Create causal mask if needed
            if self.causal and attn_mask is None:
                seq_len = x.shape[1]
                attn_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )

            attn_out, _ = self.attn(
                normed, normed, normed,
                attn_mask=attn_mask,
                is_causal=self.causal and attn_mask is None,
            )

        x = x + attn_out

        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))

        return x


class SpatialTransformer(nn.Module):
    """Spatial transformer that processes T positions.

    At each position t, input is:
    - PE_T(t): positional encoding (additive) OR RoPE (applied to attention)
    - Sum of embeddings from previous position's D codes

    Outputs context vector h_t for each position.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_levels: int,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_levels = codebook_levels
        self.use_rope = use_rope

        # Embeddings for codebook indices (shared across depth levels)
        self.code_embed = nn.Embedding(codebook_size, dim)

        # Positional encoding for spatial positions (only used if not using RoPE)
        if not use_rope:
            self.pos_encoding = SinusoidalPositionalEncoding(dim, max_seq_len)
        else:
            self.pos_encoding = None

        # Start of sequence token
        self.sos_token = nn.Parameter(torch.randn(dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim, num_heads, mlp_ratio, dropout,
                causal=True,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        indices: torch.Tensor | None = None,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Compute spatial context vectors.

        Args:
            indices: Previous codes (batch, seq_len, codebook_levels) or None for inference
            seq_len: Sequence length (required if indices is None)

        Returns:
            Context vectors h_t (batch, seq_len, dim)
        """
        if indices is not None:
            batch, T, D = indices.shape
        else:
            assert seq_len is not None
            batch = 1
            T = seq_len

        device = self.code_embed.weight.device

        # Build input sequence
        # u_1 = SOS token
        # u_t = sum(embeddings of codes at position t-1)
        # With RoPE: positional info is in attention, not added to embeddings
        # Without RoPE: add sinusoidal positional encoding

        if indices is not None:
            # Get code embeddings and sum across depth
            # indices: (batch, T, D) -> embeddings: (batch, T, D, dim)
            code_embeds = self.code_embed(indices)  # (batch, T, D, dim)
            summed_embeds = code_embeds.sum(dim=2)  # (batch, T, dim)

            # Shift right: u_t uses codes from position t-1
            # First position uses SOS
            sos = repeat(self.sos_token, "d -> b 1 d", b=batch)
            shifted_embeds = torch.cat([sos, summed_embeds[:, :-1]], dim=1)  # (batch, T, dim)

            x = shifted_embeds
        else:
            # Inference mode: start with zeros, first position gets SOS
            x = torch.zeros(batch, T, self.dim, device=device)
            x[:, 0] = self.sos_token

        # Add positional encoding if not using RoPE
        if not self.use_rope and self.pos_encoding is not None:
            pos_enc = self.pos_encoding(T)  # (T, dim)
            pos_enc = repeat(pos_enc, "t d -> b t d", b=batch)
            x = x + pos_enc

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        return x


class DepthTransformer(nn.Module):
    """Depth transformer that predicts D codes for each spatial position.

    At each depth d, input is:
    - PE_D(d): depth positional encoding (learnable or RoPE)
    - h_t: spatial context from SpatialTransformer
    - Sum of embeddings from previously predicted codes at this position

    Outputs logits over codebook for each depth level.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_levels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_levels = codebook_levels
        self.use_rope = use_rope

        # Embeddings for codebook indices
        self.code_embed = nn.Embedding(codebook_size, dim)

        # Depth positional encoding (learnable) - used for additive encoding
        # Keep learnable embeddings even with RoPE to add to h_t for depth conditioning
        self.depth_pos_embed = nn.Embedding(codebook_levels, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim, num_heads, mlp_ratio, dropout,
                causal=True,
                use_rope=use_rope,
                max_seq_len=codebook_levels,  # Depth sequence is at most codebook_levels
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)

        # Output head to codebook logits
        self.output_head = nn.Linear(dim, codebook_size)

    def forward(
        self,
        h_t: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict codes at all depth levels for given spatial contexts.

        Args:
            h_t: Spatial context vectors (batch, seq_len, dim) or (batch, dim) for single position
            indices: Ground truth codes for teacher forcing (batch, seq_len, codebook_levels)
                     or None for autoregressive inference

        Returns:
            Logits over codebook (batch, seq_len, codebook_levels, codebook_size)
        """
        # Handle single position case
        if h_t.dim() == 2:
            h_t = h_t.unsqueeze(1)
            single_pos = True
        else:
            single_pos = False

        batch, T, dim = h_t.shape
        D = self.codebook_levels
        device = h_t.device

        # Get depth positional embeddings
        depth_pos = self.depth_pos_embed.weight  # (D, dim)

        if indices is not None:
            # Teacher forcing mode: process all positions and depths in parallel
            # Expand h_t to all depth levels: (batch, T, D, dim)
            h_expanded = repeat(h_t, "b t d -> b t D d", D=D)

            # Get depth positional encoding for all depths
            depth_pos_expanded = repeat(depth_pos, "D d -> b t D d", b=batch, t=T)

            # Get code embeddings for teacher forcing
            # indices: (batch, T, D)
            code_embeds = self.code_embed(indices)  # (batch, T, D, dim)

            # Shift codes: at depth d, use sum of codes 0..d-1
            # Create cumulative sum then shift
            cumsum_embeds = torch.cumsum(code_embeds, dim=2)
            # Shift right: first depth has no previous codes
            zeros = torch.zeros(batch, T, 1, dim, device=device, dtype=code_embeds.dtype)
            shifted_embeds = torch.cat([zeros, cumsum_embeds[:, :, :-1]], dim=2)

            # Combine: h_t + PE_D(d) + sum of previous code embeddings
            x = h_expanded + depth_pos_expanded + shifted_embeds  # (batch, T, D, dim)

            # Flatten for transformer: (batch * T, D, dim)
            x = rearrange(x, "b t D d -> (b t) D d")

            # Apply transformer layers
            for layer in self.layers:
                x = layer(x)

            x = self.final_norm(x)

            # Output logits
            logits = self.output_head(x)  # (batch * T, D, codebook_size)

            # Reshape back
            logits = rearrange(logits, "(b t) D c -> b t D c", b=batch, t=T)

        else:
            # Autoregressive inference mode
            # Process each depth level sequentially
            all_logits = []
            prev_code_sum = torch.zeros(batch, T, dim, device=device, dtype=h_t.dtype)

            for d in range(D):
                # Input at depth d
                x = h_t + depth_pos[d] + prev_code_sum  # (batch, T, dim)
                x = x.unsqueeze(2)  # (batch, T, 1, dim)
                x = rearrange(x, "b t 1 d -> (b t) 1 d")

                # Apply transformer (single token, so no causal mask needed)
                for layer in self.layers:
                    x = layer(x)

                x = self.final_norm(x)
                logits_d = self.output_head(x)  # (batch * T, 1, codebook_size)
                logits_d = rearrange(logits_d, "(b t) 1 c -> b t c", b=batch, t=T)

                all_logits.append(logits_d)

                # Sample and update prev_code_sum for next depth
                # Use argmax for greedy decoding
                pred_codes = logits_d.argmax(dim=-1)  # (batch, T)
                code_embeds = self.code_embed(pred_codes)  # (batch, T, dim)
                prev_code_sum = prev_code_sum + code_embeds

            logits = torch.stack(all_logits, dim=2)  # (batch, T, D, codebook_size)

        if single_pos:
            logits = logits.squeeze(1)

        return logits


class RQTransformer(nn.Module):
    """Full RQ-Transformer combining spatial and depth transformers.

    Efficient architecture for predicting RQ-VAE latent codes:
    - Spatial transformer: O(T^2) for T positions
    - Depth transformer: O(D^2) per position, O(T*D^2) total
    - Combined: O(T^2 + T*D^2) vs O((T*D)^2) for naive approach

    Supports both standard sinusoidal positional encoding and RoPE.
    """

    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 512,
        codebook_levels: int = 8,
        spatial_layers: int = 12,
        depth_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        """
        Args:
            dim: Model dimension
            codebook_size: Size of each codebook (K)
            codebook_levels: Number of RQ depth levels (D)
            spatial_layers: Number of layers in spatial transformer
            depth_layers: Number of layers in depth transformer
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            use_rope: Whether to use Rotary Position Embeddings instead of sinusoidal
            rope_base: Base frequency for RoPE (default 10000)
        """
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_levels = codebook_levels
        self.use_rope = use_rope

        self.spatial_transformer = SpatialTransformer(
            dim=dim,
            codebook_size=codebook_size,
            codebook_levels=codebook_levels,
            num_layers=spatial_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        self.depth_transformer = DepthTransformer(
            dim=dim,
            codebook_size=codebook_size,
            codebook_levels=codebook_levels,
            num_layers=depth_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base,
        )

    def forward(
        self,
        indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with teacher forcing.

        Args:
            indices: Ground truth codes (batch, seq_len, codebook_levels)

        Returns:
            Dictionary containing:
                - logits: Predicted logits (batch, seq_len, codebook_levels, codebook_size)
                - loss: Cross-entropy loss
        """
        batch, T, D = indices.shape

        # Spatial transformer: get context for each position
        h = self.spatial_transformer(indices)  # (batch, T, dim)

        # Depth transformer: predict codes at each depth
        logits = self.depth_transformer(h, indices)  # (batch, T, D, codebook_size)

        # Compute loss
        loss = F.cross_entropy(
            rearrange(logits, "b t d c -> (b t d) c"),
            rearrange(indices, "b t d -> (b t d)"),
        )

        return {
            "logits": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        seq_len: int,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Generate latent codes autoregressively.

        Args:
            seq_len: Number of spatial positions to generate
            batch_size: Batch size
            temperature: Sampling temperature
            top_k: Top-k sampling (None to disable)
            top_p: Nucleus sampling threshold (None to disable)
            device: Device to generate on

        Returns:
            Generated codes (batch, seq_len, codebook_levels)
        """
        if device is None:
            device = next(self.parameters()).device

        D = self.codebook_levels
        generated = []

        # Generate position by position
        for t in range(seq_len):
            if t == 0:
                # First position: use SOS
                h_t = self.spatial_transformer.sos_token.unsqueeze(0)  # (1, dim)
                h_t = repeat(h_t, "1 d -> b d", b=batch_size)
                # Add positional encoding only if not using RoPE
                if not self.use_rope and self.spatial_transformer.pos_encoding is not None:
                    h_t = h_t + self.spatial_transformer.pos_encoding(1)[0]
            else:
                # Use previous codes to compute spatial context
                prev_indices = torch.stack(generated, dim=1)  # (batch, t, D)
                h = self.spatial_transformer(prev_indices)  # (batch, t, dim)
                h_t = h[:, -1]  # (batch, dim) - context for current position

                # Add positional encoding for current position only if not using RoPE
                if not self.use_rope and self.spatial_transformer.pos_encoding is not None:
                    h_t = h_t + self.spatial_transformer.pos_encoding(t + 1)[t]

            # Generate D codes for this position using depth transformer
            codes_t = []
            prev_code_sum = torch.zeros(batch_size, self.dim, device=device)

            for d in range(D):
                # Input for depth d
                depth_pos = self.depth_transformer.depth_pos_embed.weight[d]
                x = h_t + depth_pos + prev_code_sum  # (batch, dim)
                x = x.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, dim)
                x = rearrange(x, "b 1 1 d -> b 1 d")

                # Apply depth transformer layers
                for layer in self.depth_transformer.layers:
                    x = layer(x)

                x = self.depth_transformer.final_norm(x)
                logits = self.depth_transformer.output_head(x)  # (batch, 1, codebook_size)
                logits = logits.squeeze(1)  # (batch, codebook_size)

                # Apply temperature
                logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    top_k_vals, _ = logits.topk(top_k, dim=-1)
                    logits[logits < top_k_vals[:, -1:]] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                code = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch,)
                codes_t.append(code)

                # Update prev_code_sum
                code_embed = self.depth_transformer.code_embed(code)  # (batch, dim)
                prev_code_sum = prev_code_sum + code_embed

            # Stack codes for this position
            codes_t = torch.stack(codes_t, dim=-1)  # (batch, D)
            generated.append(codes_t)

        # Stack all positions
        generated = torch.stack(generated, dim=1)  # (batch, seq_len, D)

        return generated

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
