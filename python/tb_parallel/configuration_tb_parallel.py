"""Configuration for the cl-transformer-blocks portable parallel decoder."""

from transformers import PretrainedConfig


class TBParallelConfig(PretrainedConfig):
    """Version-one portable causal LM configuration emitted by Common Lisp."""

    model_type = "tb_parallel"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=None,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        parallel_residual=True,
        portable_architecture_version=1,
        initializer_range=0.02,
        use_cache=False,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.parallel_residual = parallel_residual
        self.portable_architecture_version = portable_architecture_version
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        if portable_architecture_version != 1 or parallel_residual is not True:
            raise ValueError("unsupported portable architecture version or residual topology")
        if any(
            value <= 0
            for value in (
                vocab_size,
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                self.head_dim,
                max_position_embeddings,
            )
        ):
            raise ValueError("portable dimensions must be positive")
        if (
            num_attention_heads <= 0
            or num_key_value_heads <= 0
            or self.head_dim <= 0
            or self.head_dim % 2
            or num_attention_heads % num_key_value_heads
            or hidden_size != num_attention_heads * self.head_dim
        ):
            raise ValueError("invalid portable grouped-attention dimensions")
        if (
            hidden_act != "silu"
            or attention_bias is not False
            or mlp_bias is not False
            or attention_dropout != 0
            or use_cache is not False
        ):
            raise ValueError(
                "portable v1 requires SiLU, bias-free projections, zero dropout, "
                "and uncached Python execution"
            )
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
