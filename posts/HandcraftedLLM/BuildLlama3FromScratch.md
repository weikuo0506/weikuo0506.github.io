Previously, we handwrote the code for GPT2 and Llama2 from scratch. Now, we will make modifications based on the Llama2 code and present the model code for Llama3.

For a comparison of the architectures of GPT and Llama3, please refer to[ the link ](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-1.11/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html)below:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8b3768de929e409cbb1cc2d695829313~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752892518&x-orig-sign=Ff28cXheYNAszo%2FTh3cNvFjRl0I%3D)

Main differences:

-   Position Encoding has changed from Learned Absolute Positional Embeddings to RoPE. This is the same as Llama2, but Llama3's RoPE introduces a frequency-varying mechanism (Dynamic NTK scaling or Multi-Scale RoPE).
-   The attention mechanism has changed from MHA to Grouped Query Attention, which also differs from Llama2.
-   Tokenizer uses a Tiktoken-compatible tokenizer, which is different from Llama2.
-   Other changes, such as RMSNorm, SwiGLU layer, etc., are all the same as Llama2.

# Scaled RoPE

Compared to Llama2's original RoPE, Llama3 uses a variant of RoPE, namely RoPE-scaling, which employs frequency scaling technology and can support longer contexts.

The original RoPE (Rotary Positional Embedding) may experience a decline in the accuracy of positional encoding due to the rapid oscillation of high-frequency dimensions when the context length increases. Scaled RoPE enhances the robustness of positional encoding for long contexts by adjusting the frequency distribution (such as smoothing intermediate frequencies and stretching low-frequency periods).

Scaled RoPE modifies the position of the incoming `RoPE(x)` from position \( x \) to position \( \frac{x}{\alpha} \), that is:

\[
RoPE(x) \rightarrow RoPE\left(\frac{x}{\alpha}\right)
\]



The core steps are:

-   Calculate the "wavelength" for each frequency: convert `inv_freq` (used for frequency encoding in RoPE) into the corresponding **wavelength** for each frequency.

    -   `inv_freq` The larger the value, the shorter the wavelength (higher frequency).
    -   `inv_freq` The smaller it is, the longer the wavelength (low frequency)

-   Define low-frequency/high-frequency thresholds:

    -   Wavelength > low_wavelen ⇒ Low frequency (insensitive, suitable for scaling)
    -   Wavelength < high_wavelen ⇒ is high frequency (keep original value)
    -   The middle interval is the intermediate frequency (gradually transitioning from scaling to non-scaling)

-   Process the low-frequency part (direct scaling)

    -   **If it is the low-frequency part** (long wavelength), then for ` inv_freq  `**divide by scale_factor** , i.e., the fluctuations slow down (more robust)
    -   **Otherwise, remain unchanged**

-   Smooth the intermediate frequency part

    -   `smooth_factor=1` ⇒ fully retain the original `inv_freq`
    -   `smooth_factor=0` ⇒ fully use scaled `inv_freq / scale_factor`
    -   Intermediate ⇒ Interpolation

-   Apply smoothing frequency: perform linear interpolation: gradually transition from the scaled version to the original value.

-   Final Fusion:

    -   If it is **intermediate frequency** ⇒ use `smoothed_inv_freq`
    -   If it is **low frequency** ⇒ use `inv_freq / scale_factor`
    -   If it is **high frequency** ⇒ retain `inv_freq`

The overall achieved effect is: low frequency → frequency is **divided by scale_factor (e.g., 8)** → **frequency decreases, wavelength increases** → encoding changes more slowly → adapts to longer distances.

Notably, the frequency here has no relation to the position of the token, but is related to the numbering of the embedding dimensions. Low frequency corresponds to the latter part of the embedding dimensions (dimensions with higher indices).

Intuitively feel the difference before and after scaling, the code is as follows:

```
import numpy as np
import matplotlib.pyplot as plt

# Parameters
base = 500000  # RoPE base used in LLaMA 3
scale_factor = 2.0  # Scaling factor for low frequencies
d = 512        # Total embedding dimension
positions = np.arange(0, 1000)  # Token position range

# Frequency index range: 0 to d/2 - 1 (each frequency corresponds to 2 embedding dims)
i_high = 0      # High frequency (fast variation, low dimension index)
i_mid  = 32    # Mid frequency
i_low  = 64    # Low frequency (slow variation, high dimension index)

# Calculate corresponding angular frequency ω_i = 1 / base^{2i/d}
def calc_omega(i):
    return 1 / (base ** (2 * i / d))

omega_high = calc_omega(i_high)
omega_mid = calc_omega(i_mid)
omega_low = calc_omega(i_low)

# Piecewise scaling
omega_high_scaled = omega_high  # High frequency remains unchanged
smooth_factor = 0.5             # Interpolation factor between mid and low frequencies
omega_mid_scaled = smooth_factor * omega_mid + (1 - smooth_factor) * omega_low
omega_low_scaled = omega_low / scale_factor  # Scale down low frequency (make frequency smaller, wavelength longer)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Standard RoPE plot
ax1.plot(positions, np.sin(omega_high * positions), label=f'High freq (i={i_high}, ω={omega_high:.2e})', color='blue')
ax1.plot(positions, np.sin(omega_mid * positions), label=f'Mid freq (i={i_mid}, ω={omega_mid:.2e})', color='green')
ax1.plot(positions, np.sin(omega_low * positions), label=f'Low freq (i={i_low}, ω={omega_low:.2e})', color='red')
ax1.set_title("Standard RoPE: sin(ω × position)")
ax1.set_ylabel("sin(ω × position)")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Scaled RoPE plot
ax2.plot(positions, np.sin(omega_high_scaled * positions), label=f'High freq (i={i_high}, ω={omega_high_scaled:.2e})', color='blue')
ax2.plot(positions, np.sin(omega_mid_scaled * positions), label=f'Mid freq scaled (i={i_mid}, ω={omega_mid_scaled:.2e})', color='green')
ax2.plot(positions, np.sin(omega_low_scaled * positions), label=f'Low freq scaled (i={i_low}, ω={omega_low_scaled:.2e})', color='red')
ax2.set_title(f"Segmented Scaled RoPE (scale_factor={scale_factor}, smooth_factor={smooth_factor})")
ax2.set_xlabel("Position")
ax2.set_ylabel("sin(ω × position)")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/dc52f50081194638be782855cdbc9be8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752892518&x-orig-sign=Jvx3CxpvlTM78uF3cN3941Kt9Jc%3D)

The complete code for rope-scaling and precomputing rope is as follows:

```
from typing import Optional

import torch

def precompute_rope_params(
    context_length: int,
    head_dim: int,
    theta_base: float = 500000.0,  # Default base for LLaMA 3
    freq_config: Optional[dict] = None,
):
    """
Precompute sin and cos tensors for RoPE with optional frequency scaling/smoothing.

Args:
context_length: Sequence length
head_dim: Embedding dimension (must be even)
theta_base: Base for inverse frequency calculation (default 500000)
freq_config: Optional dict with keys:
- original_context_length: int, original training context length
- low_freq_factor: float, low frequency threshold factor (>1)
- high_freq_factor: float, high frequency threshold factor (>1)
- factor: float, scaling factor (>1)

Returns:
sin, cos: Tensors of shape (seq_len, half_dim)
"""
assert head_dim % 2 == 0, "head_dim must be even"

    half_dim = head_dim // 2
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))

    if freq_config is not None:
        # Extract frequency config parameters
        orig_len = freq_config["original_context_length"]
        low_factor = freq_config["low_freq_factor"]
        high_factor = freq_config["high_freq_factor"]
        scale_factor = freq_config["factor"]

        # Compute wavelength
        wavelen = 2 * torch.pi / inv_freq
        low_wavelen = orig_len / low_factor
        high_wavelen = orig_len / high_factor

        # Scale inverse frequencies for low frequency bands
        condition = wavelen > low_wavelen
        inv_freq_scaled = torch.where(condition, inv_freq / scale_factor, inv_freq)

        # Compute smooth factor for medium frequency band
        smooth_factor = (orig_len / wavelen - low_factor) / (high_factor - low_factor)
        smooth_factor = smooth_factor.clamp(0.0, 1.0)
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / scale_factor) + smooth_factor * inv_freq

        # Apply smoothed frequencies for medium band
        is_medium = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
        inv_freq = torch.where(is_medium, smoothed_inv_freq, inv_freq_scaled)

    # Compute position angles
    positions = torch.arange(context_length, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)  # Shape: (seq_len, half_dim)
    return torch.sin(angles), torch.cos(angles)
```

The context and rope base used by Llama3 are as follows:

```
# Instantiate RoPE parameters

llama_3_context_len = 8192
llama_3_theta_base = 500_000
```

Construct an example to calculate RoPE, as follows:

```
from Llama2_v1 import rotary_pos_emb

# Settings
batch_size = 2
num_heads = 4
head_dim = 16

# Instantiate RoPE parameters
cos, sin = precompute_rope_params(
    head_dim=head_dim,
    theta_base=llama_3_theta_base,
    context_length=llama_3_context_len
)

# Dummy query and key tensors
torch.manual_seed(123)
queries = torch.randn(batch_size, num_heads, llama_3_context_len, head_dim)
keys = torch.randn(batch_size, num_heads, llama_3_context_len, head_dim)

# Apply rotary position embeddings
queries_rot = rotary_pos_emb(queries, cos, sin)
keys_rot = rotary_pos_emb(keys, cos, sin)

print("shape of queries:", queries.shape)
print("shape of keys:", keys.shape)
```

The results are as follows:

```
shape of queries: torch.Size([2, 4, 8192, 16])
shape of keys: torch.Size([2, 4, 8192, 16])
```

To optimize performance and resource utilization, we usually store precomputed tensors such as `cos` and `sin` in buffers. Buffers are a special type of variable in the model, which are different from model parameters (parameters) and do not participate in layer computation and the training process. They are typically used to store intermediate variables that do not need to be updated but are frequently used in inference and forward propagation.

```
class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
            cos, sin = precompute_rope_params(context_length, head_dim, rope_base, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]
```

# Grouped-query attention

According to the[paper](https://arxiv.org/pdf/2305.13245)of GQA, the following comparison chart shows the attention mechanism of each model:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c120d3f2b92d49c38323e3895998ece0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752892518&x-orig-sign=AHikxsR53XmJjkczjdPFi9TQ%2BD0%3D)

The differences are obvious at a glance:

-   The number of queries, keys, and values in MHA is the same;
-   The number of queries in MQA remains unchanged, but the number of keys and values is only 1, meaning that all heads share the same key and value;
-   GQA is an intermediate state between the two, where keys and values share weights in groups.

In other words, GQA introduces the concept of grouped sharing, under which both MHA and MQA can be regarded as special cases (kv_groups = heads or kv_groups = 1). The main purpose of this is to save a significant amount of computational and memory overhead without significantly degrading performance.

Once the above ideas are understood, the code for GQA becomes very simple, as follows:

```
from torch import nn

class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,
            rope_base=10_000,
            rope_config=None,
            dtype=None
        ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        self.group_size = num_heads // num_kv_groups
        log.debug(f"d_out={self.d_out}, num_heads={self.num_heads}, num_kv_groups={self.num_kv_groups}, head_dim={self.head_dim}, group_size={self.group_size}")

        linear_kwargs = dict(bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, **linear_kwargs)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, **linear_kwargs)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, **linear_kwargs)
        self.out_proj = nn.Linear(d_out, d_out, **linear_kwargs)

        mask, cos, sin = SharedBuffers.get_buffers(
            context_length, self.head_dim, rope_base, rope_config, dtype
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, seq_len, _ = x.shape
        log.debug("shape of x: %s", x.shape)

        queries = self.W_query(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        log.debug("shape of queries: %s", queries.shape)
        log.debug("shape of keys: %s", keys.shape)

        # Apply rotary positional embeddings
        queries = rotary_pos_emb(queries, self.cos, self.sin)
        keys = rotary_pos_emb(keys, self.cos, self.sin)
        log.debug("shape of queries: %s", queries.shape)

        # Repeat keys and values to match num_heads
        keys = keys.repeat_interleave(self.group_size, dim=1)  # (b, num_heads, seq_len, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)
        log.debug("shape of keys: %s", keys.shape)
        log.debug("shape of values: %s", values.shape)

        # Compute attention scores with causal mask
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        log.debug("shape of attn_scores: %s", attn_scores.shape)

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        log.debug("shape of attn_weights: %s", attn_weights.shape)
        assert keys.shape[-1] == self.head_dim

        context = torch.matmul(attn_weights, values)  # (b, num_heads, seq_len, head_dim)
        log.debug("shape of context: %s", context.shape)
        context = context.transpose(1, 2).reshape(b, seq_len, self.d_out)
        log.debug("shape of context: %s", context.shape)

        out = self.out_proj(context)
        log.debug("shape of out: %s", out.shape)
        return out
```

We perform the following calculation example:

```
batch_size = 2
context_len = 3000
max_context_len = 8192
embed_dim = 4096
num_heads = 32

example_batch = torch.randn((batch_size, context_len, embed_dim))

gqa = GroupedQueryAttention(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=max_context_len,
    num_heads=num_heads,
    num_kv_groups=8,
    rope_base=llama_3_theta_base
)

gqa(example_batch)

print("W_query:", gqa.W_query.weight.shape)
print("W_key:", gqa.W_key.weight.shape)
print("W_value:", gqa.W_value.weight.shape)
```

The result is:

```
W_query: torch.Size([4096, 4096])
W_key: torch.Size([1024, 4096])
W_value: torch.Size([1024, 4096])
```

As can be seen, 32 heads are divided into 8 groups, which means that every 4 heads share keys and values. Therefore, it is equivalent to reducing the number of keys and values to 1/4 of the original. The embedding dimension of the query remains unchanged, the same as in MHA, but the embedding dimension of keys and values is reduced from [4096, 4096] in MHA to [1024, 4096].

If interested, you can calculate and compare the number of parameters of the two. In the above example, the total number of parameters of MHA is 67,108,864, while that of GQA is 41,943,040, a reduction of approximately 40%.

# Update Transformer Block with GQA

Since rope scaling and GQA have been introduced, the Transformer code needs to be simply updated as follows:

```
from Llama2_v1 import FeedForward, RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],  # NEW
            rope_base=cfg["rope_base"],        # NEW
            rope_config=cfg["rope_freq"],      # NEW
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)

        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-5)
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x
```

# Update Model

Llama3 is almost identical to Llama2 in terms of the model, and only requires updating the model code as follows:

```
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits
```

# Initialize Model

The configuration used by Llama3 is as follows:

```
LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # Increased vocabulary size for broader language coverage
    "context_length": 8192,  # Extended context window for handling longer sequences
    "emb_dim": 4096,         # Embedding dimension for token representations
    "n_heads": 32,           # Number of attention heads in each self-attention layer
    "n_layers": 32,          # Total number of transformer blocks
    "hidden_dim": 14_336,    # Expanded feedforward network dimension (MLP inner size)
    "n_kv_groups": 8,        # Number of key-value groups for grouped-query attention (GQA)
    "rope_base": 500_000.0,  # Higher RoPE base to better encode longer positions
    "rope_freq": None,       # Optional override for RoPE frequency scaling
    "dtype": torch.bfloat16  # Use bfloat16 for lower memory usage and faster compute
}
```

Load the model as follows:

```
model = Llama3Model(LLAMA3_CONFIG_8B)
```

If interested, you can calculate the total number of model parameters and the total memory required for different types, as follows:

```
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```

```
def model_memory_size(model, input_dtype=torch.float32):
    element_size = torch.tensor([], dtype=input_dtype).element_size()
    total_elements = sum(p.numel() * (1 + int(p.requires_grad)) for p in model.parameters())
    total_elements += sum(b.numel() for b in model.buffers())
    return total_elements * element_size / (1024 ** 3)

print(f"float32 (PyTorch default): {model_memory_size(model, torch.float32):.2f} GB")
print(f"bfloat16: {model_memory_size(model, torch.bfloat16):.2f} GB")
```

The current model has a total of 8,030,261,248 parameters, which is 8B.

Under the default float32 precision, 60GB of memory is required, while under bfloat16 precision, 30GB of memory is needed. For the following demonstration, we will use bfloat16 precision.

# Load Tokenizer

LLaMA 3 uses a custom Byte Pair Encoding (BPE) tokenizer, implemented based on SentencePiece, which is different from OpenAI's cl100k_base, p50k_base, and r50k_base tokenizers.

tiktoken.get_encoding() is designed for OpenAI models (such as GPT-3.5, GPT-4), loads a predefined BPE tokenizer, and relies on a specific Regular Expression (pat_str) for text pre-segmentation.

The tokenizer of LLaMA 3 defines merge rules through the proprietary tokenizer.model file, which requires explicit loading (e.g., via the transformers library of Hugging Face) and cannot use the default tokenizer of tiktoken.get_encoding().

The BPE tokenizer of LLaMA 3 does not rely on explicit Regular Expression pre-segmentation, but directly trains merge rules at the Unicode character or byte level, adapting to the characteristics of its training data (such as multilingual text, code, and academic data), and handling Unicode characters, line breaks, and punctuation more flexibly.

Therefore, the tokenizer of LLaMA 3 cannot directly use the fixed Regular Expression (pat_str) or tokenizer of tiktoken, and it is necessary to load its custom BPE model to ensure that tokenization is consistent with Model Training.

The tokenizer used by Llama3 can be implemented with the following code:

```
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    """
Tokenizer wrapper for LLaMA 3 using custom tiktoken BPE files.

Automatically loads custom merge rules, special tokens, and regex-based tokenization pattern.
"""

def __init__(self, model_path: str):
        """
Initialize the tokenizer with a given BPE model file.

Args:
model_path (str): Path to the .tiktoken file used by LLaMA 3.
"""
model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Tokenizer model file not found: {model_path}")

        # Load mergeable BPE ranks from file
        mergeable_ranks = load_tiktoken_bpe(str(model_path))

        # Define special token IDs
        special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }

        # Add reserved special tokens from 128002 to 128257 (excluding used IDs)
        special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i
            for i in range(256)
            if (128002 + i) not in special_tokens.values()
        })

        # Regex pattern string used for LLaMA-style tokenization
        pat_str = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            r"[^\r\n\p{L}\p{N}]?\p{L}+|"
            r"\p{N}{1,3}|"
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
            r"\s*[\r\n]+|"
            r"\s+(?!\S)|"
            r"\s+"
        )

        self.special_tokens = special_tokens

        # Create the tiktoken Encoding instance
        self.model = tiktoken.Encoding(
            name=model_path.name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

    def encode(self, text: str, bos: bool = False, eos: bool = False,
               allowed_special: set = set(), disallowed_special=()) -> list[int]:
        """
Encode a text string into token IDs.

Args:
text (str): Input string to tokenize.
bos (bool): Whether to prepend <|begin_of_text|> token.
eos (bool): Whether to append <|end_of_text|> token.
allowed_special (set): Set of allowed special token strings.
disallowed_special: Set or policy for disallowed tokens.

Returns:
List[int]: Token ID list.
"""
tokens = []
        if bos:
            tokens.append(self.special_tokens["<|begin_of_text|>"])

        tokens += self.model.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special
        )

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
Decode a list of token IDs back into text.

Args:
tokens (List[int]): Token ID list.

Returns:
str: Decoded string.
"""
return self.model.decode(tokens)
```

However, we need to download the tokenizer.model file from HuggingFace, as follows:

```
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_tokenizer_if_needed(repo_id: str, filename: str, local_dir: str) -> str:
    local_path = Path(local_dir) / filename
    if local_path.exists():
        print(f"Tokenizer file {local_path} already exists, skipping.")
        return str(local_path)

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir
    )

# Example usage
tokenizer_file_path = download_tokenizer_if_needed(
    repo_id="meta-llama/Meta-Llama-3-8B",
    filename="original/tokenizer.model",
    local_dir="Llama-3-8B"
)
```

Initialize the tokenizer and run the example as follows:

```
tokenizer = Tokenizer(tokenizer_file_path)
```

```
# Encode with BOS and EOS tokens
tokens = tokenizer.encode("Hello world!",bos=True,eos=False)
print(tokens)

# Decode back to text
text = tokenizer.decode(tokens)
print(text)
```

The results are as follows:

```
[128000, 9906, 1917, 0]
<|begin_of_text|>Hello world!
```

Additionally, we can also directly use HuggingFace's AutoTokenizer API to load it, with the same result, as follows:

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokens = tokenizer.encode("Hello world!")
print(tokens)
# Decode back to text
text = tokenizer.decode(tokens)
print(text)
```

We can also simply compare the differences in the results between the Llama3 and GPT2 tokenizers, as follows:

```
from transformers import AutoTokenizer
import tiktoken

text = "hello\nworld, 世界！"
# LLaMA 3 tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
llama_tokens = llama_tokenizer.encode(text)
print("LLaMA 3 tokens:", llama_tokens)
print("LLaMA 3 decoded:", llama_tokenizer.decode(llama_tokens))

# tiktoken using gpt or cl100k_base
tiktoken_encoding = tiktoken.get_encoding("gpt2")
# tiktoken_encoding = tiktoken.get_encoding("cl100k_base")

tiktoken_tokens = tiktoken_encoding.encode(text)
print("tiktoken tokens:", tiktoken_tokens)
print("tiktoken decoded:", tiktoken_encoding.decode(tiktoken_tokens))
```

The results are as follows:

```
LLaMA 3 tokens: [128000, 15339, 198, 14957, 11, 127365, 6447]
LLaMA 3 decoded: <|begin_of_text|>hello
world, 世界！
tiktoken tokens: [31373, 198, 6894, 11, 220, 10310, 244, 45911, 234, 171, 120, 223]
tiktoken decoded: hello
world, 世界！
```

# Load pretrained weights

Similar to Llama2, we need to first download the public weights of Llama3 from HuggingFace (apply for permission required). The code is as follows:

```
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def load_combined_weights(repo_id, filenames, local_dir):
    combined = {}
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for name in filenames:
        local_path = local_dir / name
        if not local_path.exists():
            # download if not already present
            hf_hub_download(
                repo_id=repo_id,
                filename=name,
                local_dir=str(local_dir)
            )
        weights = load_file(str(local_path))
        combined.update(weights)

    return combined

# Use the function
filenames = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
combined_weights = load_combined_weights(
    repo_id="meta-llama/Meta-Llama-3-8B",
    filenames=filenames,
    local_dir="Llama-3-8B"
)
```

There are a total of 4 files, requiring 4.6+4.7+4.6+1.1=15G of hard disk space.

Next comes the rather tedious assignment process, the core of which is to compare the parameters on both sides. The code is as follows:

```
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach()) if isinstance(right, torch.Tensor) else torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params):
    # Embedding
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"])

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]

        # map of attribute path (relative to block) -> param name
        attr_param_map = {
            f"att.W_query.weight": f"model.layers.{l}.self_attn.q_proj.weight",
            f"att.W_key.weight": f"model.layers.{l}.self_attn.k_proj.weight",
            f"att.W_value.weight": f"model.layers.{l}.self_attn.v_proj.weight",
            f"att.out_proj.weight": f"model.layers.{l}.self_attn.o_proj.weight",
            f"norm1.weight": f"model.layers.{l}.input_layernorm.weight",
            f"ff.fc1.weight": f"model.layers.{l}.mlp.gate_proj.weight",
            f"ff.fc2.weight": f"model.layers.{l}.mlp.up_proj.weight",
            f"ff.fc3.weight": f"model.layers.{l}.mlp.down_proj.weight",
            f"norm2.weight": f"model.layers.{l}.post_attention_layernorm.weight",
        }

        for attr_path, param_name in attr_param_map.items():
            obj = block
            *parents, attr = attr_path.split('.')
            for p in parents:
                obj = getattr(obj, p)
            old_tensor = getattr(obj, attr)
            setattr(obj, attr, assign(old_tensor, params[param_name]))

    # Final normalization
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"])

    # Output head with fallback (for weight tying)
    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"])
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"])
        print("Model uses weight tying.")
```

Load the weights into the model as follows:

```
device = torch.device("cpu")
load_weights_into_llama(model, LLAMA3_CONFIG_8B, combined_weights)
model.to(device)
del combined_weights
```

Finally, we also run the previous example to see if the model can complete the text, as follows:

```
from gpt2_v2 import generate_text_simple, text_to_tensor, tensor_to_text

torch.manual_seed(123)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_tensor("At the start of", tokenizer).to("cpu"),
    max_new_tokens=30,
    context_size=LLAMA3_CONFIG_8B["context_length"],
    top_k=1,
    temperature=0.
)

print("Output text:\n", tensor_to_text(token_ids, tokenizer))
```

The results are as follows:

```
Output text:
 At the start of the 2018 season, the club was in the 2nd division of the Dutch football league. The team is in the 1st place
```

This proves that our Llama3 model code is correct.

The base model of Llama3 8B we downloaded here can only perform text completion and cannot respond to instructions. If you're interested, you can similarly download the instruction-finetuned version, i.e., meta-llama/Meta-Llama-3-8B-Instruct, which will not be elaborated on here.




Code links involved in this article:[Llama2](https://github.com/weikuo0506/CreateYourOwnLLM/blob/main/Llama2_v1.py) [Llama3](https://github.com/weikuo0506/CreateYourOwnLLM/blob/main/Llama3.ipynb)