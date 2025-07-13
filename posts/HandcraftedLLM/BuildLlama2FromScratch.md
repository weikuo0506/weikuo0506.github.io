LLaMA (**Large Language Model Meta AI**) is a series of open-source large language models developed by Meta (formerly Facebook). Among them, [Llama2](https://www.llama.com/llama2/) was released in 2023, with sizes of 7B, 13B, and 70B. For a detailed introduction, please refer to its paper [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288) and other materials, which will not be repeated here. Instead, this article focuses on its core technology and code implementation. Based on the technology of the [Handmade Large Model - GPT2 Series] blog and by making changes to the GPT2 source code, this article will write the source code for Llama2 and load its publicly available weights.

Code link for this article: [Llama2](https://github.com/weikuo0506/CreateYourOwnLLM/blob/main/Llama2.ipynb), original reference code link: [rasbt](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb).

# RMSNorm Layer

In GPT2, the standard LayerNorm is used for regularization, which means "subtract the mean and divide by Standard Deviation", while in Llama2, [RMSNorm ](https://arxiv.org/pdf/1910.07467)is used, which means "no longer subtract the mean, directly divide by the root mean square". This is equivalent to only scaling without centralization. The core reason for doing this is to reduce computational complexity, omitting the calculation of mean and variance, and only calculating the sum of squares. In practice, it has been found that in larger models, even without centralization, scaling still retains directional information, which is more stable and effective in actual training.

The comparison code is as follows:

```
import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # Learnable scale (gamma) and shift (beta) parameters
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Compute mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize input
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # Apply scale and shift
        return self.scale * norm_x + self.shift

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        # Learnable scaling parameter (gamma)
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        # Compute root mean square (RMS)
        means = x.pow(2).mean(dim=-1, keepdim=True)
        # Normalize input by RMS
        x_normed = x * torch.rsqrt(means + self.eps)
        # Apply scaling and restore original dtype
        return (x_normed * self.weight).to(dtype=x.dtype)
```

We construct examples and execute Norm respectively:

```
# Set random seed for reproducibility
torch.manual_seed(123)

# Create input tensor with uniform distribution shifted away from zero
example = torch.rand(2, 3, 10) * 4 + 3  # values roughly in [3,7]

print("Input tensor (example):")
print("Raw mean:", example.mean().item())
print("Raw std :", example.std().item())
print("Raw RMS :", torch.sqrt(example.pow(2).mean(dim=-1).mean()).item())

# Instantiate normalization layers
layer_norm = LayerNorm(emb_dim=example.shape[-1])
rms_norm = RMSNorm(emb_dim=example.shape[-1])
rms_norm_pytorch = torch.nn.RMSNorm(example.shape[-1], eps=1e-5)  # PyTorch built-in

# Apply normalization
out_layer = layer_norm(example)
out_rms = rms_norm(example)
out_rms_pt = rms_norm_pytorch(example)

# Print normalized outputs statistics
print("After LayerNorm:")
print("Mean:", out_layer.mean().item())
print("Std :", out_layer.std().item())
print("RMS :", torch.sqrt(out_layer.pow(2).mean(dim=-1).mean()).item())

print("After RMSNorm (custom):")
print("Mean:", out_rms.mean().item())
print("Std :", out_rms.std().item())
print("RMS :", torch.sqrt(out_rms.pow(2).mean(dim=-1).mean()).item())

print("After RMSNorm (PyTorch built-in):")
print("Mean:", out_rms_pt.mean().item())
print("Std :", out_rms_pt.std().item())
print("RMS :", torch.sqrt(out_rms_pt.pow(2).mean(dim=-1).mean()).item())
```

The results are as follows:

```
Input tensor (example):
Raw mean: 5.003686428070068
Raw std : 1.1390745639801025
Raw RMS : 5.129594802856445
After LayerNorm:
Mean: -1.033147185580674e-07
Std : 1.0084344148635864
RMS : 0.9999955296516418
After RMSNorm (custom):
Mean: 0.9775436520576477
Std : 0.2125103920698166
RMS : 0.9999997615814209
After RMSNorm (PyTorch built-in):
Mean: 0.9775436520576477
Std : 0.2125103920698166
RMS : 0.9999997615814209
```

It can be seen that after standard LayerNorm, the mean will be close to 0, and both the standard deviation (std) and root mean square (rms) will be close to 1; however, after RMS norm, only rms will be close to 1.

Additionally, the RMSNorm implemented above is consistent with the result of the built-in one in PyTorch, and the built-in torch.nn.RMSNorm module in PyTorch can be directly used in the future.

# SiLU activation

Compared to GPT2, which uses the GELU activation function, Llama2 switches to using[SiLU](https://arxiv.org/abs/1702.03118)(Sigmoid Linear Unit), also known as the Swish function.

Its formula is:

$$\text{silu}(x) = x \cdot \sigma(x), \quad \text{where} \quad \sigma(x) \text{ is the logistic sigmoid.}$$

The code implementation is extremely simple:

```
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
```

More simply, you can also directly use the torch.nn.Silu() module provided by PyTorch.

We can draw a diagram to compare GELU and SiLU:

```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

gelu = nn.GELU()
silu = nn.SiLU()

x = torch.linspace(-5, 5, 200)

y_gelu = gelu(x)
y_silu = silu(x)

plt.figure(figsize=(12, 3))
for i, (y, label) in enumerate(zip(
    [y_gelu, y_silu], ["GELU", "SiLU"]), 1):
    plt.subplot(1, 3, i)
    plt.plot(x.numpy(), y.detach().numpy(), label=label)
    plt.title(f"{label} activation")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
```

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/cdadbb52f74b4a87a4bde5efdaa7408b~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752481631&x-orig-sign=A9zosMHNfDEk9Sc16fQBw4qr01g%3D)

It can be seen that the two are very close. However, SiLU is computationally simpler and faster because it only involves sigmoid and multiplication.

# SwiGLU in FeedForward

GPT2 directly uses the GELU activation function in the Feedforward module, while Llama2 uses a variant of the SiLU-based gated activation function (Gated Linear Unit) called SwiGLU, whose formula is as follows:

$$\text{SwiGLU}(x) = \text{SiLU}(\text{Linear}_1(x)) * (\text{Linear}_2(x))$$

That is to say, SwiGLU requires two input linear layers to implement the gating structure.

The complete comparison code for the FeedForward module is as follows:

```
class FeedForwardInGPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
```

As can be seen, in Llama2, nn.Sequential can no longer be used for linear stacking; instead, two linear input layers, fc1 and fc2, are required. After performing the SwiGLU gating operation, the result is then fed into fc3. Additionally, bias has been removed from all linear layers in Llama2, i.e., bias=False. This is because removing bias can reduce computational load, and in practice, it has been found that this does not affect the training results.

# RoPE

Compared to traditional absolute position encoding, Llama2 uses Rotary Position Encoding[RoPE](https://arxiv.org/abs/2104.09864) to capture both absolute and relative position information simultaneously. RoPE achieves "relative position sensitivity" by converting position encoding into angular rotation and applying it to Q/K. RoPE's design is very ingenious, inspired by complex rotation, and the detailed design ideas can also be found in the author's[ blog](https://kexue.fm/archives/8130).

Here we briefly explain the underlying mathematics, and you can also refer to[wiki](https://en.wikipedia.org/wiki/Rotation_(mathematics)#Rotation_in_the_plane).

1.  Rotate the original vector by an angle θ:

Suppose the original vector is represented as a complex number: $$ z = a + b i $$

Multiply it by a complex number of unit modulus: $$e^{i\theta} = \cos\theta + i\sin\theta$$

We obtain: $$z' = z \cdot e^{i\theta} = (a\cos\theta - b\sin\theta) + i(a\sin\theta + b\cos\theta)$$

This process can be expressed as matrix multiplication:

$$ z' = \begin{bmatrix} a' \\ b' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} $$

That is to say, multiplying the original vector by a complex number with unit modulus is equivalent to rotating the original two-dimensional vector [a, b] clockwise by an angle θ.

2.  The distance between vectors after rotation depends on the relative position m-n:

Assume that the tokens at positions m and n, after rotation by an angle θ, have vectors as follows:

$$z_m = z \cdot e^{i\theta_m}, \quad z_n = z \cdot e^{i\theta_n}$$

The Euclidean distance (2-norm) between them is:

$$\text{dist}(z_m, z_n) = |z \cdot e^{i\theta_m} - z \cdot e^{i\theta_n}| = |z| \cdot |e^{i\theta_m} - e^{i\theta_n}| = |z| \cdot |e^{i(\theta_m - \theta_n)}=|z| \cdot |e^{i(m - n)\omega} - 1|$$

That is to say, the distance between vectors after rotation depends on their relative positions.

The key idea of RoPE is to rotate the vector at each position by a specific angle, rather than directly adding positional encoding. Instead of using `x + pos_embedding`, it rotates **Query** **and Key vectors** to implicitly incorporate positional information. RoPE encodes positional information into the vector by treating each pair of dimensions in the token vector as a complex number and performing position-based rotation on it.

The complete code for RoPE is as follows:

```
import torch

def precompute_rope_params(seq_len, head_dim):
    """
Precompute sin and cos tensors for RoPE.

Args:
seq_len: sequence length
head_dim: embedding dimension (must be even)

Returns:
sin, cos: tensors of shape (seq_len, dim//2)
"""
half_dim = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(half_dim).float() / half_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, half_dim)
    return torch.sin(angles), torch.cos(angles)


def rotary_pos_emb(x, sin, cos):
    """
Apply Rotary Positional Embedding on input tensor x using precomputed sin and cos.

Args:
x: tensor of shape (batch, seq_len, dim)
sin: precomputed sin tensor of shape (seq_len, dim//2)
cos: precomputed cos tensor of shape (seq_len, dim//2)

Returns:
tensor same shape as x with RoPE applied.
"""
print("Rotary Positional Embedding",x.shape)
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch, num_heads, seq_len, head_dim = x.shape

    # x: (batch, seq_len, dim) -> (batch, seq_len, half_dim, 2)
    x_ = x.view(batch, num_heads, seq_len, head_dim // 2, 2)
    print("shape of x_", x_.shape)
    print("shape of cos", cos.shape)

    # ➤ Crop sin/cos to match actual seq_len
    sin = sin[:seq_len, :]
    cos = cos[:seq_len, :]

    x_rotated = torch.zeros_like(x_)
    x_rotated[..., 0] = x_[..., 0] * cos - x_[..., 1] * sin
    x_rotated[..., 1] = x_[..., 0] * sin + x_[..., 1] * cos

    return x_rotated.view_as(x)
```

The key steps to explain the RoPE code are as follows:

1.  Calculate the rotation frequency (inv_freq)

```
    half_dim = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
```

**half_dim**: Divide the embedding dimension into two halves (since each pair of dimensions forms the real and imaginary parts of a complex number).

**inv_freq**: Calculate the inverse frequency of rotation, with the formula derived from the classic positional encoding concept of Transformer:$$ω_j = \frac{1}{10000^{2j / d}}$$

2.  Generate position index

```
positions = torch.arange(seq_len, dtype=torch.float32)
```

Generate a sequence of `[0, 1, 2,..., seq_len-1]` representing the position index of each token.

3.  Calculate the rotation angle matrix

```
angles = torch.einsum("i,j->ij", positions, inv_freq)
```

Here, the Einstein summation convention for matrices is used to calculate the outer product, which is equivalent to the following matrix multiplication:

```
angles = positions[:, None] * inv_freq[None, :]
```

Each position is multiplied by each frequency, resulting in a matrix of shape `(seq_len, half_dim/2)`, representing the rotation angle of each position in each dimension.

4.  Calculate sin and cos

```
sin = torch.sin(angles)
cos = torch.cos(angles)
```

Take the sine and cosine of the angle matrix respectively to generate the corresponding elements of the rotation matrix.

5.  Adjust the dimensions of the input tensor

```
x_ = x.view(*x.shape[:-1], half_dim, 2)
```

Reshape the input `(batch, seq_len, dim)` into `(batch, seq_len, half_dim, 2)`, that is, treat every two dimensions as a group, resembling the real and imaginary parts of a complex number.

6.  Rotation Calculation

```
x_rotated = torch.zeros_like(x_)
x_rotated[..., 0] = x_[..., 0] * cos - x_[..., 1] * sin
x_rotated[..., 1] = x_[..., 0] * sin + x_[..., 1] * cos
```

Where:

`x_[..., 0]` represents the first component of each pair of dimensions, similar to the x component (real part) of a two-dimensional vector.

`x_[..., 1]` represents the second component of each pair of dimensions, similar to the y component (imaginary part) of a two-dimensional vector.

These two lines of code use the two-dimensional rotation matrix formula to rotate each two-dimensional vector (composed of two-dimensional embeddings):

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

However, `cos` and `sin` have shapes of `(seq_len, half_dim/2)`, but due to the broadcast rule, they will automatically match the dimensions of `(batch, seq_len, half_dim, 2)`.

7.  Restore shape output

```
return x_rotated.view_as(x)
```

Reshape the rotated tensor back to the original `(batch, seq_len, dim)` shape.

Usage examples are as follows:

```
# Example usage
batch_size, seq_len, dim = 2, 16, 64
x = torch.randn(batch_size, seq_len, dim)
# Step 1: Precompute sin and cos for RoPE
sin, cos = precompute_rope_params(seq_len, dim)
# Step 2: Apply rotary positional embedding with precomputed sin and cos
x_rope = rotary_pos_emb(x, sin, cos)

print(x_rope.shape)  # Should be (2, 16, 64)
```

# update MHA with RoPE

The absolute position encoding used by GPT2 is applied to the inputs, while the rotary position encoding of Llama2 is applied to the Query and Key in the attention mechanism. Therefore, we update the code of MHA as follows:

```
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Use nn.Linear with shared kwargs to reduce repetition
        linear_kwargs = dict(bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, **linear_kwargs)
        self.W_key = nn.Linear(d_in, d_out, **linear_kwargs)
        self.W_value = nn.Linear(d_in, d_out, **linear_kwargs)
        self.out_proj = nn.Linear(d_out, d_out, **linear_kwargs)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

        cos, sin = precompute_rope_params(seq_len=context_length, head_dim=self.head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Project inputs
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        keys = rotary_pos_emb(keys, self.cos, self.sin)
        queries = rotary_pos_emb(queries, self.cos, self.sin)

        # Attention scores with causal mask
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], float('-inf'))

        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)

        context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
```

Construct example to compute mha as follows:

```
batch_size = 2
context_len = 128
max_context_len = 1280
embed_dim = 64
num_heads = 4

example_batch = torch.randn(batch_size, context_len, embed_dim)

mha = MultiHeadAttention(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=max_context_len,
    num_heads=num_heads
)

output = mha(example_batch)
print(output.shape)  # Expected: (batch_size, context_len, embed_dim)
```

```
Rotary Positional Embedding torch.Size([2, 4, 128, 16])
shape of x_ torch.Size([2, 4, 128, 8, 2])
shape of cos torch.Size([1280, 8])
Rotary Positional Embedding torch.Size([2, 4, 128, 16])
shape of x_ torch.Size([2, 4, 128, 8, 2])
shape of cos torch.Size([1280, 8])
torch.Size([2, 128, 64])
```

# Update TransformerBlock

By now, we have completed the core code in Llama2. Next, we will integrate the above code to update the TransformerBlock. The core changes include:

1) Replace LayerNorm with RMSNorm to simplify the computation.

2) Remove Dropout, because when the model parameters are large enough, dropout is not necessary.

3) Remove the bias setting to reduce computational load and improve numerical stability.

4) Added dtype setting to support more efficient low-precision training and inference; for example, using bfloat16 to save GPU memory and accelerate training.

The complete code is as follows:

```
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)

        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
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

# Update the Model

Recalling GPT2, the models are all multiple stacked repetitions of TransformerBlock. In Llama2, we need to remove pos_emb and replace it with RoPE, change to RMSNorm, and set dtype. The code is as follows:

```
class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

# Initialize Model

The model parameters of Llama2 are as follows:

```
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  
}
```

Load the model as follows:

```
model = Llama2Model(LLAMA2_CONFIG_7B)
```

Calculate total parameters:

```
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```

Total number of parameters: 6,738,415,616

The visible parameter is 6.7B, usually abbreviated as 7B.

# Load Tokenizer

GPT2 uses the tiktoken Tokenizer, while Llama2 uses Google's [SentencePiece ](https://github.com/google/sentencepiece). Meta has published the trained public weights and Tokenizer vocabulary on [HuggingFace ](https://huggingface.co/meta-llama/Llama-2-7b). We can download and load them from HF, but since Llama2 is not fully open source (only for personal and non-commercial use), we need to first apply for permission and log in to HF.

Download the tokenizer as follows:

```
from huggingface_hub import login

login(token="your hf oken")
```

```
from huggingface_hub import hf_hub_download

tokenizer_file = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="tokenizer.model",
    local_dir="Llama-2-7b"
)
```

Define the tokenizer as follows:

```
import sentencepiece as spm

class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)

    def decode(self, ids):
        return self.tokenizer.decode_pieces(ids)


tokenizer = LlamaTokenizer(tokenizer_file)
```

Try running the generated text, as follows:

```
from gpt2_v2 import generate_text_simple, text_to_tensor, tensor_to_text

torch.manual_seed(123)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_tensor("At the start of", tokenizer).to("cpu"),
    max_new_tokens=30,
    context_size=LLAMA2_CONFIG_7B["context_length"],
    top_k=1,
    temperature=0.
)

print("Output text:\n", tensor_to_text(token_ids, tokenizer))
```

```
Output text:
 At the start ofзей Warjarewnę обще Opera з went eeuwể Other collaborationlauf’Powerремħ’Powerремħ’ep kur extremely____dataset Multi vida curv
```

It can be seen that the tokenizer has been successfully loaded, but the generated output is nearly garbled, because the model has not been trained and only has randomly initialized weights.

# Load pretrained Weights

Similarly, we can download and load the pre-trained and publicly available model weights from Meta AI, as follows:

```
weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b",
   filename="consolidated.00.pth",
   local_dir="Llama-2-7b"
)
```

```
weights = torch.load(weights_file, weights_only=True)
```

The process of loading weight parameters is essentially parameter copying, as follows:

```
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach()) if isinstance(right, torch.Tensor) else torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]

        # map of attribute path (relative to block) -> param name
        attr_param_map = {
            f"att.W_query.weight": f"layers.{l}.attention.wq.weight",
            f"att.W_key.weight": f"layers.{l}.attention.wk.weight",
            f"att.W_value.weight": f"layers.{l}.attention.wv.weight",
            f"att.out_proj.weight": f"layers.{l}.attention.wo.weight",
            f"norm1.weight": f"layers.{l}.attention_norm.weight",
            f"ff.fc1.weight": f"layers.{l}.feed_forward.w1.weight",
            f"ff.fc2.weight": f"layers.{l}.feed_forward.w3.weight",  # swapped order
            f"ff.fc3.weight": f"layers.{l}.feed_forward.w2.weight",
            f"norm2.weight": f"layers.{l}.ffn_norm.weight",
        }

        for attr_path, param_name in attr_param_map.items():
            obj = block
            *parents, attr = attr_path.split('.')
            for p in parents:
                obj = getattr(obj, p)
            old_tensor = getattr(obj, attr)
            setattr(obj, attr, assign(old_tensor, params[param_name]))

    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])
```

```
device = torch.device("cpu")
load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
model.to(device);
```

Running the above example generation statement again yields:

```
Output text:
 At the start of the 20th century, the city was a major industrial center, with a large number of factories and mills. The city was also
```

As can be seen, the generated results are semantically consistent, which also indicates that our model code is correct and the weights have been successfully loaded.

# try instruction-finetuned model

In the above example, we loaded the 7B base model, which has only been pre-trained and not fine-tuned. Therefore, it can only complete text and cannot respond to instructions. Next, similarly, we download and load the instruction-FT weights.

Download and load the weights of the chat model as follows:

```
weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b-chat",
   filename="consolidated.00.pth",
   local_dir="Llama-2-7b-chat"
)
```

```
model = Llama2Model(LLAMA2_CONFIG_7B)
load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
model.to(device)
```

For this practice run, the example is as follows:

```
from gpt2_v2 import generate_text_simple, text_to_tensor, tensor_to_text

torch.manual_seed(123)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_tensor("What do llamas eat?", tokenizer).to(device),
    max_new_tokens=30,
    context_size=LLAMA2_CONFIG_7B["context_length"],
    top_k=1,
    temperature=0.
)

print("Output text:\n", tensor_to_text(token_ids, tokenizer))
```

```
Output text:
 What do llamas eat?
Llamas are herbivores, which means they eat plants. They eat grass, hay, and other plants.
What do llam
```

By now, we have completed the code implementation of Llama2, as well as downloaded and loaded the pre-trained weights.