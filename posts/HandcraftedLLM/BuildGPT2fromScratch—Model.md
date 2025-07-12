# Build GPT2 from Scratch — Model

# GPT2 Model Architecture

We have already implemented Embedding and Multi-Head Attention, and now we will start implementing the complete gpt2. The overall architecture of gpt2 is as shown in[the following figure](https://medium.com/@vipul.koti333/from-theory-to-code-step-by-step-implementation-and-code-breakdown-of-gpt-2-model-7bde8d5cecda):

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/a601da06b6804a0cac90701d6a55a79f~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752401768&x-orig-sign=dOR7Q8MfxWHsf9Rhaf%2BJdtXo%2F6I%3D)

Where:

1) The most important Transformer Block is repeated and stacked 12 times.

2) The Transformer Block includes MHA, Layer Norm, and FeedFroward Neutral Network.

3) MHA contains 12 multi-heads.

As previously mentioned, we also know that GPT2 has a Vocab Size of 50257, a maximum seq length of 1024, and an embedding dimension of 768.

In summary, the model configuration for GPT2 is as follows:

```
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
```

# Dummy model

Now we will start by building the skeleton according to the above model architecture of GPT2, that is, first providing an overall dummy implementation, as follows:

```
import torch
import torch.nn as nn

class DummyGPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        token_emb = self.token_emb(token_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=token_ids.device))
        x = token_emb + pos_emb
        x = self.drop(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x
```

In the above code, we use DummyTransformerBlock as a dummy implementation, which contains empty operations. However, the overall framework is in place, and we only need to continue implementing the dummy part in the future.

Actually, the dummy code above can still run, but the result doesn't make much sense. However, we can still run it to see the structure of the final output.

The running example is as follows. First, we manually construct 2 batches of input token IDs with a context of 5:

```
import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

texts = ["Once upon a time there", "were four little Rabbits"]
batch = torch.stack([torch.tensor(tokenizer.encode(t)) for t in texts])
print(batch)
```

> tensor([[ 7454, 2402, 257, 640, 612],
>
> [22474, 1440, 1310, 22502, 896]])

We then input the above token id into the model:

```
torch.manual_seed(123)
model = DummyGPT2(GPT_CONFIG_124M)

logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
```

> Output shape: torch.Size([2, 5, 50257])
>
> tensor([[[-0.1453, -0.5939, 0.3767, ..., 0.4361, 0.3913, 1.1740],
>
> [ 0.2646, 0.5527, -1.0897, ..., 0.3165, 0.7068, 1.9168],
>
> [-0.2009, -0.7217, 0.7162, ..., 0.6297, 0.6221, -0.1177],
>
> [ 0.1959, 0.4116, 1.1859, ..., 2.2309, 0.2540, 0.7609],
>
> [-0.4772, -0.7713, 0.6711, ..., 0.9593, -1.1426, -1.0256]],
>
>
>
>
> [[-0.7387, 0.2473, -2.2699, ..., -0.9243, -1.1297, 0.1037],
>
> [-0.5791, 1.0997, -0.4741, ..., -0.7711, 0.9321, 1.0572],
>
> [ 0.7911, 1.0512, 0.4935, ..., 0.8441, -0.2399, -0.5090],
>
> [ 1.1721, 0.9144, -0.7984, ..., 1.6035, 0.5685, 1.0169],
>
> [-1.0692, -1.7418, 0.1271, ..., 0.1854, -0.5162, -0.7783]]],
>
> grad_fn=<UnsafeViewBackward0>)

The resulting vector has dimensions (batch, seq_len, vocab_size). After subsequent processing, the output represents the probability of each word in the 50257-word dictionary, so the last dimension of the output must be vocab_size.

# LayerNorm

In the above dummy model, we also reserved DummyLayerNorm, and now we start to implement it. In fact, the idea of LayerNorm is very simple, which is to normalize the sample features to a mean of 0 and a variance of 1.

Implementation is as follows:

```
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

It is very simple. According to statistical common sense, it is the input X minus the mean and then divided by the Standard Deviation; only a very small constant eps is introduced to prevent the divisor from being 0.

Run the following example:

```
torch.manual_seed(123)

batch_example = torch.randn(2, 5)
ln = LayerNorm(emb_dim=5)
out = ln(batch_example)
print(out)
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, unbiased=False, keepdim=True)
print("mean:\n", mean)
print("var:\n", var)
```

> tensor([[ 0.5528, 1.0693, -0.0223, 0.2656, -1.8654],
>
> [ 0.9087, -1.3767, -0.9564, 1.1304, 0.2940]], grad_fn=<AddBackward0>)
>
> mean:
>
> tensor([[-2.9802e-08],
>
> [ 0.0000e+00]], grad_fn=<MeanBackward1>)
>
> var:
>
> tensor([[1.0000],
>
> [1.0000]], grad_fn=<VarBackward0>)

It can be seen that after layernorm, the obtained mean is 0 and the variance is 1.

Actually, LayerNorm is very common in LLMs, and its implementation is also very simple. We can directly use the one provided by PyTorch later, as follows:

```
torch.manual_seed(123)

batch_example = torch.randn(2, 5)

layer = nn.LayerNorm(5)
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, unbiased=False, keepdim=True)
print("mean:\n", mean)
print("var:\n", var)
```

> tensor([[ 0.5528, 1.0693, -0.0223, 0.2656, -1.8654],
>
> [ 0.9087, -1.3767, -0.9564, 1.1304, 0.2940]],
>
> grad_fn=<NativeLayerNormBackward0>)
>
> mean:
>
> tensor([[-3.5763e-08],
>
> [ 2.3842e-08]], grad_fn=<MeanBackward1>)
>
> var:
>
> tensor([[1.0000],
>
> [1.0000]], grad_fn=<VarBackward0>)

The result is the same as above. Subsequently, you can directly use the nn.LayerNorm() class provided by PyTorch; it should be noted that this is not a function, but a layer/class in PyTorch, which is very convenient for stacking layers using nn.Sequential.

# Activations: Relu, GELU, SwiGLU

In neural networks, in addition to the aforementioned linear transformations (such as matrix projection, MHA, normalization, etc.), it is also necessary to introduce non-linear activation to enhance the network's expressive power.

can be intuitively understood:

1) All linear transformations, in essence, are matrix operations that preserve linear structure.

2) The purpose of linear transformation is to perform mapping between different spaces.

3) A linear transformation can be written in the form of matrix multiplication y = W ⋅ x + b, and it has the properties of closure under addition and closure under scaling.

4) Transformations such as rotation, scaling, projection, and shearing all belong to linear transformations. After linear superposition, they remain linear.

However, linearity cannot introduce mechanisms such as curvature, inflection points, and gating, and theoretically cannot fit all functions.

And to express any complex function, it only requires introducing a seemingly very simple non-linear activation function.

Behind this lies a relatively rigorous mathematical theory ([Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)), whose simplified idea is:

> **A** **feedforward neural network** **containing at least one layer of nonlinear activation functions can approximate any continuous function (on a compact interval) with arbitrarily small error, as long as the number of nerve cells in the** **hidden layer** **is large enough.**

That is to say, as long as we introduce a bit of non-linearity to the model on the basis of linear transformation, theoretically, as long as the model is deep enough, it can represent any complex function.

However, nonlinear activation functions are actually very simple; even the simplest, such as a piecewise linear function, can actually serve as a nonlinear activation function.

We can directly provide the code and explain based on the figure:

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SwiGLU_Simple(nn.Module):
    def forward(self, x):
        return x * F.silu(x)

gelu = nn.GELU()
relu = nn.ReLU()
swiglu = SwiGLU_Simple()

x = torch.linspace(-5, 5, 200)

y_gelu = gelu(x)
y_relu = relu(x)
y_swiglu = swiglu(x)

plt.figure(figsize=(12, 3))
for i, (y, label) in enumerate(zip(
    [y_relu, y_gelu, y_swiglu], ["ReLU", "GELU", "SwiGLU"]), 1):
    plt.subplot(1, 3, i)
    plt.plot(x.numpy(), y.detach().numpy(), label=label)
    plt.title(f"{label} activation")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
```

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b99ac31726a34020bb19a477658fb909~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752401768&x-orig-sign=uLfuEvCkqpLBk8b28dKzvuH8U%2FE%3D)

As can be seen, the simplest is ReLU, which is actually the piecewise function learned in junior high school. The more commonly used GELU and SwiGLU nowadays simply add some curvature to make it smoother and facilitate gradient calculation. The specific function definitions are quite boring, so I won't go into details here. We will use GELU later.

# FeedForward Network

In the model architecture of GPT, there is also a very important feedforward neural network (FFN) layer, as follows.

$$\text{FFN}(x) = \text{Linear}_2(\ \text{Activation}(\ \text{Linear}_1(x)\ )\ )$$

That is, for input x → linear transformation → nonlinear activation (ReLU/GELU) → linear transformation again → output.

Put simply, it involves inserting a non-linear layer between two linear layers to enhance the model's expressive power. Generally speaking, it is necessary to first increase the dimensionality in the first linear layer, perform non-linear activation, and then reduce the dimensionality back to the original dimension.

Let's directly look at the code example:

```
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), nn.GELU(), nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)

print("model structure: \n",FeedForward(GPT_CONFIG_124M))
```

> model structure:
>
> FeedForward(
>
> (layers): Sequential(
>
> (0): Linear(in_features=768, out_features=3072, bias=True)
>
> (1): GELU(approximate='none')
>
> (2): Linear(in_features=3072, out_features=768, bias=True)
>
> )
>
> )

As can be seen, in the above example, we first expand from 768 dimensions to 3072 dimensions by a factor of 4; then perform the GeLU activation operation; and finally reduce it back to 768 dimensions.

# ShortCut Connections

The optimization of neural networks relies on layer computation, and it can be said that layer is the engine of neural network training. When the number of layers in modern neural networks becomes increasingly astonishing, a relatively significant practical engineering challenge is layer explosion and layer vanishing; I believe everyone is familiar with this.

Backpropagation is the core algorithm for computing gradients in neural networks, and its concept is also very simple, still using the chain rule of differentiation to propagate layer by layer and compute gradients. Those interested can refer tokarpathy's introduction to[autograd](https://github.com/karpathy/micrograd), which will not be repeated here.

Automatic differentiation can also be said to be the soul of PyTorch. We can directly experience the layer changes during the calculation process through examples. The code is as follows:

```
import torch
import torch.nn as nn

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.GELU()) for i in range(len(layer_sizes)-1))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            out = layer(x)
            if self.use_shortcut and x.shape[-1] == out.shape[-1]:
                x = x + out
            else:
                x = out
        return x

def print_gradients(model,x):
    output = model(x)
    target = torch.zeros_like(output)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```

In the above code, we define multiple stacks of Linear+Gelu, and in forward, the simplest approach is to directly return layer(x).

We give the model a simulated input, generate a 1 x 4 tensor as the input, and see how the layer of each layer behaves, as follows:

```
layer_sizes = [4] * 6

x = torch.randn(1, 4)

torch.manual_seed(123)
model = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print(model)
print_gradients(model, x)
```

> ExampleDeepNeuralNetwork(
>
> (layers): ModuleList(
>
> (0-4): 5 x Sequential(
>
> (0): Linear(in_features=4, out_features=4, bias=True)
>
> (1): GELU(approximate='none')
>
> )
>
> )
>
> )
>
> layers.0.0.weight has gradient mean of 3.108993041678332e-05
>
> layers.1.0.weight has gradient mean of 7.357167487498373e-05
>
> layers.2.0.weight has gradient mean of 0.0006941530737094581
>
> layers.3.0.weight has gradient mean of 0.005131533369421959
>
> layers.4.0.weight has gradient mean of 0.014868268743157387

As can be seen, the example network has a total of 5 layers; we note that the gradient of layer0 is particularly small, 3e-5, and is already close to vanishing. We focus on the first few layers because, according to the principle of backpropagation, gradients are calculated starting from the last layer and moving forward, so the layers at the front of the model are most prone to problems.

Now we turn on use_shortcut=True and run it again, and the results are as follows:

> layers.0.0.weight has gradient mean of 0.06876879185438156
>
> layers.1.0.weight has gradient mean of 0.15942829847335815
>
> layers.2.0.weight has gradient mean of 0.12936799228191376
>
> layers.3.0.weight has gradient mean of 0.13758598268032074
>
> layers.4.0.weight has gradient mean of 0.173927441239357

It can be seen that the gradient of layers0 has magically increased to 6e-2; the only change is that the return value in the above forward has changed from layer(x) to x+layer(x).

This is the magic of shortcut, which seems to simply add the input x on top of the original output F(x), as follows:

$$ y = F(x) + x $$

The implied meaning is that at this time, F(x) represents the difference between the true output y and the input x, and is therefore also known as the Residual Network (ResNet).

Looking back now, it does seem very simple, but standing at that time, being the first to think of it and realizing the significance behind it was actually not easy.

Intuitively understood, a shortcut is equivalent to adding new pathways between different layers of very deep neural networks, allowing information to flow across layers; this "cross-layer channel" enables both information and layer to propagate in a leapfrog manner, thereby enhancing training stability and efficiency.

# Transformer Code

With the above implementations, we can directly combine them to present the actual code for the Transformer, as follows:

```
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_O = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        # shape (batch_size, seq_len, d_in)
        batch_size, seq_len, _ = x.size()

        # Split Q, K, V into multiple heads
        # (batch_size, seq_len, d_in) -> (batch_size, seq_len, d_out) ->
        # -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / (self.d_out ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], -torch.inf)

        # Compute softmax weights and apply dropout
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Compute output
        output = weights @ V  # (batch_size, num_heads, seq_len, head_dim)
        # Concatenate heads and project to output dimension
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # ->   (batch_size, seq_len, d_out)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # Should be helpful, but not strictly necessary.
        output = self.W_O(output)
        return output
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), nn.GELU(), nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg["emb_dim"])
        self.ln2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
```

Among them, the MHA code comes from the Attention module.

We can directly call it, check the output, as follows:

```
torch.manual_seed(123)

x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

> Input shape: torch.Size([2, 4, 768])
>
> Output shape: torch.Size([2, 4, 768])

It can be seen that after a series of operations by the Transformer, the dimension of the final output is exactly the same as that of the input.

# GPT-2 code

By stacking the Transformer 12 more times, we obtain the complete GPT-2 code.

```
import torch
import torch.nn as nn

class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        token_emb = self.token_emb(token_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=token_ids.device))
        x = token_emb + pos_emb
        x = self.drop(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

We can still simulate input to the model and check the results, as follows

```
torch.manual_seed(123)
model = GPT2Model(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("Output shape:", out.shape)
```

> Input batch:
>
> tensor([[ 7454, 2402, 257, 640, 612],
>
> [22474, 1440, 1310, 22502, 896]])
>
> Output shape: torch.Size([2, 5, 50257])

As can be seen, the last dimension of the output dimension is still equal to the vocab size. Here, what we output are the raw logits, which have not yet undergone the softmax transformation. The softmax will be explained in detail later, and it is simply equivalent to performing probability normalization.

# Model Overview

By now, the code for GPT-2 has been fully constructed. Let's take a look at the details of the model. Here we introduce the torchinfo package, and by simply using summary, we can see the model architecture and parameters, as follows:

```
from torchinfo import summary

summary(model)
```

> ================================================================= Layer (type:depth-idx) Param # ================================================================= GPT2Model -- ├─Embedding: 1-1 38,597,376 ├─Embedding: 1-2 786,432 ├─Dropout: 1-3 -- ├─Sequential: 1-4 -- │ └─TransformerBlock: 2-1 -- │ │ └─MultiHeadAttention: 3-1 2,360,064 │ │ └─FeedForward: 3-2 4,722,432 │ │ └─LayerNorm: 3-3 1,536 │ │ └─LayerNorm: 3-4 1,536 │ │ └─Dropout: 3-5 -- │ └─TransformerBlock: 2-2 -- │ │ └─MultiHeadAttention: 3-6 2,360,064 │ │ └─FeedForward: 3-7 4,722,432 │ │ └─LayerNorm: 3-8 1,536 │ │ └─LayerNorm: 3-9 1,536 │ │ └─Dropout: 3-10 -- ........(Omit TransformerBlock2-3 to 2-11 to save space) │ └─TransformerBlock: 2-12 -- │ │ └─MultiHeadAttention: 3-56 2,360,064 │ │ └─FeedForward: 3-57 4,722,432 │ │ └─LayerNorm: 3-58 1,536 │ │ └─LayerNorm: 3-59 1,536 │ │ └─Dropout: 3-60 -- ├─LayerNorm: 1-5 1,536 ├─Linear: 1-6 38,597,376 ================================================================= Total params: 163,009,536 Trainable params: 163,009,536 Non-trainable params: 0 =================================================================

It can be seen that the levels of the model are:

1) Overall Structure: Token Embedding -> Position Embedding -> Dropout -> Transformer * 12 -> LayerNorm -> Linear;

2) Transformer Structure: LayerNorm -> MHA -> Dropout -> LayerNorm -> FeedForward -> Dropout; the order in the summary here is not very accurate and should be based on the code. Among them, Dropout is optional.

And it can also be seen that the total number of parameters of the model is 163M.

Actually, we can also conveniently calculate model parameters using the built-in function of PyTorch, as follows:

```
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

print("Token embedding layer shape:", model.token_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

total_size_mb = total_params * 4 / (1024 ** 2)
print(f"Total size of the model: {total_size_mb:.2f} MB")
```

> Total number of parameters: 163,009,536
>
> Token embedding layer shape: torch.Size([50257, 768])
>
> Output layer shape: torch.Size([50257, 768])
>
> Number of trainable parameters considering weight tying: 124,412,160
>
> Total size of the model: 621.83 MB

The total parameters are the same as above, among which the parameters other than out_head are 124M; usually out_head is removed because in GPT2, shared parameters are used, and the weight tensor of out_head used in the last layer is actually the tensor of token_embedding directly; so this part of the duplicate parameters can be deduplicated. Therefore, the total trainable parameters of the model are 124M; the total model size is 621M; but in fact, the core code is only 100 lines.

# Generate Text

Although the above GPT2 code has not been trained, its structure is actually complete, and we can directly use it to test text generation. Of course, we expect it to generate garbled text that fails to convey the intended meaning. However, the test output can help us check for code issues, as follows:

```
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # Get logits from model
        with torch.no_grad():
            logits = model(idx_cond)

        # Take logits for the last time step
        # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
```

In the above code, we first use torch.no_grad() to set the torch mode to the evaluation phase instead of the training phase, avoiding the additional overhead caused by automatic gradient calculation.

Moreover, only one word is generated each time, so we only extract the 50257 dimensions corresponding to the last token from the generated (batch, n_tokens, vocab_size) each time, which becomes (batch, vocab_size); we perform a softmax transformation on the logits, and then pick out the id corresponding to the value with the highest probability; this id is actually the token_id corresponding to the generated word; in fact, the softmax here is redundant, even if we only look at the original logits, picking out the number corresponding to the maximum value from the 50257 dimensions is also the tokenId corresponding to the generated word. Because softmax only performs probability normalization, in fact we don't care about the magnitude of the value, we just want to find the id with the maximum value.

Let's implement the conversion from the token_id tensor to text again, which is the tokenizer encode and decode mentioned in the Embedding section, as follows:

```
def text_to_tensor(text,tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)

def tensor_to_text(tensor,tokenizer):
    return tokenizer.decode(tensor.squeeze(0).tolist())
```

Let's directly run the example and see what the model generates:

```
start_context = "Once upon a time there"

encoded_tensor = text_to_tensor(start_context,tokenizer)
print("encoded_tensor.shape:", encoded_tensor.shape)
print("encoded_tensor:", encoded_tensor)

model.eval() # disable dropout

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
decoded_text = tensor_to_text(out,tokenizer)
print(decoded_text)
```

We input a text with only 1 batch and a length of 5, first convert it to token ids; then change the model to eval mode and input it into the model. We set the maximum number of new tokens to be generated to 6.

The results are as follows:

> encoded_tensor.shape: torch.Size([1, 5])
>
> encoded_tensor: tensor([[7454, 2402, 257, 640, 612]])
>
> Output: tensor([[ 7454, 2402, 257, 640, 612, 41117, 4683, 36413, 33205, 35780,
>
> 22580]])
>
> Once upon a time there discriminated existing REALLY JehovahQUEST valve

As can be seen, the model did generate 6 new tokens, but it seems to be babbling. After all, this is an untrained model, and all the weights are still at their initial values. Here, we can still see that although everyone knows that Model Training is extremely costly, in fact, the cost of model generation in the eval inference mode is still relatively low.

So far, we have completed the full construction of the GPT2 model. Although it currently lacks intelligence, it already has the necessary connections and pathways of an intelligent agent. This is equivalent to a baby whose neural pathways are intact; subsequent Model Training to establish neural connections will enable it to acquire intelligence.