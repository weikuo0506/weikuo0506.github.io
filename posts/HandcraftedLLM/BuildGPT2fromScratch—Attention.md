# Understand Attention in an Intuitive Way

Previously, we discussed token Embedding and position Embedding. However, embedding is essentially still a vector related to the token itself. Once training is completed, embedding becomes fixed weights. We need to introduce a new mechanism to focus on the dependencies between tokens, that is, the context in which the token resides.

The following example:

> “He sat by the **bank** of the river.”
>
> "The cat that the dog chased was black."

Whether "bank" refers to a bank or a riverbank depends on the nearby "river"; although "black" is closer to "dog", we still know that it actually refers to "cat".

Focusing on the impact of different tokens in a sentence with emphasis, that is, paying attention to semantic relevance, is precisely the starting point of the Attention mechanism. In other words, the essence of the Attention mechanism is to enable the model to independently judge which tokens are more relevant at each step and construct the context accordingly.

In [the paper on the Attention ](https://arxiv.org/abs/1706.03762)mechanism, the three tensors Query/Key/Value were innovatively introduced:

> An attention function can be described as mapping a query and a set of key-value pairs to an output,
>
> where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
>
> of the values, where the weight assigned to each value is computed by a compatibility function of the
>
> query with the corresponding key.

Where: Query represents a question or concern, Key represents the index of information, and Value represents the specific value of information.

Intuitively, Attention simulates the process of "query - retrieval - extraction", and the following simple example helps to understand:

1) Suppose you go to the library to look for books (query is the topic you want to know),

2) The library has many shelf labels (keys) and books (values),

3) First, check how relevant (compatibility) each bookshelf label is to the topic you want,

4) Then you decide how many books to take from which bookshelves based on relevance (weighted as compatibility),

5) Combine the content of the books you've obtained (weighted sum), and that will be your final answer (output).

# Scaled Dot-Product Attention

## Attention Definition

The strict definition of Attention is as follows:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

Here, Q, K, and V represent the matrices corresponding to query, key, and value respectively; while dk represents the dimension of the matrix, used for scaling.

The definition is concise, and in fact, the calculation process is also particularly clear and simple.

Take the following example with 5 tokens and an embedding dimension of 4:

```
import torch
import torch.nn as nn

torch.manual_seed(123)

tokens = ["Once", "upon", "a", "time", "there"]
token_to_idx = {token: idx for idx, token in enumerate(tokens)}
embedding_dim = 4

embedding_layer = nn.Embedding(num_embeddings=len(tokens), embedding_dim=embedding_dim)

input_indices = torch.tensor([token_to_idx[token] for token in tokens])  # [0,1,2,3,4]
X = embedding_layer(input_indices)
print("shape of input X:", X.shape)
print(X)
```

> shape of input X: torch.Size([5, 4])
>
> tensor([[ 0.3374, -0.1778, -0.3035, -0.5880],
>
> [ 1.5810, 1.3010, 1.2753, -0.2010],
>
> [-0.1606, -0.4015, 0.6957, -1.8061],
>
> [-1.1589, 0.3255, -0.6315, -2.8400],
>
> [-0.7849, -1.4096, -0.4076, 0.7953]], grad_fn=<EmbeddingBackward0>)

Obtain a 5*4 two-dimensional matrix, as follows:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c57cb22bf95f4d14b30f05517ee77ddf~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752401940&x-orig-sign=I1lJWF%2BAsZeClMDhHJCP1U5MgCk%3D)

## Q K V matrix

Based on this, the method of creating the Q/K/V matrices is extremely simple. It only requires specifying the dimensions, randomly initializing to obtain the initial matrix, and then using this matrix to perform a linear mapping on the input X, as follows:

```
torch.manual_seed(123)

W_Q = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim), requires_grad=False)
W_K = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim), requires_grad=False)
W_V = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim), requires_grad=False)

print("shape of W_Q:", W_Q.shape)
print("W_Q:", W_Q)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("shape of Q:", Q.shape)
print("Q:", Q)
```

> shape of W_Q: torch.Size([4, 4])
>
> W_Q: Parameter containing:
>
> tensor([[0.2961, 0.5166, 0.2517, 0.6886],
>
> [0.0740, 0.8665, 0.1366, 0.1025],
>
> [0.1841, 0.7264, 0.3153, 0.6871],
>
> [0.0756, 0.1966, 0.3164, 0.4017]])
>
> shape of Q: torch.Size([5, 4])
>
> Q: tensor([[-0.0136, -0.3159, -0.2211, -0.2307],
>
> [ 0.7839, 2.8310, 0.9140, 2.0175],
>
> [-0.0858, -0.2806, -0.4474, -0.3992],
>
> [-0.6501, -1.3338, -1.3449, -2.3394],
>
> [-0.3515, -1.7666, -0.2669, -0.6454]], grad_fn=<MmBackward0>)

Please note that the dimension of X is [5, 4], the dimension of W_Q obtained by random initialization is [4, 4], and according to matrix multiplication, the dimension of Q obtained is [5, 4].

Here, matrix multiplication is used (@ is equivalent to torch.matmul), which is equivalent to performing a linear projection on the original input X.

Notably, all three of W_Q, W_K, and W_V are trainable parameters. This means that their initial values are not important; what matters is the constructed space Degree of Freedom and the information pathway.

## Similarity as Scores

According to the above formula, $$\text{scores} = Q K^T$$, we need to calculate the dot product between Q and K to compute the correlation or similarity between them.

```
scores = Q @ K.T
print("shape of scores:", scores.shape)
print("scores:", scores)
```

> shape of scores: torch.Size([5, 5])
>
> scores: tensor([[ 0.3101, -2.0474, 0.7024, 1.8280, 1.0647],
>
> [ -2.5714, 17.4476, -5.5017, -14.6920, -9.3044],
>
> [ 0.6084, -2.9632, 1.4480, 3.1775, 1.4642],
>
> [ 2.8736, -14.6337, 6.4597, 14.7155, 7.4156],
>
> [ 0.9222, -8.1955, 1.8808, 5.9959, 4.5150]],
>
> grad_fn=<MmBackward0>)

In the above example, the dimensions of Q and K are [5,4]. Transposing K results in dimensions [4,5]. Taking the dot product of the two yields a matrix of [5,5].

*Note: What is actually done here is* *batch* ***dot product* *, i.e.,* *matrix* *multiplication is used.*

## Scaled Scores

$$ \text{scaled\_scores} = \frac{Q K^T}{\sqrt{d_k}} $$

```
import math

attention_scores = scores / math.sqrt(embedding_dim)
print(attention_scores)
```

> tensor([[ 0.1551, -1.0237, 0.3512, 0.9140, 0.5323],
>
> [-1.2857, 8.7238, -2.7508, -7.3460, -4.6522],
>
> [ 0.3042, -1.4816, 0.7240, 1.5888, 0.7321],
>
> [ 1.4368, -7.3169, 3.2298, 7.3577, 3.7078],
>
> [ 0.4611, -4.0977, 0.9404, 2.9979, 2.2575]], grad_fn=<DivBackward0>)

So why do we perform scaling, and why do we choose the above values for scaling? Scaling is mainly to compress the score, avoid an overly extreme distribution in the subsequent softmax output, and make layer computation smoother; choosing the square root of dk should have its mathematical and statistical significance. However, personally, it still seems to be an empirical and compromise solution, and dividing by other values is also reasonable, so there's no need to pay excessive attention, as it's essentially just a data regularization method.

At this point, we have completed the calculation of Scaled Attention scores.

## Compute Attention Weights via Softmax

To convert attention scores into usable weights, further normalization is required, namely through the softmax operation:

$$\text{attention\_weights} = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right)$$

Draw a picture to take a look at the softmax function, which is extremely simple, as follows:

```
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
scores = torch.stack([x, torch.zeros_like(x)], dim=1)
softmax_vals = torch.softmax(scores, dim=1)

plt.plot(x.numpy(), softmax_vals[:,0].numpy())
plt.show()
```

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b0fa71d099a84ec7bd32455792b9818c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752401940&x-orig-sign=XJMGpm5CwYGomyv7dGqFqs6y%2FSE%3D)

It can be seen that the softmax function compresses all inputs into the range (0, 1), making them look more like probability values.

*Note: Softmax is essentially a data normalization method and can also be replaced with other similar functions.*

We directly use the softmax function provided by PyTorch to calculate as follows:

```
attention_weights = torch.softmax(attention_scores, dim=-1)
print("shape of attention_weights:", attention_weights.shape)
print(attention_weights)
```

> shape of attention_weights: torch.Size([5, 5])
>
> tensor([[1.6344e-01, 5.0283e-02, 1.9885e-01, 3.4910e-01, 2.3833e-01],
>
> [4.4966e-05, 9.9994e-01, 1.0389e-05, 1.0494e-07, 1.5519e-06],
>
> [1.2761e-01, 2.1395e-02, 1.9418e-01, 4.6106e-01, 1.9576e-01],
>
> [2.5676e-03, 4.0538e-07, 1.5426e-02, 9.5713e-01, 2.4878e-02],
>
> [4.6963e-02, 4.9191e-04, 7.5844e-02, 5.9361e-01, 2.8309e-01]],
>
> grad_fn=<SoftmaxBackward0>)

It can be seen that the obtained weights are all within (0, 1), which is very suitable for weighted calculations.

## Output as weighted sum

According to the definition of Attention, after obtaining the weights matrix, it needs to be multiplied by the Value matrix to obtain the final Attention output:

$$\text{output} = \text{attention\_weights} \cdot V$$

```
# Final output of self-attention
output = attention_weights @ V
print("shape of output:", output.shape)
print(output)
```

> shape of output: torch.Size([5, 4])
>
> tensor([[-1.0221, -1.1318, -1.0966, -1.2475],
>
> [ 1.6613, 1.7716, 2.1347, 2.5049],
>
> [-1.3064, -1.3985, -1.3982, -1.5418],
>
> [-2.2928, -2.2490, -2.4211, -2.5138],
>
> [-1.6010, -1.6693, -1.7563, -1.9028]], grad_fn=<MmBackward0>)

Note the change in dimensions. [5,5] * [5,4] results in the shape of the final output being [5,4], which is exactly the same as the shape of the input X. That is to say, after the Attention transformation, the dimension of the output remains the same as the input.

So far, we have completed the full calculation of Attention.

# Simple Self-Attention Code

Having understood the above process, we can very conveniently build the self-attention module using PyTorch, as follows:

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # (d_in, d_out)
        self.W_Q = nn.Linear(d_in, d_out, bias=False)
        self.W_K = nn.Linear(d_in, d_out, bias=False)
        self.W_V = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        # (seq_len, d_in) x (d_in, d_out) -> (seq_len, d_out)
        Q = self.W_Q(x)  # equal to: x @ W_Q.T
        K = self.W_K(x)
        V = self.W_V(x)

        # (seq_len, d_out) x (d_out, seq_len) -> (seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / K.shape[-1]**0.5
        # (seq_len, seq_len)
        weights = F.softmax(scores, dim=-1)
        # (seq_len, seq_len) x (seq_len, d_out) -> # (seq_len, d_out)
        context = weights @ V
        return context
```

```
torch.manual_seed(123)
sa = SelfAttentionV2(4, 4)
output = sa(X)
print(output)
```

> tensor([[ 0.1318, -0.1000, -0.4239, -0.0858],
>
> [-0.0532, 0.2164, -0.8386, -0.1107],
>
> [ 0.2318, -0.2270, -0.4083, -0.0919],
>
> [ 0.4762, -0.5514, -0.2901, -0.0859],
>
> [ 0.0700, -0.0399, -0.3281, -0.0728]], grad_fn=<MmBackward0>)

Please pay special attention to the changes in tensor dimensions.

*Note: Here, nn.Linear is used to construct a linear layer to initialize the Q weights, and you can also manually create the parameter matrix using nn.Parameter(torch.rand(d_in, d_out)). However, the internal initialization methods of the two are slightly different.*

# Casual Attention: Mask future words

The calculation of the above Attention weights includes the entire context, but this is inconsistent with the training process of large generative models, for example:

> “He sat by the **bank** of the river.”

When the model is trying to generate "bank", the context can only contain the preceding words, not the subsequent words such as "river". This is because if we allow the model to see the entire context during the training phase, the trained Model Generalization Ability will be poor; when faced with a real generation task, the performance will be subpar. Therefore, we need to block the "future words" that the model should not see to better enhance the model's capabilities.

In the Embedding section, we already know that the training of large models is an autoregressive process, as follows:

> Once --> upon
>
> Once upon --> a
>
> Once upon a --> time
>
> Once upon a time --> there
>
> Once upon a time there --> were

Actually, masking future words becomes very simple, requiring only the removal of all elements above the diagonal in the aforementioned Attention.

For example, we can leverage the following lower triangular matrix to easily mask out future tokens through matrix operations. The mask matrix is as follows:

```
import torch

context_size = attention_scores.shape[0]
# Lower triangular mask
mask = torch.tril(torch.ones(context_size, context_size))
print(mask)
mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
print(mask)
```

> tensor([[1., 0., 0., 0., 0.],
>
> [1., 1., 0., 0., 0.],
>
> [1., 1., 1., 0., 0.],
>
> [1., 1., 1., 1., 0.],
>
> [1., 1., 1., 1., 1.]])
>
> tensor([[0., -inf, -inf, -inf, -inf],
>
> [0., 0., -inf, -inf, -inf],
>
> [0., 0., 0., -inf, -inf],
>
> [0., 0., 0., 0., -inf],
>
> [0., 0., 0., 0., 0.]])

We finally obtained a matrix with zeros in the lower triangle and negative infinity in the upper triangle.

The process of masking is simply matrix addition, as follows:

```
print("original scores: \n", attention_scores)
# Apply mask to scores
masked_scores = attention_scores + mask
print("masked scores:\n", masked_scores)
```

> original scores:
>
> tensor([[ 0.1551, -1.0237, 0.3512, 0.9140, 0.5323],
>
> [-1.2857, 8.7238, -2.7508, -7.3460, -4.6522],
>
> [ 0.3042, -1.4816, 0.7240, 1.5888, 0.7321],
>
> [ 1.4368, -7.3169, 3.2298, 7.3577, 3.7078],
>
> [ 0.4611, -4.0977, 0.9404, 2.9979, 2.2575]], grad_fn=<DivBackward0>)
>
> masked scores:
>
> tensor([[ 0.1551, -inf, -inf, -inf, -inf],
>
> [-1.2857, 8.7238, -inf, -inf, -inf],
>
> [ 0.3042, -1.4816, 0.7240, -inf, -inf],
>
> [ 1.4368, -7.3169, 3.2298, 7.3577, -inf],
>
> [ 0.4611, -4.0977, 0.9404, 2.9979, 2.2575]], grad_fn=<AddBackward0>)

As can be seen, we only retain the lower triangular part of the Attention scores, while the upper triangular part is filled with -inf. We use -inf because a subsequent softmax operation is required, and softmax(-inf) = 0, which contributes nothing to the calculation of weights.

# Dropout

Additionally, to enhance Model Generalization Ability, a commonly used technique is random dropout, i.e., dropout. We provide a simple example of dropout using PyTorch code as follows:

```
import torch

torch.manual_seed(123)

# Create a dropout layer with 20% dropout rate
dropout = torch.nn.Dropout(0.2)
dropout.train()  # Explicitly set to training mode to enable dropout

example = torch.ones(5, 5)
print("Input tensor:\n",example)

# Apply dropout to the input tensor
output = dropout(example)
print("tensor after Dropout:\n",output)
print(f"Number of zeros in output: {(output == 0).sum().item()}")
print(f"Output mean value (should be ~1.0 due to scaling): {output.mean().item():.4f}")
```

> Input tensor:
>
> tensor([[1., 1., 1., 1., 1.],
>
> [1., 1., 1., 1., 1.],
>
> [1., 1., 1., 1., 1.],
>
> [1., 1., 1., 1., 1.],
>
> [1., 1., 1., 1., 1.]])
>
> tensor after Dropout:
>
> tensor([[1.2500, 1.2500, 1.2500, 1.2500, 1.2500],
>
> [1.2500, 1.2500, 1.2500, 0.0000, 1.2500],
>
> [0.0000, 1.2500, 1.2500, 1.2500, 1.2500],
>
> [1.2500, 1.2500, 1.2500, 1.2500, 1.2500],
>
> [1.2500, 1.2500, 1.2500, 1.2500, 1.2500]])
>
> Number of zeros in output: 2
>
> Output mean value (should be ~1.0 due to scaling): 1.1500

As can be seen, in the 5x5 all-1 matrix, some values were set to 0, and the remaining values became 1.25. This is because when performing dropout, PyTorch scales according to scale=1/(1-drop_rate) to keep the overall mean unchanged.

However, it seems that the above results do not quite meet expectations, because the dimensions of the matrix used in the example are too small. Don't forget that statistical probability only works for Big data. Simply changing the above dimensions to 500x500, you can see that the mean is still very close to 1. In the real environment of gpt2, as previously mentioned, the dimensions of the weights in this step are seq_len x seq_len, which is 1024 x 1024, actually very large.

In the Attention mechanism, dropout is applied to the weights obtained after softmax, and the code is as follows:

```
weights = F.softmax(masked_scores, dim=-1)
print("weights after mask: \n", weights)
torch.manual_seed(123)
output = dropout(weights)
print("weights after Dropout: \n", output)
```

> weights after mask:
>
> tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
>
> [4.4967e-05, 9.9996e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
>
> [3.7185e-01, 6.2345e-02, 5.6581e-01, 0.0000e+00, 0.0000e+00],
>
> [2.6332e-03, 4.1573e-07, 1.5819e-02, 9.8155e-01, 0.0000e+00],
>
> [4.6963e-02, 4.9191e-04, 7.5844e-02, 5.9361e-01, 2.8309e-01]],
>
> grad_fn=<SoftmaxBackward0>)
>
> weights after Dropout:
>
> tensor([[1.2500e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
>
> [5.6209e-05, 1.2499e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
>
> [0.0000e+00, 7.7931e-02, 7.0726e-01, 0.0000e+00, 0.0000e+00],
>
> [3.2914e-03, 5.1966e-07, 1.9774e-02, 1.2269e+00, 0.0000e+00],
>
> [5.8704e-02, 6.1489e-04, 9.4804e-02, 7.4201e-01, 3.5387e-01]],
>
> grad_fn=<MulBackward0>)

As can be seen, softmax is applied to masked scores, resulting in weights with only the lower triangular part; when weights go through dropout, 20% of the values are set to 0, and the remaining values are scaled to 1.25 times.

# Casual Self-Attention Code

Based on the previous SimpleSelfAttention, we add mask and dropout to obtain the complete code as follows:

```
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    """
Implements single-head causal self-attention with optional dropout.
"""
def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        # (d_in, d_out)
        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Create a fixed causal mask (upper triangular) [1 means "mask"]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        # x: shape (batch_size, seq_len, d_in)
        batch_size, seq_len, _ = x.size()
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / (d_out ** 0.5)  # (batch_size, seq_len, seq_len)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], -torch.inf)

        # Compute softmax weights and apply dropout
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Compute output
        output = weights @ V  # (batch_size, seq_len, d_out)
        return output
```

In this code, to better align with the real environment, we added a batch dimension to the input X, changing its dimension to: (batch_size, seq_len, d_in), and the dimension of the final output also correspondingly changed to (batch_size, seq_len, d_out).

We simulate the calculation of the final context vector through the above CausalAttention by generating two batches of random matrices with a maximum length of 5 and a dimension of 4. The code is as follows:

```
torch.manual_seed(123)

batch = torch.randn(2, 5, 4)  # (batch_size=2, seq_len=5, d_in=4)
d_in = 4
d_out = 4
context_length = batch.size(1)

ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
context_vecs = ca(batch)

print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:\n", context_vecs)
```

> context_vecs.shape: torch.Size([2, 5, 4])
>
> context_vecs:
>
> tensor([[[-0.0487, -0.0112, 0.0449, 0.3506],
>
> [ 0.0439, 0.1278, 0.1848, 0.1733],
>
> [-0.2467, -0.1078, 0.2722, 0.5128],
>
> [-0.1638, 0.0053, 0.3753, 0.3111],
>
> [ 0.0264, 0.1455, 0.3622, 0.0182]],
>
>
>
>
> [[ 0.0960, 0.4257, 1.7419, 0.2045],
>
> [-0.0967, 0.2774, 1.1946, 0.5023],
>
> [ 0.1017, 0.2037, 0.4849, 0.1862],
>
> [-0.0775, 0.1062, 0.3737, 0.3387],
>
> [-0.1181, -0.0113, 0.1070, 0.2743]]], grad_fn=<UnsafeViewBackward0>)

Please pay special attention that the dimension of the finally generated context vector must be exactly the same as that of the input vector.

So far, we have completed [the complete calculation of single-head attention in the Attention paper ](https://arxiv.org/pdf/1706.03762).

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/25280eb13c12476592d8345fe5d5cfd0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752401940&x-orig-sign=xMgEUKjlOFd5TywQIT2D9eaQ%2BTY%3D)

# Multi-Head Attention

Previously, we completed the calculation of single-head attention. To further enhance the model's expressive ability, we further introduced multi-head attention calculation, as shown in the Attention paper image above.

For example, in

> “The cat that the dog chased was black.”

In this example, multiple attention heads can be used to focus on different semantic structures respectively:

> "cat" <------ "was" (Head 1 strong attention) Head 1: Focus on subject-predicate (sentence backbone)
>
> "that", "dog", "chased" ----> "cat" (Head 2 strong attention) Head 2: Focus on the modification structure of the attributive clause
>
> "dog" ----> "chased" (Head 3 strong attention) Head 3: Focus on object structure
>
> "cat" <---- "black" (Head 4 attention) Head 4: Focus on adjective modification relationships

## Concat heads code

The most straightforward implementation of multi-head is to directly repeat the above single-head multiple times and then stack them together. The code is as follows:

```
class MultiHeadAttentionWrapper(nn.Module):
    """
Implements multi-head self-attention by stacking multiple heads.
"""

def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.head_dim = d_out // num_heads
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, self.head_dim, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return output
```

```
torch.manual_seed(123)

batch = torch.randn(2, 5, 6)  # (batch_size=2, seq_len=5, d_in=6)
d_in = 6
d_out = 6
context_length = batch.size(1)

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0,num_heads=2)
context_vecs = mha(batch)

print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:\n", context_vecs)
```

> context_vecs.shape: torch.Size([2, 5, 6])
>
> context_vecs:
>
> tensor([[[-0.0067, -0.0370, 0.2712, -0.5243, -0.0242, -0.0438],
>
> [-0.1782, 0.0173, -0.0166, -0.2391, -0.0284, 0.2177],
>
> [-0.1541, 0.2878, -0.2018, 0.2535, 0.0242, 0.3002],
>
> [-0.2817, 0.5219, -0.0699, 0.5508, -0.2767, 0.3709],
>
> [-0.0355, -0.1721, 0.0981, 0.2389, -0.1460, 0.1938]],
>
>
>
>
> [[ 0.7943, -1.9382, 0.2171, -1.6710, 0.7970, -1.3094],
>
> [ 0.2519, -1.1446, 0.2991, -1.5203, 0.3135, -0.9541],
>
> [ 0.1920, -0.8646, 0.3794, -0.9135, 0.0203, -0.5454],
>
> [ 0.2565, -0.8320, 0.1292, -0.9259, 0.2156, -0.4762],
>
> [ 0.1519, -0.5043, 0.1079, -0.3281, 0.1523, -0.1446]]],
>
> grad_fn=<CatBackward0>)

In the above code, we manually simulated 2 heads and adjusted the dimension of a single head through d_out // num_heads.

## Weight split code

But in fact, the above code is not a true implementation of MHA (Multi-Head Attention). The method of stacking matrices is less efficient. A better approach is to first perform a large projection using a large matrix and then split it, which is equivalent to weight splits.

The code is as follows:

```
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
Implements multi-head attention by splitting the attention matrix into multiple heads.
"""

def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
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
        scores = Q @ K.transpose(-2, -1) / (d_out ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

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
```

```
torch.manual_seed(123)

batch = torch.randn(2, 5, 6)  # (batch_size=2, seq_len=5, d_in=6)
d_in = 6
d_out = 6
context_length = batch.size(1)

mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0,num_heads=2)
context_vecs = mha(batch)

print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:\n", context_vecs)
```

> context_vecs.shape: torch.Size([2, 5, 6])
>
> context_vecs:
>
> tensor([[[-0.5829, -0.5644, 0.1930, -0.1541, 0.2518, -0.2252],
>
> [-0.2962, -0.2681, 0.1179, 0.1136, 0.0953, -0.4015],
>
> [-0.2039, -0.0745, 0.1557, -0.0494, 0.1125, -0.5282],
>
> [-0.2540, 0.1181, 0.2729, -0.1182, 0.0321, -0.5292],
>
> [-0.2007, 0.0280, 0.1645, -0.0798, 0.1264, -0.5020]],
>
>
>
>
> [[-0.2307, -1.7354, -0.4065, 0.3778, 0.9090, -0.1498],
>
> [-0.5355, -1.2480, -0.0049, 0.1522, 0.5635, -0.0269],
>
> [-0.4674, -0.8466, 0.0176, 0.1337, 0.4053, -0.2230],
>
> [-0.3683, -0.6768, 0.0088, 0.0933, 0.3034, -0.3600],
>
> [-0.2545, -0.5944, -0.0236, 0.0762, 0.3629, -0.3780]]],
>
> grad_fn=<ViewBackward0>)

Compared to before:

1) A unified large matrix W_Q is used for projection, and then split into multiple heads through the view operation.

2) An additional linear mapping was applied to the output to further fuse the multi-heads. However, this step is not strictly necessary.

During the operation, we should pay special attention to the changes in tensor dimensions. As long as we understand the changes in tensor dimensions, we can basically figure out the logic of the entire computation.

By now, we have completed the code implementation of MHA, which is also the most core implementation in the Transformer architecture of GPT2.
















# [Handcrafted Large Model] Writing GPT2 from Scratch - Attention

# Understand Attention in an Intuitive Way

Previously, we discussed token Embedding and position Embedding. However, embedding is essentially still a vector related to the token itself. Once training is completed, embedding becomes fixed weights. We need to introduce a new mechanism to focus on the dependencies between tokens, that is, the context in which the token resides.

The following example:

> “He sat by the **bank** of the river.”
>
> "The cat that the dog chased was black."

Whether "bank" refers to a bank or a riverbank depends on the nearby "river"; although "black" is closer to "dog", we still know that it actually refers to "cat".

Focusing on the impact of different tokens in a sentence with emphasis, that is, paying attention to semantic relevance, is precisely the starting point of the Attention mechanism. In other words, the essence of the Attention mechanism is to enable the model to independently judge which tokens are more relevant at each step and construct the context accordingly.

In [the paper on the Attention ](https://arxiv.org/abs/1706.03762)mechanism, the three tensors Query/Key/Value were innovatively introduced:

> An attention function can be described as mapping a query and a set of key-value pairs to an output,
>
> where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
>
> of the values, where the weight assigned to each value is computed by a compatibility function of the
>
> query with the corresponding key.

Where: Query represents a question or concern, Key represents the index of information, and Value represents the specific value of information.

Intuitively, Attention simulates the process of "query - retrieval - extraction", and the following simple example helps to understand:

1) Suppose you go to the library to look for books (query is the topic you want to know),

2) The library has many shelf labels (keys) and books (values),

3) First, check how relevant (compatibility) each bookshelf label is to the topic you want,

4) Then you decide how many books to take from which bookshelves based on relevance (weighted as compatibility),

5) Combine the content of the books you've obtained (weighted sum), and that will be your final answer (output).

# Scaled Dot-Product Attention

## Attention Definition

The strict definition of Attention is as follows:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

Here, Q, K, and V represent the matrices corresponding to query, key, and value respectively; while dk represents the dimension of the matrix, used for scaling.

The definition is concise, and in fact, the calculation process is also particularly clear and simple.

Take the following example with 5 tokens and an embedding dimension of 4:

```
import torch
import torch.nn as nn

torch.manual_seed(123)

tokens = ["Once", "upon", "a", "time", "there"]
token_to_idx = {token: idx for idx, token in enumerate(tokens)}
embedding_dim = 4

embedding_layer = nn.Embedding(num_embeddings=len(tokens), embedding_dim=embedding_dim)

input_indices = torch.tensor([token_to_idx[token] for token in tokens])  # [0,1,2,3,4]
X = embedding_layer(input_indices)
print("shape of input X:", X.shape)
print(X)
```

> shape of input X: torch.Size([5, 4])
>
> tensor([[ 0.3374, -0.1778, -0.3035, -0.5880],
>
> [ 1.5810, 1.3010, 1.2753, -0.2010],
>
> [-0.1606, -0.4015, 0.6957, -1.8061],
>
> [-1.1589, 0.3255, -0.6315, -2.8400],
>
> [-0.7849, -1.4096, -0.4076, 0.7953]], grad_fn=<EmbeddingBackward0>)

Obtain a 5*4 two-dimensional matrix, as follows:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ee944a4110d44a8e9805660762805311~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752401941&x-orig-sign=hGHZfHlgyr8sz0WpNNH6nTOeO2M%3D)

## Q K V matrix

Based on this, the method of creating the Q/K/V matrices is extremely simple. It only requires specifying the dimensions, randomly initializing to obtain the initial matrix, and then using this matrix to perform a linear mapping on the input X, as follows:

```
torch.manual_seed(123)

W_Q = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim), requires_grad=False)
W_K = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim), requires_grad=False)
W_V = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim), requires_grad=False)

print("shape of W_Q:", W_Q.shape)
print("W_Q:", W_Q)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("shape of Q:", Q.shape)
print("Q:", Q)
```

> shape of W_Q: torch.Size([4, 4])
>
> W_Q: Parameter containing:
>
> tensor([[0.2961, 0.5166, 0.2517, 0.6886],
>
> [0.0740, 0.8665, 0.1366, 0.1025],
>
> [0.1841, 0.7264, 0.3153, 0.6871],
>
> [0.0756, 0.1966, 0.3164, 0.4017]])
>
> shape of Q: torch.Size([5, 4])
>
> Q: tensor([[-0.0136, -0.3159, -0.2211, -0.2307],
>
> [ 0.7839, 2.8310, 0.9140, 2.0175],
>
> [-0.0858, -0.2806, -0.4474, -0.3992],
>
> [-0.6501, -1.3338, -1.3449, -2.3394],
>
> [-0.3515, -1.7666, -0.2669, -0.6454]], grad_fn=<MmBackward0>)

Please note that the dimension of X is [5, 4], the dimension of W_Q obtained by random initialization is [4, 4], and according to matrix multiplication, the dimension of Q obtained is [5, 4].

Here, matrix multiplication is used (@ is equivalent to torch.matmul), which is equivalent to performing a linear projection on the original input X.

Notably, all three of W_Q, W_K, and W_V are trainable parameters. This means that their initial values are not important; what matters is the constructed space Degree of Freedom and the information pathway.

## Similarity as Scores

According to the above formula, $$\text{scores} = Q K^T$$, we need to calculate the dot product between Q and K to compute the correlation or similarity between them.

```
scores = Q @ K.T
print("shape of scores:", scores.shape)
print("scores:", scores)
```

> shape of scores: torch.Size([5, 5])
>
> scores: tensor([[ 0.3101, -2.0474, 0.7024, 1.8280, 1.0647],
>
> [ -2.5714, 17.4476, -5.5017, -14.6920, -9.3044],
>
> [ 0.6084, -2.9632, 1.4480, 3.1775, 1.4642],
>
> [ 2.8736, -14.6337, 6.4597, 14.7155, 7.4156],
>
> [ 0.9222, -8.1955, 1.8808, 5.9959, 4.5150]],
>
> grad_fn=<MmBackward0>)

In the above example, the dimensions of Q and K are [5,4]. Transposing K results in dimensions [4,5]. Taking the dot product of the two yields a matrix of [5,5].

*Note: What is actually done here is batch dot product, i.e., matrix multiplication is used.*

## Scaled Scores

$$ \text{scaled\_scores} = \frac{Q K^T}{\sqrt{d_k}} $$

```
import math

attention_scores = scores / math.sqrt(embedding_dim)
print(attention_scores)
```

> tensor([[ 0.1551, -1.0237, 0.3512, 0.9140, 0.5323],
>
> [-1.2857, 8.7238, -2.7508, -7.3460, -4.6522],
>
> [ 0.3042, -1.4816, 0.7240, 1.5888, 0.7321],
>
> [ 1.4368, -7.3169, 3.2298, 7.3577, 3.7078],
>
> [ 0.4611, -4.0977, 0.9404, 2.9979, 2.2575]], grad_fn=<DivBackward0>)

So why do we perform scaling, and why do we choose the above values for scaling? Scaling is mainly to compress the score, avoid an overly extreme distribution in the subsequent softmax output, and make layer computation smoother; choosing the square root of dk should have its mathematical and statistical significance. However, personally, it still seems to be an empirical and compromise solution, and dividing by other values is also reasonable, so there's no need to pay excessive attention, as it's essentially just a data regularization method.

At this point, we have completed the calculation of Scaled Attention scores.

## Compute Attention Weights via Softmax

To convert attention scores into usable weights, further normalization is required, namely through the softmax operation:

$$\text{attention\_weights} = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right)$$

Draw a picture to take a look at the softmax function, which is extremely simple, as follows:

```
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
scores = torch.stack([x, torch.zeros_like(x)], dim=1)
softmax_vals = torch.softmax(scores, dim=1)

plt.plot(x.numpy(), softmax_vals[:,0].numpy())
plt.show()
```

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d8f90d3ab562453eabc11f4159e3ef84~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752401940&x-orig-sign=4APiNMhA2w99xFa6vM1DIUCeo1M%3D)

It can be seen that the softmax function compresses all inputs into the range (0, 1), making them look more like probability values.

*Note: Softmax is essentially a data normalization method and can also be replaced with other similar functions.*

We directly use the softmax function provided by PyTorch to calculate as follows:

```
attention_weights = torch.softmax(attention_scores, dim=-1)
print("shape of attention_weights:", attention_weights.shape)
print(attention_weights)
```

> shape of attention_weights: torch.Size([5, 5])
>
> tensor([[1.6344e-01, 5.0283e-02, 1.9885e-01, 3.4910e-01, 2.3833e-01],
>
> [4.4966e-05, 9.9994e-01, 1.0389e-05, 1.0494e-07, 1.5519e-06],
>
> [1.2761e-01, 2.1395e-02, 1.9418e-01, 4.6106e-01, 1.9576e-01],
>
> [2.5676e-03, 4.0538e-07, 1.5426e-02, 9.5713e-01, 2.4878e-02],
>
> [4.6963e-02, 4.9191e-04, 7.5844e-02, 5.9361e-01, 2.8309e-01]],
>
> grad_fn=<SoftmaxBackward0>)

It can be seen that the obtained weights are all within (0, 1), which is very suitable for weighted calculations.

## Output as weighted sum

According to the definition of Attention, after obtaining the weights matrix, it needs to be multiplied by the Value matrix to obtain the final Attention output:

$$\text{output} = \text{attention\_weights} \cdot V$$

```
# Final output of self-attention
output = attention_weights @ V
print("shape of output:", output.shape)
print(output)
```

> shape of output: torch.Size([5, 4])
>
> tensor([[-1.0221, -1.1318, -1.0966, -1.2475],
>
> [ 1.6613, 1.7716, 2.1347, 2.5049],
>
> [-1.3064, -1.3985, -1.3982, -1.5418],
>
> [-2.2928, -2.2490, -2.4211, -2.5138],
>
> [-1.6010, -1.6693, -1.7563, -1.9028]], grad_fn=<MmBackward0>)

Note the change in dimensions. [5,5] * [5,4] results in the shape of the final output being [5,4], which is exactly the same as the shape of the input X. That is to say, after the Attention transformation, the dimension of the output remains the same as the input.

So far, we have completed the full calculation of Attention.

# Simple Self-Attention Code

Having understood the above process, we can very conveniently build the self-attention module using PyTorch, as follows:

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # (d_in, d_out)
        self.W_Q = nn.Linear(d_in, d_out, bias=False)
        self.W_K = nn.Linear(d_in, d_out, bias=False)
        self.W_V = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        # (seq_len, d_in) x (d_in, d_out) -> (seq_len, d_out)
        Q = self.W_Q(x)  # equal to: x @ W_Q.T
        K = self.W_K(x)
        V = self.W_V(x)

        # (seq_len, d_out) x (d_out, seq_len) -> (seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / K.shape[-1]**0.5
        # (seq_len, seq_len)
        weights = F.softmax(scores, dim=-1)
        # (seq_len, seq_len) x (seq_len, d_out) -> # (seq_len, d_out)
        context = weights @ V
        return context
```

```
torch.manual_seed(123)
sa = SelfAttentionV2(4, 4)
output = sa(X)
print(output)
```

> tensor([[ 0.1318, -0.1000, -0.4239, -0.0858],
>
> [-0.0532, 0.2164, -0.8386, -0.1107],
>
> [ 0.2318, -0.2270, -0.4083, -0.0919],
>
> [ 0.4762, -0.5514, -0.2901, -0.0859],
>
> [ 0.0700, -0.0399, -0.3281, -0.0728]], grad_fn=<MmBackward0>)

Please pay special attention to the changes in tensor dimensions.

*Note: Here, nn.Linear is used to construct a linear layer to initialize the Q weights, and you can also manually create the parameter matrix using nn.Parameter(torch.rand(d_in, d_out)). However, the internal initialization methods of the two are slightly different.*

# Casual Attention: Mask future words

The calculation of the above Attention weights includes the entire context, but this is inconsistent with the training process of large generative models, for example:

> “He sat by the **bank** of the river.”

When the model is trying to generate "bank", the context can only contain the preceding words, not the subsequent words such as "river". This is because if we allow the model to see the entire context during the training phase, the trained Model Generalization Ability will be poor; when faced with a real generation task, the performance will be subpar. Therefore, we need to block the "future words" that the model should not see to better enhance the model's capabilities.

In the Embedding section, we already know that the training of large models is an autoregressive process, as follows:

> Once --> upon
>
> Once upon --> a
>
> Once upon a --> time
>
> Once upon a time --> there
>
> Once upon a time there --> were

Actually, masking future words becomes very simple, requiring only the removal of all elements above the diagonal in the aforementioned Attention.

For example, we can leverage the following lower triangular matrix to easily mask out future tokens through matrix operations. The mask matrix is as follows:

```
import torch

context_size = attention_scores.shape[0]
# Lower triangular mask
mask = torch.tril(torch.ones(context_size, context_size))
print(mask)
mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
print(mask)
```

> tensor([[1., 0., 0., 0., 0.],
>
> [1., 1., 0., 0., 0.],
>
> [1., 1., 1., 0., 0.],
>
> [1., 1., 1., 1., 0.],
>
> [1., 1., 1., 1., 1.]])
>
> tensor([[0., -inf, -inf, -inf, -inf],
>
> [0., 0., -inf, -inf, -inf],
>
> [0., 0., 0., -inf, -inf],
>
> [0., 0., 0., 0., -inf],
>
> [0., 0., 0., 0., 0.]])

We finally obtained a matrix with zeros in the lower triangle and negative infinity in the upper triangle.

The process of masking is simply matrix addition, as follows:

```
print("original scores: \n", attention_scores)
# Apply mask to scores
masked_scores = attention_scores + mask
print("masked scores:\n", masked_scores)
```

> original scores:
>
> tensor([[ 0.1551, -1.0237, 0.3512, 0.9140, 0.5323],
>
> [-1.2857, 8.7238, -2.7508, -7.3460, -4.6522],
>
> [ 0.3042, -1.4816, 0.7240, 1.5888, 0.7321],
>
> [ 1.4368, -7.3169, 3.2298, 7.3577, 3.7078],
>
> [ 0.4611, -4.0977, 0.9404, 2.9979, 2.2575]], grad_fn=<DivBackward0>)
>
> masked scores:
>
> tensor([[ 0.1551, -inf, -inf, -inf, -inf],
>
> [-1.2857, 8.7238, -inf, -inf, -inf],
>
> [ 0.3042, -1.4816, 0.7240, -inf, -inf],
>
> [ 1.4368, -7.3169, 3.2298, 7.3577, -inf],
>
> [ 0.4611, -4.0977, 0.9404, 2.9979, 2.2575]], grad_fn=<AddBackward0>)

As can be seen, we only retain the lower triangular part of the Attention scores, while the upper triangular part is filled with -inf. We use -inf because a subsequent softmax operation is required, and softmax(-inf) = 0, which contributes nothing to the calculation of weights.

# Dropout

Additionally, to enhance Model Generalization Ability, a commonly used technique is random dropout, i.e., dropout. We provide a simple example of dropout using PyTorch code as follows:

```
import torch

torch.manual_seed(123)

# Create a dropout layer with 20% dropout rate
dropout = torch.nn.Dropout(0.2)
dropout.train()  # Explicitly set to training mode to enable dropout

example = torch.ones(5, 5)
print("Input tensor:\n",example)

# Apply dropout to the input tensor
output = dropout(example)
print("tensor after Dropout:\n",output)
print(f"Number of zeros in output: {(output == 0).sum().item()}")
print(f"Output mean value (should be ~1.0 due to scaling): {output.mean().item():.4f}")
```

> Input tensor:
>
> tensor([[1., 1., 1., 1., 1.],
>
> [1., 1., 1., 1., 1.],
>
> [1., 1., 1., 1., 1.],
>
> [1., 1., 1., 1., 1.],
>
> [1., 1., 1., 1., 1.]])
>
> tensor after Dropout:
>
> tensor([[1.2500, 1.2500, 1.2500, 1.2500, 1.2500],
>
> [1.2500, 1.2500, 1.2500, 0.0000, 1.2500],
>
> [0.0000, 1.2500, 1.2500, 1.2500, 1.2500],
>
> [1.2500, 1.2500, 1.2500, 1.2500, 1.2500],
>
> [1.2500, 1.2500, 1.2500, 1.2500, 1.2500]])
>
> Number of zeros in output: 2
>
> Output mean value (should be ~1.0 due to scaling): 1.1500

As can be seen, in the 5x5 all-1 matrix, some values were set to 0, and the remaining values became 1.25. This is because when performing dropout, PyTorch scales according to scale=1/(1-drop_rate) to keep the overall mean unchanged.

However, it seems that the above results do not quite meet expectations, because the dimensions of the matrix used in the example are too small. Don't forget that statistical probability only works for Big data. Simply changing the above dimensions to 500x500, you can see that the mean is still very close to 1. In the real environment of gpt2, as previously mentioned, the dimensions of the weights in this step are seq_len x seq_len, which is 1024 x 1024, actually very large.

In the Attention mechanism, dropout is applied to the weights obtained after softmax, and the code is as follows:

```
weights = F.softmax(masked_scores, dim=-1)
print("weights after mask: \n", weights)
torch.manual_seed(123)
output = dropout(weights)
print("weights after Dropout: \n", output)
```

> weights after mask:
>
> tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
>
> [4.4967e-05, 9.9996e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
>
> [3.7185e-01, 6.2345e-02, 5.6581e-01, 0.0000e+00, 0.0000e+00],
>
> [2.6332e-03, 4.1573e-07, 1.5819e-02, 9.8155e-01, 0.0000e+00],
>
> [4.6963e-02, 4.9191e-04, 7.5844e-02, 5.9361e-01, 2.8309e-01]],
>
> grad_fn=<SoftmaxBackward0>)
>
> weights after Dropout:
>
> tensor([[1.2500e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
>
> [5.6209e-05, 1.2499e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
>
> [0.0000e+00, 7.7931e-02, 7.0726e-01, 0.0000e+00, 0.0000e+00],
>
> [3.2914e-03, 5.1966e-07, 1.9774e-02, 1.2269e+00, 0.0000e+00],
>
> [5.8704e-02, 6.1489e-04, 9.4804e-02, 7.4201e-01, 3.5387e-01]],
>
> grad_fn=<MulBackward0>)

As can be seen, softmax is applied to masked scores, resulting in weights with only the lower triangular part; when weights go through dropout, 20% of the values are set to 0, and the remaining values are scaled to 1.25 times.

# Casual Self-Attention Code

Based on the previous SimpleSelfAttention, we add mask and dropout to obtain the complete code as follows:

```
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    """
Implements single-head causal self-attention with optional dropout.
"""
def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        # (d_in, d_out)
        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Create a fixed causal mask (upper triangular) [1 means "mask"]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        # x: shape (batch_size, seq_len, d_in)
        batch_size, seq_len, _ = x.size()
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / (d_out ** 0.5)  # (batch_size, seq_len, seq_len)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], -torch.inf)

        # Compute softmax weights and apply dropout
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Compute output
        output = weights @ V  # (batch_size, seq_len, d_out)
        return output
```

In this code, to better align with the real environment, we added a batch dimension to the input X, changing its dimension to: (batch_size, seq_len, d_in), and the dimension of the final output also correspondingly changed to (batch_size, seq_len, d_out).

We simulate the calculation of the final context vector through the above CausalAttention by generating two batches of random matrices with a maximum length of 5 and a dimension of 4. The code is as follows:

```
torch.manual_seed(123)

batch = torch.randn(2, 5, 4)  # (batch_size=2, seq_len=5, d_in=4)
d_in = 4
d_out = 4
context_length = batch.size(1)

ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
context_vecs = ca(batch)

print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:\n", context_vecs)
```

> context_vecs.shape: torch.Size([2, 5, 4])
>
> context_vecs:
>
> tensor([[[-0.0487, -0.0112, 0.0449, 0.3506],
>
> [ 0.0439, 0.1278, 0.1848, 0.1733],
>
> [-0.2467, -0.1078, 0.2722, 0.5128],
>
> [-0.1638, 0.0053, 0.3753, 0.3111],
>
> [ 0.0264, 0.1455, 0.3622, 0.0182]],
>
>
>
>
> [[ 0.0960, 0.4257, 1.7419, 0.2045],
>
> [-0.0967, 0.2774, 1.1946, 0.5023],
>
> [ 0.1017, 0.2037, 0.4849, 0.1862],
>
> [-0.0775, 0.1062, 0.3737, 0.3387],
>
> [-0.1181, -0.0113, 0.1070, 0.2743]]], grad_fn=<UnsafeViewBackward0>)

Please pay special attention that the dimension of the finally generated context vector must be exactly the same as that of the input vector.

So far, we have completed [the complete calculation of single-head attention in the Attention paper ](https://arxiv.org/pdf/1706.03762).

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/40fe4d1c57374a06ad35ed9e0f5f025c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752401940&x-orig-sign=34ieOhczJrCA0RJKxWAJeYCO7Ho%3D)

# Multi-Head Attention

Previously, we completed the calculation of single-head attention. To further enhance the model's expressive ability, we further introduced multi-head attention calculation, as shown in the Attention paper image above.

For example, in

> “The cat that the dog chased was black.”

In this example, multiple attention heads can be used to focus on different semantic structures respectively:

> "cat" <------ "was" (Head 1 strong attention) Head 1: Focus on subject-predicate (sentence backbone)
>
> "that", "dog", "chased" ----> "cat" (Head 2 strong attention) Head 2: Focus on the modification structure of the attributive clause
>
> "dog" ----> "chased" (Head 3 strong attention) Head 3: Focus on object structure
>
> "cat" <---- "black" (Head 4 attention) Head 4: Focus on adjective modification relationships

## Concat heads code

The most straightforward implementation of multi-head is to directly repeat the above single-head multiple times and then stack them together. The code is as follows:

```
class MultiHeadAttentionWrapper(nn.Module):
    """
Implements multi-head self-attention by stacking multiple heads.
"""

def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.head_dim = d_out // num_heads
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, self.head_dim, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return output
```

```
torch.manual_seed(123)

batch = torch.randn(2, 5, 6)  # (batch_size=2, seq_len=5, d_in=6)
d_in = 6
d_out = 6
context_length = batch.size(1)

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0,num_heads=2)
context_vecs = mha(batch)

print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:\n", context_vecs)
```

> context_vecs.shape: torch.Size([2, 5, 6])
>
> context_vecs:
>
> tensor([[[-0.0067, -0.0370, 0.2712, -0.5243, -0.0242, -0.0438],
>
> [-0.1782, 0.0173, -0.0166, -0.2391, -0.0284, 0.2177],
>
> [-0.1541, 0.2878, -0.2018, 0.2535, 0.0242, 0.3002],
>
> [-0.2817, 0.5219, -0.0699, 0.5508, -0.2767, 0.3709],
>
> [-0.0355, -0.1721, 0.0981, 0.2389, -0.1460, 0.1938]],
>
>
>
>
> [[ 0.7943, -1.9382, 0.2171, -1.6710, 0.7970, -1.3094],
>
> [ 0.2519, -1.1446, 0.2991, -1.5203, 0.3135, -0.9541],
>
> [ 0.1920, -0.8646, 0.3794, -0.9135, 0.0203, -0.5454],
>
> [ 0.2565, -0.8320, 0.1292, -0.9259, 0.2156, -0.4762],
>
> [ 0.1519, -0.5043, 0.1079, -0.3281, 0.1523, -0.1446]]],
>
> grad_fn=<CatBackward0>)

In the above code, we manually simulated 2 heads and adjusted the dimension of a single head through d_out // num_heads.

## Weight split code

But in fact, the above code is not a true implementation of MHA (Multi-Head Attention). The method of stacking matrices is less efficient. A better approach is to first perform a large projection using a large matrix and then split it, which is equivalent to weight splits.

The code is as follows:

```
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
Implements multi-head attention by splitting the attention matrix into multiple heads.
"""

def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
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
        scores = Q @ K.transpose(-2, -1) / (d_out ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

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
```

```
torch.manual_seed(123)

batch = torch.randn(2, 5, 6)  # (batch_size=2, seq_len=5, d_in=6)
d_in = 6
d_out = 6
context_length = batch.size(1)

mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0,num_heads=2)
context_vecs = mha(batch)

print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:\n", context_vecs)
```

> context_vecs.shape: torch.Size([2, 5, 6])
>
> context_vecs:
>
> tensor([[[-0.5829, -0.5644, 0.1930, -0.1541, 0.2518, -0.2252],
>
> [-0.2962, -0.2681, 0.1179, 0.1136, 0.0953, -0.4015],
>
> [-0.2039, -0.0745, 0.1557, -0.0494, 0.1125, -0.5282],
>
> [-0.2540, 0.1181, 0.2729, -0.1182, 0.0321, -0.5292],
>
> [-0.2007, 0.0280, 0.1645, -0.0798, 0.1264, -0.5020]],
>
>
>
>
> [[-0.2307, -1.7354, -0.4065, 0.3778, 0.9090, -0.1498],
>
> [-0.5355, -1.2480, -0.0049, 0.1522, 0.5635, -0.0269],
>
> [-0.4674, -0.8466, 0.0176, 0.1337, 0.4053, -0.2230],
>
> [-0.3683, -0.6768, 0.0088, 0.0933, 0.3034, -0.3600],
>
> [-0.2545, -0.5944, -0.0236, 0.0762, 0.3629, -0.3780]]],
>
> grad_fn=<ViewBackward0>)

Compared to before:

1) A unified large matrix W_Q is used for projection, and then split into multiple heads through the view operation.

2) An additional linear mapping was applied to the output to further fuse the multi-heads. However, this step is not strictly necessary.

During the operation, we should pay special attention to the changes in tensor dimensions. As long as we understand the changes in tensor dimensions, we can basically figure out the logic of the entire computation.

By now, we have completed the code implementation of MHA, which is also the most core implementation in the Transformer architecture of GPT2.