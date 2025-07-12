# Build GPT2 from Scratch — Embedding

The essence of large models is space mapping (Mapping Between Spaces) and space optimization (Optimization in Latent Space). The code of large models is essentially a function approximator that maps the input space to the output space; rather than directly programming to implement rules, large models search for the optimal solution in the parameter space (weights) through a large-scale training process, enabling the mapping function to fit the true distribution of the input and output.

Embedding is the process of mapping the original input space, such as text, speech, images, videos, etc., to the intermediate space, the latent space.

# Tokenize: from text to words/tokens

For text, the first step is tokenization; for example, the following code uses the simplest character splitting to break long or short text into words or tokens.

```
import re

def tokenize(text):
    # Split by punctuation and whitespace
    tokens = re.split(r'([,.:;?_!"()']|--|\s)', text)
    # Remove empty strings and strip whitespace
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens
```

Let's take[texts in Gutenberg with fewer than 1000 words](https://www.gutenberg.org/ebooks/14838)as an example. The word segmentation results are as follows:

```
with open("Peter_Rabbit.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokens = tokenize(raw_text)
print(tokens[:10])
```

> ['Once', 'upon', 'a', 'time', 'there', 'were', 'four', 'little', 'Rabbits', ',']

# Encode: from token to ID

Suppose the large model is still just an infant, and all the knowledge it sees is only the text above. We can tokenize the text and then number the tokens starting from 0.

```
def build_vocab(whole_text):
    tokens = tokenize(whole_text)
    vocab = {token:id for id,token in enumerate(sorted(set(tokens)))}
    return vocab

vocab = build_vocab(raw_text)
print(len(vocab))
print(list(vocab.items())[:20])
```

> 405
>
> [('!', 0), ("'", 1), (',', 2), ('--', 3), ('.', 4), (':', 5), (';', 6), ('A', 7), ('After', 8), ('Also', 9), ('An', 10), ('And', 11), ('Benjamin', 12), ('Bunny', 13), ('But', 14), ('Cotton-tail', 15), ('Cottontail', 16), ('END', 17), ('Father', 18), ('First', 19)]

As can be seen, the above text has only 405 different tokens.

The process of encoding is to map different tokens to numbers starting from 0, which is like the "dictionary lookup" process of primary school students, checking the serial number of a certain word in the dictionary.

```
def encode(vocab, text):
    return [vocab[token] for token in tokenize(text)]

print(encode(vocab, "Once upon a time there were four little Rabbits"))
```

> [33, 373, 46, 354, 346, 386, 155, 210, 38]

As shown in the above example, encode returns the number of each token.

# Decode: from ID to token

The Decode process, on the contrary, is the process of restoring from the number to the original text.

```
def decode(vocab, ids):
    vocab_inverse = {id:token for token,id in vocab.items()}
    text= " ".join([vocab_inverse[id] for id in ids])
    return text

print(decode(vocab,[33, 373, 46, 354, 346, 386, 155, 210, 38]))
```

> Once upon a time there were four little Rabbits

As shown above, we successfully restored the original text based on the ID.

# Tokenizer: vocab, encode, decode

```
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_inverse = {id:token for token,id in vocab.items()}

    def encode(self, text):
        return [self.vocab[token] for token in tokenize(text)]

    def decode(self, ids):
        return " ".join([self.vocab_inverse[id] for id in ids])

tokenizer = SimpleTokenizerV1(vocab)
print(tokenizer.decode(tokenizer.encode("Once upon a time there were four little Rabbits")))
```

> Once upon a time there were four little Rabbits

Putting the above code together, we can verify that encoding the text first and then decoding it restores the original text.

*Note: Sometimes it may not succeed; because particularly simple word segmentation is deliberately used here; you can find ways to improve it.*

# Special token: UNKnown/EndOfSentence

The dictionary vocab above is clearly too small, and if it encounters new words, it will report an error, as follows:

```
print(tokenizer.decode(tokenizer.encode("Once upon a time there were four little Rabbits, and they were all very happy.")))
```

> KeyError Traceback (most recent call last)
>
> Cell In[24], line 1
>
> ----> 1 print(tokenizer.decode(tokenizer.encode("Once upon a time there were four little Rabbits, and they were all very happy.")))
>
>
>
>
> Cell In[15], line 7, in SimpleTokenizerV1.encode(self, text)
>
> 6 def encode(self, text):
>
> ----> 7 return [self.vocab[token] for token in tokenize(text)]
>
>
>
>
> Cell In[15], line 7, in <listcomp>(.0)
>
> 6 def encode(self, text):
>
> ----> 7 return [self.vocab[token] for token in tokenize(text)]
>
>
>
>
> KeyError: 'they'

Recalling kindergarten students, when they encounter an unfamiliar character, they would draw a circle. Similarly, we can add an unknown token to the vocab.

```
vocab['<unk>'] = len(vocab)

print(list(vocab.items())[-5:])
```

> [('wriggled', 401), ('you', 402), ('young', 403), ('your', 404), ('<unk>', 405)]

As mentioned above, we added <unk> at the end of the field to represent all unknown words.

Improve the above code and run it again.

```
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_inverse = {id:token for token,id in vocab.items()}

    def encode(self, text):
        unk_id = self.vocab.get("<unk>")
        return [self.vocab.get(token,unk_id) for token in tokenize(text)]

    def decode(self, ids):
        return " ".join([self.vocab_inverse[id] for id in ids])

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.decode(tokenizer.encode("Once upon a time there were four little Rabbits, and they were all very happy.")))
```

> Once upon a time there were four little Rabbits , and <unk> were all very <unk> .

It is evident that at least there are no more errors. Of course, this is still not perfect, because we cannot fully restore the original text, and all unknown tokens have become "unknown", which will inevitably result in information loss.

# BytePair Encoding: break words into chunks/subwords

Now, let's recall another method for kindergarten students to learn new words: word segmentation.

Taking the tokenizer most commonly used in current large models as an example, a word may be split into the following 4 tokens:

```
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("unbelievability"))
```

> [403, 6667, 11203, 1799]

```
print(tokenizer.decode([403,12,6667,12,11203,12,1799]))
```

> un-bel-iev-ability

[tiktoken](https://tiktokenizer.vercel.app/) is a high-performance tokenizer released by OpenAI, and the above process can also be visualized online:

<img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/9cd39b347d8a4f55819c5412e43063f3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgd2Vpa3Vv:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjc4MTEwNzg2MjY0MTk2NCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1752400751&x-orig-sign=ZVb0LsnwU2RapzEyn5IaQWms3kM%3D" width="700px">

You can check that the vocabulary size used by GPT2 is 50257.

```
print("vocab size of gpt2: ",tokenizer.n_vocab)
```

> vocab size of gpt2: 50257

By splitting in this way, GPT2 can handle any unknown word, because in the worst case, it can be split into 26 letters and punctuation marks. The Byte Pair process is to repeatedly merge the token pairs with the highest frequency to build a vocabulary of a fixed size.

*Note: The principle and detailed process of BPE will not be elaborated here. As can be intuitively understood from the above, the reason why "unbelievability" is split into "un" and "ability" is obviously because they are common prefixes and suffixes in English.*

The vocabulary is not mysterious at all, and there is no essential difference from the dictionary used by primary school students. Please see the [vocab](https://huggingface.co/gpt2/resolve/main/vocab.json) of GPT2, which has exactly 50257 entries, with the last one being "<|endoftext|>": 50256

BPE is like breaking a complete word into character puzzles, then gluing common combinations back step by step based on frequency statistics to form a token library suitable for machine processing. The merging process needs to follow the guidance of [merges ](https://huggingface.co/gpt2/resolve/main/merges.txt)until no further merging is possible, and finally each token must exist in ` vocab.json  `to serve as model input.

Throughout this article, we will consistently use this tokenizer: tokenizer = tiktoken.get_encoding("gpt2").

# Data Sampling with Sliding Window

Convert the above passage into token IDs using the GPT2 tokenizer, as follows:

```
with open("Peter_Rabbit.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print("tokens: ",len(enc_text))
print("first 15 token IDs: ", enc_text[:15])
print("first 15 tokens: ","|".join(tokenizer.decode([token]) for token in enc_text[:15]))
```

> tokens: 1547
>
> first 15 token IDs: [7454, 2402, 257, 640, 612, 547, 1440, 1310, 22502, 896, 11, 290, 511, 3891, 198]
>
> first 15 tokens: Once| upon| a| time| there| were| four| little| Rabb|its|,| and| their| names|

From now on, our focus will be on token IDs rather than the original tokens; that is, we will no longer look at the original text but only remember the word numbers.

During the training and inference process of large models, **it is essentially an autoregressive process** . That is, during the training of large models, we need to input words one by one in sequence and use the current predicted output as the input for the next step.

The maximum context length that a model can "see" at one time is called context_size. In gpt2, it is 1024, indicating that the model supports input sequences of up to 1024 tokens.

Assuming context_size is 5, then this process is as follows:

> Once --> upon
>
> Once upon --> a
>
> Once upon a --> time
>
> Once upon a time --> there
>
> Once upon a time there --> were

We use the token ID to represent:

```
context_size = 5
for i in range(1,context_size+1):
    context = enc_text[:i]
    desired = enc_text[i]
    print(context, "-->", desired)
```

> [7454] --> 2402
>
> [7454, 2402] --> 257
>
> [7454, 2402, 257] --> 640
>
> [7454, 2402, 257, 640] --> 612
>
> [7454, 2402, 257, 640, 612] --> 547

The above is the process of the sliding window, where each time the target is offset by 1 compared to the input.

During the training process, we need to divide the input into batches and shuffle them. The complete code example is as follows:

```
from torch.utils.data import Dataset
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt,tokenizer, context_size, stride):
        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > context_size, "Text is too short"

        self.input_ids = [torch.tensor(token_ids[i:i+context_size])
                          for i in range(0, len(token_ids)-context_size, stride)]
        self.target_ids = [torch.tensor(token_ids[i+1:i+context_size+1])
                          for i in range(0, len(token_ids)-context_size, stride)]
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def dataloader_v1(txt,batch_size=3,context_size=5,stride=2,shuffle=False,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer,context_size,stride)
    return DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
```

Still read the above passage, and read inputs and targets through the dataloader and constructing an iterator, both of which are token ids, as follows:

```
with open("Peter_Rabbit.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = dataloader_v1(raw_text)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("shape of input: ",inputs.shape)
print("first batch, input: \n", inputs,"\n targets: \n", targets)
```

The results are as follows:

> shape of inputs: torch.Size([3, 5])
>
> first batch, input:
>
> tensor([[ 7454, 2402, 257, 640, 612],
>
> [ 257, 640, 612, 547, 1440],
>
> [ 612, 547, 1440, 1310, 22502]])
>
> targets:
>
> tensor([[ 2402, 257, 640, 612, 547],
>
> [ 640, 612, 547, 1440, 1310],
>
> [ 547, 1440, 1310, 22502, 896]])

In the above example, batch=3 and context_size=5, so the dimensions of both inputs and targets are [3, 5], which means they are divided into 3 batches, with each batch containing at most 5 token ids; while targets are always offset by 1 compared to inputs; stride=2 indicates that the offset for each sampling is 2, that is, the second row of inputs is offset by 2 compared to the first row. Therefore, batch corresponds to the number of rows in the two-dimensional tensor, context size corresponds to the number of columns, and stride corresponds to the offset between rows. And the offset between targets and inputs is always 1, which is determined by the nature of the autoregressive training of the large model described above.

# Token Embedding: From Words to Vectors

Vectors are

-   high-dimensional
-   dense
-   learnable

Embedding is

-   looking up vectors from a big table
-   usually a matrix with shape (vocab_size, embed_dim)
-   initialized with random values
-   updated during training

Above, we converted words into discrete numbered digits, which is actually quite close to the space that large models can understand. It's just that the previous ones were consecutive integers, such as 0, 1, 2. The space that large models desire is a high-dimensional, continuous floating-point tensor. Why? The most important reason is that the tensors in the embedding space are learnable parameters, so they need to be "differentiable" to facilitate differential calculations. Another implicit meaning of learnable parameters is that the initial values are not that important; during the process of Model Training, the parameters will be continuously adjusted and optimized until a final relatively perfect state is reached.

*Note: The training process is the process of optimizing parameters according to the direction indicated by the layer, which can be understood in conjunction with Pytorch to understand the computational graph and automatic differentiation process; details will not be repeated here.*

Here we briefly illustrate the Embedding space. Suppose the total number of words in the dictionary is 10, and the dimension of Embedding is 4, as follows:

```
from torch import nn

vocab_size = 10
embed_dim = 4
torch.manual_seed(123)
token_embedding_layer = nn.Embedding(vocab_size, embed_dim)
print("token_embedding_layer shape: ", token_embedding_layer.weight.shape)
print("token_embedding_layer weight: ", token_embedding_layer.weight)
```

> token_embedding_layer shape: torch.Size([10, 4])
>
> token_embedding_layer weight: Parameter containing:
>
> tensor([[ 0.3374, -0.1778, -0.3035, -0.5880],
>
> [ 0.3486, 0.6603, -0.2196, -0.3792],
>
> [ 0.7671, -1.1925, 0.6984, -1.4097],
>
> [ 0.1794, 1.8951, 0.4954, 0.2692],
>
> [-0.0770, -1.0205, -0.1690, 0.9178],
>
> [ 1.5810, 1.3010, 1.2753, -0.2010],
>
> [ 0.9624, 0.2492, -0.4845, -2.0929],
>
> [-0.8199, -0.4210, -0.9620, 1.2825],
>
> [-0.3430, -0.6821, -0.9887, -1.7018],
>
> [-0.7498, -1.1285, 0.4135, 0.2892]], requires_grad=True)

As can be seen, the dimension of the finally generated Embedding space is (vocab_size, embed_dim); and we randomly initialized this space, resulting in a two-dimensional tensor with 10 rows and 4 columns. This is equivalent to constructing another "dictionary", except that this dictionary will be trained and optimized later.

The process of Embedding is the process of querying the "new dictionary".

```
input_ids = torch.tensor([2,3,5])
token_embeddings = token_embedding_layer(input_ids)
print("token_embeddings: \n", token_embeddings) # return row 2,3,5 of weights
```

> token_embeddings:
>
> tensor([[ 0.7671, -1.1925, 0.6984, -1.4097],
>
> [ 0.1794, 1.8951, 0.4954, 0.2692],
>
> [ 1.5810, 1.3010, 1.2753, -0.2010]], grad_fn=<EmbeddingBackward0>)

As described above, assuming that we perform Embedding on tokens with token IDs 2, 3, and 5 respectively, then the returned result is the 2nd, 3rd, and 5th rows (starting from 0) of the above two-dimensional tensor.

The above is just an example. In the real GPT2, (50257 tokens × 768 dimensions) is used. The random initialization is as follows:

```
from torch import nn

vocab_size = 50527
embed_dim = 768
torch.manual_seed(123)
token_embedding_layer_gpt2 = nn.Embedding(vocab_size, embed_dim)
print("token_embedding_layer_gpt2 shape: ", token_embedding_layer_gpt2.weight.shape)
print("token_embedding_layer_gpt2 weight: ", token_embedding_layer_gpt2.weight)
```

> token_embedding_layer_gpt2 shape: torch.Size([50527, 768])
>
> token_embedding_layer_gpt2 weight: Parameter containing:
>
> tensor([[ 0.3374, -0.1778, -0.3035, ..., -0.3181, -1.3936, 0.5226],
>
> [ 0.2579, 0.3420, -0.8168, ..., -0.4098, 0.4978, -0.3721],
>
> [ 0.7957, 0.5350, 0.9427, ..., -1.0749, 0.0955, -1.4138],
>
> ...,
>
> [-1.8239, 0.0192, 0.9472, ..., -0.2287, 1.0394, 0.1882],
>
> [-0.8952, -1.3001, 1.4985, ..., -0.5879, -0.0340, -0.0092],
>
> [-1.3114, -2.2304, -0.4247, ..., 0.8176, 1.3480, -0.5107]],
>
> requires_grad=True)

You can imagine the large two-dimensional tensor above as the following table:

```
Token ID     |     Embedding vector (768 dims)
-----------------------------------------------
0            |    [0.12, -0.03, ...,  0.88]
1            |    [0.54,  0.21, ..., -0.77]
...          |    ...
50526        |    [...]
```

Similarly, the process of Embedding remains the process of querying this huge table.

```
input_ids = torch.tensor([2,3,5])
print(token_embedding_layer_gpt2(input_ids))
```

> tensor([[ 0.7957, 0.5350, 0.9427, ..., -1.0749, 0.0955, -1.4138],
>
> [-0.0312, 1.6913, -2.2380, ..., 0.2379, -1.1839, -0.3179],
>
> [-0.4334, -0.5095, -0.7118, ..., 0.8329, 0.2992, 0.2496]],
>
> grad_fn=<EmbeddingBackward0>)

As shown in the above example, when performing Embedding on token id=2, the second row of the table is taken.

As can be seen from the above, the process of Embedding is the process of converting tokens into tensors, and it is also the process of mapping one-dimensional discrete token IDs to a high-dimensional, continuous dense space. And this process is simply an ordinary lookup operation.

# Position Embedding: From Position to Vectors

position embeddin is

-   a matrix with shape (context_size, embed_dim)
-   initialized with random values
-   a learnable parameter, updated during training

The previous section described the token embedding process, which is actually the process of querying based on the token ID in a large table. However, we need to note that the different rows in the above table seem to be unrelated.

For example, “You eat fish” and “Fish eat you” are similar at the token embedding level (since Transformer itself is order-insensitive), but their expressed semantics are completely different. Therefore, it is necessary to introduce positional information and assign positional numbers starting from 0.

As shown in the following example, it is clear at a glance whether the position 0 is "you" or "fish":

```
"You eat fish"
   ↓        ↓       ↓
[you] + P0 [eat] + P1 [fish] + P2

"Fish eat you"
   ↓         ↓         ↓
[fish] + P0 [eat] + P1 [you] + P2

→ 即使 Token 一样，只要位置不同，最终向量就不同。
→ Transformer 能区分主语、宾语等结构含义。
```

Obviously, the position encoding should start from 0 and go up to context_size - 1.

As previously mentioned, using discrete integers leads to a weakened spatial representation ability and the inability to perform automatic differentiation optimization. Therefore, similarly, we still need to convert the position numbers into high-dimensional dense tensors.

As follows, assume context_size is 5 and Embedding dimension is 4:

```
from torch import nn

context_size = 5
embed_dim = 4
torch.manual_seed(123)
position_embedding_layer = nn.Embedding(context_size, embed_dim)
print("position_embedding_layer shape: ", position_embedding_layer.weight.shape)
print("position_embedding_layer weight: ", position_embedding_layer.weight)
```

> position_embedding_layer shape: torch.Size([5, 4])
>
> position_embedding_layer weight: Parameter containing:
>
> tensor([[ 0.3374, -0.1778, -0.3035, -0.5880],
>
> [ 1.5810, 1.3010, 1.2753, -0.2010],
>
> [-0.1606, -0.4015, 0.6957, -1.8061],
>
> [-1.1589, 0.3255, -0.6315, -2.8400],
>
> [-0.7849, -1.4096, -0.4076, 0.7953]], requires_grad=True)

We will obtain a 5*4 two-dimensional tensor.

Please note that the dimensions of the position tensor are (context_size, embed_dim), which means it has the same number of columns as the token tensor, but a different number of rows.

The Position Tensor is essentially another "position dictionary", and the embedding process is a process of querying based on position numbers.

As shown in the following example,

```
input_ids = torch.tensor([2,3,5])
# use Position of input_ids, NOT values of it
position_embeddings = position_embedding_layer(torch.arange(len(input_ids)))
print("position_embeddings: \n", position_embeddings) # return row 0,1,2 of weights
```

> position_embeddings:
>
> tensor([[ 0.3374, -0.1778, -0.3035, -0.5880],
>
> [ 1.5810, 1.3010, 1.2753, -0.2010],
>
> [-0.1606, -0.4015, 0.6957, -1.8061]], grad_fn=<EmbeddingBackward0>)

The returned result is the first 3 rows of the above Embedding because the input consists of 3 tokens. Please pay special attention that when performing position Embedding, the position of the token id is used, rather than the value of the token id.

*Note: The PE used here is Learnable Absolute Positional Embeddings. In addition, there are also non-trainable, fixed position encodings, such as Sinusoidal PE (fixed position, using sin/cos), RoPE (relative rotation), etc., which have stronger performance. However, from the perspective of beginners, Learnable PE is the simplest and most easily understandable position encoding concept.*

# Input Embedding: token_embedding + position_embedding

In summary, after having token embedding and pos embedding, we only need to simply add them together to obtain the final input embedding.

```
input_embeddings = token_embeddings + position_embeddings
print("shape of input_embeddings : ",input_embeddings.shape)
print("input_embeddings: ", input_embeddings)
```

```
shape of input_embeddings :  torch.Size([3, 4])
input_embeddings:  tensor([[ 1.1045, -1.3703,  0.3948, -1.9977],
        [ 1.7603,  3.1962,  1.7707,  0.0682],
        [ 1.4204,  0.8996,  1.9710, -2.0070]], grad_fn=<AddBackward0>)
```

You can manually calculate and verify,0.7671 + 0.3374 = 1.1045




GPT2 uses position numbers of (1024 positions × 768 dimensions), where 1024 represents the maximum input length the model can handle, and 768 is the same as the token Embedding dimension, choosing a relatively high-dimensional and dense space.

We reproduce it through the following code:

```
from torch import nn

context_size = 1024
embed_dim = 768
torch.manual_seed(123)
position_embedding_layer_gpt2 = nn.Embedding(context_size, embed_dim)
print("position_embedding_layer_gpt2 shape: ", position_embedding_layer_gpt2.weight.shape)
print("position_embedding_layer_gpt2 weight: ", position_embedding_layer_gpt2.weight)
```

> position_embedding_layer_gpt2 shape: torch.Size([1024, 768])
>
> position_embedding_layer_gpt2 weight: Parameter containing:
>
> tensor([[ 0.3374, -0.1778, -0.3035, ..., -0.3181, -1.3936, 0.5226],
>
> [ 0.2579, 0.3420, -0.8168, ..., -0.4098, 0.4978, -0.3721],
>
> [ 0.7957, 0.5350, 0.9427, ..., -1.0749, 0.0955, -1.4138],
>
> ...,
>
> [-1.2094, 0.6397, 0.6342, ..., -0.4582, 1.4911, 1.2406],
>
> [-0.2253, -0.1078, 0.0479, ..., 0.2521, -0.2893, -0.5639],
>
> [-0.5375, -1.1562, 2.2554, ..., 1.4322, 1.2488, 0.1897]],
>
> requires_grad=True)




We still take the above passage as an example:

```
with open("Peter_Rabbit.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = dataloader_v1(raw_text,batch_size=3, context_size=1024,stride=2)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("shape of input: ",inputs.shape)
print("first batch, input: \n", inputs,"\n targets: \n", targets)
```

> shape of input: torch.Size([3, 1024])
>
> first batch, input:
>
> tensor([[ 7454, 2402, 257, ..., 480, 517, 290],
>
> [ 257, 640, 612, ..., 290, 517, 36907],
>
> [ 612, 547, 1440, ..., 36907, 13, 1763]])
>
> targets:
>
> tensor([[ 2402, 257, 640, ..., 517, 290, 517],
>
> [ 640, 612, 547, ..., 517, 36907, 13],
>
> [ 547, 1440, 1310, ..., 13, 1763, 1473]])

The dimension of the input is: 3 batches * 1024 context length.

We can perform Embedding on the input of batch0 as follows:

```
token_embeddings = token_embedding_layer_gpt2(inputs)
print("shape of token_embeddings: ",token_embeddings.shape)

position_embeddings = position_embedding_layer_gpt2(torch.arange(context_size))
print("shape of position_embeddings: ",position_embeddings.shape)

# token_embeddings shape: [batch_size, seq_len, embedding_dim]
# position_embeddings shape: [seq_len, embedding_dim]
# PyTorch automatically broadcasts position_embeddings across batch dimension
input_embeddings = token_embeddings + position_embeddings
print("shape of input_embeddings : ",input_embeddings.shape)
```

> shape of token_embeddings: torch.Size([3, 1024, 768])
>
> shape of position_embeddings: torch.Size([1024, 768])
>
> shape of input_embeddings : torch.Size([3, 1024, 768])

Please pay special attention to the changes in tensor shape; among them, position embedding does not have a batch dimension, but thanks to PyTorch's broadcasting mechanism, it can still be automatically added.

*Note: Tensor is the "language" of large models, and all operations of large models are essentially operations on tensors. It is recommended to familiarize yourself with common concepts and operations of tensors, such as broadcast, view, reshape, squeeze, transpose, einsum, mul, matmul, dot, etc. Tensor is the simplest and most elegant language for describing sets and high-dimensional spaces. Beginners should not think of tensors as complex mathematical problems; they can be regarded as a simple "foreign language", just a concise convention and means of expression.*

So far, we have obtained the input space that large models can truly process, namely inputs embedding, which contains token-related information from token embedding and position embedding information about token positions.