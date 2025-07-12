# Build, Train and Finetune GPT2 from Scratch

> What I cannot create, I do not understand.
>
> *—— Richard* *Feynman*

-   The best way to understand large models should be to implement them from scratch by doing it yourself. The "large" in large models lies in the parameters (often dozens of billions), not in the amount of code (even very powerful models are only a few hundred lines of code). In this way, we can think about problems, discover problems, and solve problems while writing code.

-   This article does not delve into the underlying principles but provides the simplest possible implementation to facilitate an overall understanding of large models.

-   Referenced [the tutorials of Sebastian Raschka](https://www.manning.com/authors/sebastian-raschka) and [Andrej Karpathy](https://karpathy.ai/), reorganized them, and optimized the core code to make it simpler and clearer.

-   No prior experience, possess basic Python skills, and understand the basic operations of Pytorch and Tensor.

-   Resources: All code runs on a personal computer without the need for a GPU. All data used are publicly available datasets.

-   Series Articles: Will be divided into the following 5 articles

    -   **[Handcrafted LLM: Build GPT2 from Scratch — Embedding](https://weikuo0506.github.io/posts/HandcraftedLLM/BuildGPT2fromScratch%E2%80%94Embedding)**: Introduce how to go from text to token and then to vector; understand the concept of BPE; learn to use sliding window sampling; understand that the essence of Embedding is a table lookup operation; understand positional encoding.
    -   **[Handcrafted LLM: Build GPT2 from Scratch - Attention](https://weikuo0506.github.io/posts/HandcraftedLLM/BuildGPT2fromScratch%E2%80%94Attention)**: Understanding the Attention Mechanism, Masking Future Words, Dropout for Random Discarding, Implementing Single and Multi-Head Attention Mechanisms.
    -   **[Handcrafted LLM: Build GPT2 from Scratch — Model](https://weikuo0506.github.io/posts/HandcraftedLLM/BuildGPT2fromScratch%E2%80%94Model)**: Build the complete framework of GPT2, understand LayerNorm and ReLU activation, implement Transformer Block; use untrained GPT2 to complete text.
    -   **[Handcrafted LLM: Train GPT2 from Scratch](https://weikuo0506.github.io/posts/HandcraftedLLM/TrainGPT2fromScratch):** Understanding Cross-Entropy, implementing the calculation of Loss on datasets and batches; implementing training code and training on an ultra-small dataset; implementing methods to control randomness in decoding, including temperature and top k; attempting to train on a larger dataset, and learning to save and load model parameters.
    -   **[Handcrafted LLM: Fine-tune GPT2 from Scratch](https://weikuo0506.github.io/posts/HandcraftedLLM/FinetuneGPT2fromScratch):** Manually load public model weights; fine-tune GPT2 using a tiny dataset to enable GPT2 to respond to instructions rather than complete text; evaluate the training results using locally run llama3.

-  This series was originally published in Chinese here: [手搓大模型](https://juejin.cn/column/7525657014703276047)



* * *