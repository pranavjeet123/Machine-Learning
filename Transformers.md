# Seq2Seq, Attention & Transformers 

A curated, de-duplicated and technically validated Q&A covering **encoder–decoder seq2seq, the context-vector bottleneck, additive attention, Query/Key/Value, self-attention, multi-head attention, positional encoding and the full Transformer block**, compiled from a live lecture discussion. Every answer is kept to 4–5 lines.

**Notation used throughout**

| Symbol | Meaning |
|---|---|
| $h_i$ | Encoder hidden state for source token $i$ |
| $s_t$ | Decoder hidden state at step $t$ |
| $e_{ti}$ | Raw alignment score between $s_{t-1}$ and $h_i$ |
| $\alpha_{ti}$ | Normalised attention weight (softmax of $e_{ti}$) |
| $c_t$ | Context vector for decoder step $t$ |
| $Q, K, V$ | Query, Key, Value matrices |
| $W_Q, W_K, W_V$ | Learned projection matrices |
| $d_k$ | Dimension of the key/query vectors |
| $d_{model}$ | Model width (512 in the original paper) |

---

## Table of Contents

1. [Seq2Seq and the Bottleneck Problem](#1-seq2seq-and-the-bottleneck-problem)
2. [Attention — Intuition and Scores](#2-attention--intuition-and-scores)
3. [Query, Key and Value](#3-query-key-and-value)
4. [Self-Attention Mechanics](#4-self-attention-mechanics)
5. [Multi-Head Attention](#5-multi-head-attention)
6. [Positional Encoding and Tokenization](#6-positional-encoding-and-tokenization)
7. [The Transformer Block — Residuals, Norm, FFN](#7-the-transformer-block--residuals-norm-ffn)
8. [Encoder, Decoder and Cross-Attention](#8-encoder-decoder-and-cross-attention)
9. [Training, Inference and Practice](#9-training-inference-and-practice)

---

## 1. Seq2Seq and the Bottleneck Problem

### Q1. What is the role of the context vector and the initial decoder state?
In classical seq2seq the encoder reads the entire source sentence and compresses it into a single fixed-size **context vector** — its final hidden state. That vector is the only channel through which source information reaches the decoder. It is typically handed to the decoder as its **initial hidden state** $s_0$, giving the decoder its starting memory. Everything the decoder generates is conditioned on that one summary.

### Q2. What exactly is the bottleneck problem in vanilla seq2seq?
The whole source sentence — 5 words or 50 — must fit into one fixed-dimensional vector. Capacity does not grow with input length, so longer sentences are compressed more lossily and early tokens fade first. This is compounded by vanishing gradients over long recurrences. The result: translation quality degrades sharply as source length grows, which is precisely the failure attention was invented to fix.

### Q3. Why is $s_0$ used? Is it random?
No. $s_0$ is the decoder's **initial hidden state**, normally initialised from the encoder's final state (sometimes through a small learned projection). It should not be confused with the `<sos>` start-of-sequence *token*, which is the first input the decoder is fed. Both are involved at step 1, but one is a state and the other is an input embedding.

### Q4. How does attention solve the bottleneck?
It removes the single-vector constraint. Instead of one frozen context, the decoder computes a **fresh context vector at every step** as a weighted sum over *all* encoder hidden states. Because every source position remains directly reachable, information no longer has to survive a long recurrent chain. This mitigates — not literally eliminates — forgetting, which is why performance on long sentences improves dramatically.

### Q5. Why is the vanilla RNN always used in these explanations?
Purely for pedagogical clarity — it is the simplest recurrence that exposes the core idea. In real seq2seq systems nobody used vanilla RNNs; LSTMs and GRUs were the practical choice because they resist forgetting far better. Treating the vanilla case first makes the motivation for gating, and later for attention, much easier to see.

### Q6. How is the "forgetting" problem generated in the first place?
It comes from repeated multiplication during backpropagation through time. The gradient at an early step is a product of many Jacobians involving the recurrent weight matrix; when the relevant singular values are below 1, that product decays exponentially. Early tokens therefore receive almost no learning signal. Transformers sidestep this entirely by replacing recurrence with direct all-to-all connections.

### Q7. Why are the examples always text encoding/decoding? Can seq2seq be used for anything else?
Text is used because it is intuitive and easy to inspect. The architecture only requires that the input be **tokenizable into a sequence of vectors** — nothing about it is language-specific. It is applied to speech (audio frames), time series (sensor readings), protein sequences, music (MIDI events) and image patches in Vision Transformers. The mechanism is modality-agnostic.

---

## 2. Attention — Intuition and Scores

### Q8. Why is the decoder state $s$ compared against each $h_i$?
Because the decoder needs to know **which source words matter right now**. Comparing $s_{t-1}$ against every encoder state $h_i$ produces an alignment score measuring how relevant source position $i$ is to the token about to be generated. Those scores become weights, so translating "bank" can lean on "river" rather than the whole sentence equally. It is a learned soft lookup, recomputed at every decoder step.

### Q9. What are $e$ and $\alpha$ here?
They are two stages of the same computation, not synonyms. $e_{ti}$ is the **raw, unbounded alignment score** produced by the scoring function. $\alpha_{ti}$ is that score after **softmax normalisation**, so the weights are non-negative and sum to 1 across source positions.

$$e_{ti} = \text{score}(s_{t-1}, h_i), \qquad \alpha_{ti} = \frac{\exp(e_{ti})}{\sum_j \exp(e_{tj})}, \qquad c_t = \sum_i \alpha_{ti} h_i$$

### Q10. When $\alpha_i$ is called "significance", is that the same as learning rate?
No — they live in completely different places. The **learning rate** is an optimiser hyperparameter that scales the weight update during gradient descent; you choose it. $\alpha$ is a **per-token importance weight** computed inside the forward pass from the data itself, and it changes at every step and for every input. One controls training dynamics, the other controls information routing.

### Q11. Is the similarity in attention cosine similarity?
Not in the standard Transformer — it is a **scaled dot product**, $QK^\top / \sqrt{d_k}$. Cosine similarity would additionally divide by both vector norms, discarding magnitude; the dot product deliberately keeps it, so a confidently-scaled key can dominate. The $\sqrt{d_k}$ divisor is variance control, not normalisation: without it, dot products in high dimensions grow large and push softmax into saturated regions with near-zero gradients.

### Q12. Are the context vectors $c_1, c_2, \dots$ computed sequentially or in parallel?
It depends on the architecture. In **RNN-based attention**, they are sequential — $c_t$ needs $s_{t-1}$, which needs the previous step's output. In a **Transformer encoder** (and in the decoder during training with teacher forcing and causal masking), all positions are computed **in parallel** as one matrix operation. That parallelism over the sequence axis is the Transformer's central practical advantage.

### Q13. Why is an attention layer needed at all?
Because a fixed context vector forces every output token to consult the same lossy summary. Attention lets each output position build its **own** view of the input, weighted by current relevance. It also creates a direct gradient path from any output to any input, bypassing long recurrent chains. In self-attention it further lets each token be re-represented in terms of its context — resolving pronouns, agreement and long-range dependencies.

---

## 3. Query, Key and Value

### Q14. Explain query and key vectors with a sentence example.
Take: *"The dog chased the ball because it was fast."* For the token **it**, the **query** encodes what it is looking for — an antecedent for a pronoun. Every other token offers a **key** advertising what it is: *dog* signals an animate noun, *ball* an inanimate object. The query matches *dog*'s key most strongly, so attention concentrates there and *dog*'s **value** (its content) is mixed into the representation of *it*.

> Query = what I need. Key = what I offer. Value = what I hand over if chosen.

### Q15. If query vectors are needed for training, do we need a huge dataset of queries?
No — queries are not data and are never collected. Each token **generates its own query** by multiplying its embedding with the learned matrix $W_Q$: $q_i = x_i W_Q$. There is only one $W_Q$ per attention head to train, regardless of how many tokens you process. The only dataset you need is plain text; queries fall out of it automatically.

### Q16. What happens if a key is new / unseen for the matrix?
Keys are not stored lookup-table entries, so "unseen" does not apply. Every key is computed fresh as $k_i = x_i W_K$, meaning any token — including one never encountered in that position or context — still produces a valid key. The only true out-of-vocabulary risk is at the tokenizer, and subword tokenization largely removes even that.

### Q17. What is $W$?
$W$ denotes a **learnable parameter matrix** — a set of weights updated by gradient descent. In attention there are three per head, $W_Q$, $W_K$ and $W_V$, which project the same input embedding into three different subspaces. Additional learned matrices appear in the output projection $W_O$ and in the feed-forward sublayers.

### Q18. How are $W_Q$, $W_K$ and $W_V$ calculated or decided?
They are not calculated by a formula — they are **initialised, then learned**. Initialisation uses a standard scheme such as Xavier/Glorot or He to keep activation variance stable at the start. From there, every value is updated by backpropagation on the task loss. Their final content is entirely determined by what minimises that loss.

### Q19. How are the embeddings generated?
Each token ID indexes into a **learned embedding matrix** of shape (vocabulary size × $d_{model}$). At initialisation those vectors are random; they are trained jointly with the rest of the network, so semantically similar tokens drift toward similar directions. Positional information is then added on top, since the embedding itself carries no notion of order.

---

## 4. Self-Attention Mechanics

### Q20. In a sentence like "I can see you", why can't $h_1$ be a scalar?
Because a single number cannot encode the many attributes a token carries — part of speech, semantics, number, tense, register and so on. Each token is represented as a **vector** (512 dimensions in the original Transformer), giving the model many independent axes to encode along. A scalar would also make the dot-product similarity between tokens degenerate to simple multiplication, destroying any notion of directional matching.

### Q21. Why is a non-linearity needed after the self-attention stage?
Self-attention is essentially a **weighted average of value vectors** — a convex combination, which is linear in $V$. Stacking such layers without a non-linearity would collapse into an equivalent single linear map, so depth would buy nothing. The position-wise feed-forward network (two linear layers with ReLU/GELU between) supplies that non-linearity, letting the model form non-linear decision boundaries.

### Q22. What role does temperature play in softmax, and how does it change the distribution?
Temperature $T$ rescales logits before softmax: $\text{softmax}(z/T)$. Low $T$ ($<1$) sharpens the distribution toward the argmax, giving more deterministic and repetitive output; high $T$ ($>1$) flattens it, increasing diversity and risk of incoherence. This is a **sampling-time** control on the output distribution. Note the $\sqrt{d_k}$ divisor inside attention is a fixed scaling constant, not a tunable temperature.

---

## 5. Multi-Head Attention

### Q23. What is the difference between attention and multi-head attention?
Single-head attention computes one set of $Q$, $K$, $V$ projections and produces one weighted representation per token. **Multi-head** attention splits $d_{model}$ into $h$ smaller subspaces (8 heads × 64 dims in the original paper), runs attention independently in each, then concatenates and projects the results through $W_O$. Parameter cost is roughly the same, but the model gains several simultaneous relational views instead of one averaged one.

### Q24. Why have multiple heads?
Because one attention distribution has to commit to a single weighting, averaging away competing relationships. With several heads, different subspaces can specialise — one tracking syntactic dependency, another coreference, another local adjacency. Probing studies confirm heads do differentiate, though the split is emergent and not always cleanly interpretable. The concatenation then makes all these views available to the next layer.

### Q25. Are heads preconfigured with semantic hyperparameters? Can we steer them toward, say, slang?
No. Heads are architecturally identical and differ only through **random initialisation plus gradient descent**; no head is assigned a linguistic role in advance. Specialisation emerges because differently-initialised heads settle into different local solutions. You influence behaviour indirectly — through training data, fine-tuning objectives or adapters — never by configuring a head directly.

### Q26. Does multi-head attention give us parallelism?
Two distinct kinds of parallelism should not be conflated. Heads are independent, so all $h$ heads are computed **in parallel** as a single batched matrix multiplication. Separately, the Transformer is parallel across **sequence positions**, which is what removes the recurrence bottleneck. Multi-head is about representational richness; the sequence-axis parallelism is what makes training fast.

---

## 6. Positional Encoding and Tokenization

### Q27. Why is positional encoding needed?
Self-attention is **permutation-equivariant** — it treats the input as a set, so shuffling tokens leaves the attention computation structurally unchanged. Without positional information, *"dog bites man"* and *"man bites dog"* would be indistinguishable. Positional encodings inject order by adding a position-dependent signal to each embedding. The original paper used fixed sinusoids; modern models mostly use learned embeddings or RoPE.

### Q28. If the same word repeats in a sentence, does it get the same encoding?
Its **token embedding** is identical, but its **final representation is not**. Two things break the tie: positional encoding differs by index, and self-attention gives each occurrence a different neighbourhood to attend over. So the two instances of "the" in *"the dog chased the ball"* leave the first layer with clearly different vectors.

### Q29. I saw tokens like "tire", "d" on the slide — is that subword tokenization?
Yes. Algorithms such as BPE, WordPiece and SentencePiece split rare words into frequent subword units, so *tired* may become `tire` + `d`. This keeps the vocabulary bounded (typically 30k–100k) while still covering any word, including misspellings and neologisms. It is why modern models effectively have no out-of-vocabulary problem.

### Q30. What is a padding sequence and why is it needed?
Batching requires a rectangular tensor, but real sequences have different lengths. If one sample has 5 tokens and another 10, the shorter is padded to 10 so the batch forms a `(2, 10)` matrix. Crucially, a **padding mask** must accompany it so attention assigns zero weight to pad positions — otherwise the model would attend to meaningless filler.

### Q31. Why is 0 used as the padding value?
Mostly convention: index 0 is reserved in the vocabulary for the `<pad>` token, making it an unambiguous, unused slot. It is also convenient because zero-initialised tensors are already padded by default. The specific value carries no meaning — correctness comes from the **mask**, not from the number chosen.

---

## 7. The Transformer Block — Residuals, Norm, FFN

### Q32. Are residual connections, layer norm and scaled dot-product attention used together or independently?
In the standard Transformer all three are used **together, in every block, always** — they are not per-use-case options. Residuals preserve gradient flow through depth, layer norm stabilises activation statistics, and the $\sqrt{d_k}$ scaling keeps softmax out of saturation. What *does* vary by architecture is placement: pre-norm (norm before the sublayer) is now preferred over the original post-norm because it trains more stably at depth.

### Q33. In the residual connection, how can the input token vector be added to the output vector — do the dimensions match?
Yes, by design. Every sublayer in a Transformer preserves width ($d_{model} = 512$ in the original), so the attention output for position 3 has exactly the same shape as the input at position 3. The addition is element-wise, position by position:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

This constant width is precisely what makes residual connections trivially valid throughout the stack.

### Q34. My concern wasn't dimensions but the actual token mapping — "I" → "Main" won't hold for all tokens, so how is the addition safe?
The encoder never reorders or substitutes tokens. Slot 3 is *"chased"* going in and is still *"chased"* coming out — only enriched with context from its neighbours. So `output[i]` and `input[i]` always describe the **same token**, and adding them is semantically coherent. Cross-lingual mapping ("I" → "Main") happens only in the **decoder**, via cross-attention, never inside an encoder residual.

### Q35. Where do values like 3.9 and 1.9 come from in the normalisation step?
They are the **mean and standard deviation** computed by layer normalisation. Unlike batch norm, layer norm computes these statistics across the **feature dimension of a single token**, independent of batch size and sequence length. Each activation is then standardised and rescaled by learned parameters $\gamma$ and $\beta$. Batch-independence is what makes it suitable for variable-length sequences.

### Q36. Does padding standardise the matrix size for $n$ inputs?
Yes — that is exactly its purpose. It makes all sequences in a batch the same length so they can be stacked into one tensor and processed with a single GPU kernel call. The accompanying mask ensures the padded positions contribute nothing to attention or to the loss.

### Q37. Why are multiple encoder and decoder layers used?
For the same reason depth helps generally: **hierarchical abstraction**. Lower layers tend to capture surface and syntactic patterns; higher layers capture semantics and long-range structure. Each layer also lets information propagate further, since attention composes across layers. Returns diminish and training cost rises, so depth is a tuned trade-off (6 layers originally, dozens to a hundred-plus in large LLMs) rather than "more is always better".

### Q38. Why is it called a Transformer?
Because it **transforms** one sequence into another — English into Hindi, for instance — using neither recurrence nor convolution, purely through attention-based re-representation. Each layer transforms token representations into contextually enriched ones. The authors have described the name as a fairly casual choice without deeper theoretical intent.

---

## 8. Encoder, Decoder and Cross-Attention

### Q39. Is it necessary to use both an encoder and a decoder in a Transformer?
No — and this is an important correction. Three families exist: **encoder-only** (BERT, RoBERTa) for understanding tasks like classification and NER; **decoder-only** (GPT, Llama, most modern LLMs) for generation; and **encoder–decoder** (original Transformer, T5, BART) for sequence-to-sequence tasks such as translation. Both halves are needed only when input and output are distinct sequences requiring cross-attention.

### Q40. In cross-attention, why does $V$ come from the encoder rather than the decoder?
Because the decoder is the one **asking**, so it supplies $Q$ — "what do I need to write next?" The information it needs lives in the source sentence, so both $K$ and $V$ come from the encoder. $K$ is used to find *where* the relevant content is; $V$ carries the content that gets pulled across. Taking $V$ from the decoder would mean retrieving from the wrong sequence entirely.

### Q41. Since $K$ and $V$ from the encoder stay fixed during autoregressive decoding, can they be computed once and cached?
Yes — this is standard practice. The source sentence is encoded once, so its $K$ and $V$ projections are computed a single time and reused for every generated token, while only the decoder's $Q$ changes per step. A related but separate optimisation, the **KV cache**, stores the decoder's own self-attention keys and values so each new token attends over cached history instead of recomputing it. Both are essential for practical inference speed.

### Q42. In translation, where does the token get matched to Hindi? Is there one big vocabulary for all languages?
Matching happens at the **output projection**: the decoder's final hidden state is multiplied by a matrix of shape ($d_{model}$ × vocab size) to produce a logit per vocabulary entry, and softmax picks the token. Multilingual models typically use a single **shared subword vocabulary** covering all supported languages. Bilingual models may instead keep separate source and target vocabularies.

---

## 9. Training, Inference and Practice

### Q43. Does "Transformers are parallel" mean GPT generates a whole paragraph at once?
No — two different things are being conflated. **Training** is parallel: with teacher forcing and causal masking, all positions are computed in one pass. **Inference in GPT-style models is autoregressive**: each token is generated, appended, and fed back, so a paragraph takes as many sequential steps as it has tokens. Non-autoregressive alternatives based on diffusion and flow matching are being researched but are not what production LLMs use today.

### Q44. Which platforms are commonly used to train these models?
**PyTorch** is the dominant framework, with JAX/Flax also widely used. **Hugging Face Transformers** provides pretrained models, tokenizers and training utilities and is the usual starting point. For scale, DeepSpeed, Megatron-LM and FSDP handle distributed and sharded training. Compute typically comes from cloud GPU/TPU providers, with Colab or Kaggle sufficient for learning-scale experiments.

### Q45. Are there known limitations of the Transformer, and is the architecture standardised across OpenAI, Claude and Gemini models?
It is **not standardised** — the skeleton is shared but the details diverge considerably. The core limitation is quadratic $O(n^2)$ attention cost in sequence length, plus a large KV cache at inference. Active efficiency work includes **Mixture-of-Experts** (only a few sub-networks fire per token), **grouped-query attention** (fewer K/V heads, smaller cache), **FlashAttention** (identical math, memory-aware GPU kernels), **RoPE** in place of sinusoidal positions, and **state-space models** such as Mamba as a non-attention alternative.

---

## Summary — Evolution of the Architecture

| Aspect | Vanilla Seq2Seq (RNN) | RNN + Attention | Transformer |
|---|---|---|---|
| Source representation | One fixed context vector | All $h_i$ available | All positions, every layer |
| Context per output step | Same for all steps | Recomputed per step | Recomputed per position/head |
| Path length input→output | $O(n)$ | $O(1)$ via attention | $O(1)$ |
| Sequence-axis parallelism | None | None (still recurrent) | Full during training |
| Long-range dependency | Poor | Good | Excellent |
| Main cost | Bottleneck + vanishing gradient | Still sequential | $O(n^2)$ attention |

---

## Key Equations Reference

**Additive (Bahdanau) attention**

$$e_{ti} = v^\top \tanh(W_s s_{t-1} + W_h h_i), \qquad \alpha_{ti} = \text{softmax}(e_{ti}), \qquad c_t = \sum_i \alpha_{ti} h_i$$

**Scaled dot-product attention**

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Multi-head attention**

$$\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$
$$\text{MultiHead}(X) = [\text{head}_1 ; \dots ; \text{head}_h] \, W_O$$

**Transformer sublayer (post-norm form)**

$$z = \text{LayerNorm}(x + \text{MultiHead}(x))$$
$$\text{out} = \text{LayerNorm}(z + \text{FFN}(z)), \qquad \text{FFN}(z) = \max(0, zW_1 + b_1)W_2 + b_2$$

**Sinusoidal positional encoding**

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

---

*Compiled and technically validated from a lecture Q&A session. Duplicate questions merged; attributions removed.*
