# Recurrent Neural Networks (RNN) & LSTM — Technical Q&A

A curated, de-duplicated and technically validated Q&A on **Vanilla RNNs, Bidirectional RNNs, Stacked RNNs, BPTT and LSTMs**, compiled from a live lecture discussion. Every answer is kept to 4–5 lines.

**Notation used throughout**

| Symbol | Meaning |
|---|---|
| $x_t$ | Input at time step $t$ |
| $h_t$ | Hidden state (short-term context) at time $t$ |
| $c_t$ | Cell state (long-term memory, LSTM only) |
| $\hat{y}_t$ | Output / prediction at time $t$ |
| $W_x, W_h$ | Weight matrices for input and hidden state |
| $f_t, i_t, o_t$ | Forget, input and output gates (LSTM) |
| $\odot$ | Hadamard (element-wise) product |

---

## Table of Contents

1. [Why RNNs — Motivation & Comparisons](#1-why-rnns--motivation--comparisons)
2. [Vanilla RNN Internals](#2-vanilla-rnn-internals)
3. [RNN Architectures & Variants](#3-rnn-architectures--variants)
4. [Training RNNs — BPTT](#4-training-rnns--bptt)
5. [Limitations of Vanilla RNNs](#5-limitations-of-vanilla-rnns)
6. [LSTM — Cell State and Gates](#6-lstm--cell-state-and-gates)
7. [Applications & Practical Notes](#7-applications--practical-notes)

---

## 1. Why RNNs — Motivation & Comparisons

### Q1. Why is a feedforward neural network (FNN) unable to tell the difference in ordering of a sequence?
In an FNN every input is flattened and connected to all neurons of the first layer simultaneously, so the whole sentence is consumed in a single shot. Position information is only implicit in *which* weight an element happens to land on — permute the input and you get a completely unrelated mapping, not a shifted one. An RNN instead consumes tokens one at a time and carries $h_{t-1}$ forward, so order is encoded in the *process* itself. That recurrence is what gives RNNs their edge on sequential data.

### Q2. How is an RNN different from a CNN? Is an RNN just a "better CNN"?
They solve different problems, not better/worse versions of each other. A CNN exploits **spatial locality and translation invariance** through shared kernels — ideal for images. An RNN exploits **temporal dependency** by maintaining a state across time steps. A 1-D CNN on ECG/EEG can classify local waveform shapes but only within its receptive field; it has no persistent state carrying information across the full signal. An RNN can, in principle, propagate information across the entire sequence.

### Q3. RNNs look like an evolution of FNNs. Do they still matter in the era of Transformers?
Yes, though the framing needs correction: Transformers are **not** recurrent — they handle ordering via *positional encodings* plus self-attention, and they do it very well. RNNs remain relevant because they are $O(n)$ in sequence length with constant memory per step, whereas vanilla self-attention is $O(n^2)$. This makes them attractive for streaming, on-device, low-latency and long-signal settings. Recurrence has also returned in modern hybrids such as state-space models (S4/Mamba) and RWKV.

### Q4. Are RNNs memoryless, or do they have limited memory?
They have **limited, lossy memory**. The hidden state $h_t$ is a fixed-size tensor that stores a compressed summary of everything seen so far. Because that budget never grows, every new input overwrites and dilutes older content. So it is not memorylessness — it is a bounded, decaying memory, which is precisely the limitation LSTMs were designed to relax.

### Q5. How does an RNN predict stock prices or weather, and how is that different from linear regression?
Linear regression is stateless: it maps a fixed feature vector to an output with no notion of history. To give it memory you must hand-engineer lagged features, and the input dimensionality grows **linearly** with the lookback window. An RNN instead recycles a fixed-size hidden state, so the parameter count stays constant no matter how long the history. It also models non-linear temporal interactions that a linear model cannot represent.

### Q6. In FNNs each hidden neuron already sees all neurons of the previous layer. What extra information do RNNs/LSTMs provide?
An FNN's connectivity is across the **layer (depth) axis** only. RNNs add a second axis — **time** — by feeding $h_{t-1}$ back into the same cell. That recurrent edge is the extra information channel: it carries an accumulated summary of all previous time steps. LSTMs extend this further with a separate cell state $c_t$ dedicated to long-range information.

---

## 2. Vanilla RNN Internals

### Q7. Is there a separate RNN cell for every token in a sentence?
No. There is **one** cell whose parameters $(W_x, W_h, b)$ are reused at every time step. Diagrams that show a row of boxes are the "unrolled" view — a visualisation of the same cell applied repeatedly, not multiple cells. This weight sharing is what keeps the parameter count independent of sequence length and lets a trained model handle sequences of any length.

### Q8. How is the time step tracked in an RNN?
There is no explicit clock or timestamp variable. The time step is defined by the **position index in the input tensor**: element $i$ of the sequence *is* time step $i$. A 10-word sentence simply means 10 iterations of the recurrence. In PyTorch this appears as the sequence-length dimension of a `(batch, seq_len, features)` tensor.

### Q9. What is the difference between $h_t$ and $y_t$ in an RNN cell?
$h_t$ is the **internal state** — the running summary of the sequence up to time $t$, computed from $h_{t-1}$ and $x_t$. $y_t$ (or $\hat{y}_t$) is the **task output**, produced by projecting $h_t$ through an output layer. Their roles differ: $h_t$ is passed to the next time step, $\hat{y}_t$ is compared against the target in the loss.

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h), \qquad \hat{y}_t = \phi_y(W_y h_t + b_y)$$

### Q10. What are $W_x$ and $W_h$?
$W_x$ is the input-to-hidden weight matrix applied to $x_t$; $W_h$ is the hidden-to-hidden (recurrent) matrix applied to $h_{t-1}$. Their shapes are $(\text{hidden dim} \times \text{input dim})$ and $(\text{hidden dim} \times \text{hidden dim})$ respectively. $W_h$ is the one repeatedly multiplied at every step, which is why its spectral properties drive vanishing/exploding gradients.

### Q11. What is the input to the hidden layer at any step?
Exactly two things: the **current input** $x_t$ and the **previous hidden state** $h_{t-1}$. They are transformed by $W_x$ and $W_h$, summed with a bias, and passed through $\tanh$. At $t=0$, $h_{-1}$ is typically initialised to zeros (or occasionally learned).

### Q12. Why is $\tanh$ used to update the hidden state?
$\tanh$ squashes the pre-activation into $[-1, 1]$, which keeps the recurrent state bounded and prevents it from blowing up as it is fed back repeatedly. It is zero-centred, so gradients are better behaved than with sigmoid. Its gradient is also stronger near zero ($\max = 1$ vs. $0.25$ for sigmoid), which slightly mitigates vanishing gradients.

### Q13. How is the activation function related to dimensions? It is confusing versus matrix multiplication.
Keep the two operations separate. The **matrix multiply** changes dimensionality (e.g. input dim → hidden dim). The **activation** is applied element-wise *after* it, so it preserves shape exactly: a $(1 \times 128)$ input gives a $(1 \times 128)$ output. Nothing is mixed across the vector by the activation — every element is transformed independently, in parallel.

### Q14. What is temporal parameter sharing?
It means the **same weight matrices are reused at every time step** of the sequence. Step 1 processes word 1 with $W_x, W_h$; step 2 processes word 2 with the *identical* $W_x, W_h$ plus the hidden state from step 1. This gives length-independence and massively fewer parameters. Note: what is shared are the *parameters*; the hidden state is *propagated*, not shared.

### Q15. So the weight matrix is the same for the whole input sequence?
Yes — within a single recurrent layer, one set of weights serves all time steps. In a **stacked (deep) RNN**, however, each layer $k$ has its own $(W_x^{(k)}, W_h^{(k)})$. So weights are shared across *time*, but not across *depth*.

### Q16. Why is the last layer represented as $Y$?
That block is not part of the recurrence — it is a fully connected (linear) head that maps the final hidden representation to the task output space. For classification it produces logits over classes; for regression it produces continuous values. It is where the hidden features become an actual prediction.

### Q17. Why do the equations use both $t$ and $k$?
$t$ indexes the **time step** (position within the sequence) and $k$ indexes the **layer/depth** in a stacked RNN. So $h_t^{(k)}$ is the hidden state of layer $k$ at time $t$. Information flows rightward along $t$ and upward along $k$: layer $k$'s output at time $t$ becomes layer $k+1$'s input at time $t$.

---

## 3. RNN Architectures & Variants

### Q18. What do one-to-many, many-to-one and many-to-many mean?
They describe the input/output cardinality of the sequence task:
- **One-to-many** — single input, sequence output (image captioning).
- **Many-to-one** — sequence input, single output (sentiment classification).
- **Many-to-many (aligned)** — output at every step (POS tagging, video frame labelling).
- **Many-to-many (encoder–decoder)** — sequence in, sequence out with different lengths (translation).

### Q19. Why is language translation many-to-many?
The source sentence is a sequence of tokens (many inputs) and the translation is a sequence of tokens (many outputs). Crucially the two lengths need **not** match — 8 English words may become 11 Hindi words. This is handled by an encoder–decoder setup: the encoder compresses the source into a context, and the decoder emits target tokens one at a time until an end-of-sequence marker.

### Q20. Why do we need multiple layers in an RNN? Can we use just one?
One layer is perfectly functional. Stacking helps for the same reason depth helps in an MLP — **hierarchical feature abstraction**. Lower layers capture local patterns (morphology, short phrases), higher layers capture longer-range structure (syntax, topic). The trade-off is more parameters, slower training and higher overfitting risk.

### Q21. What problem does a Bidirectional RNN solve?
It addresses tasks where the **entire sequence is available up front** and future context matters for the current output. A unidirectional RNN at word 3 knows nothing about word 10. A BiRNN runs one RNN left→right and another right→left, then concatenates the two hidden states so every position sees both past and future. Typical uses: translation, summarisation, NER, POS tagging, document classification.

### Q22. Can you give a concrete example of a Bidirectional RNN?
Take English→Hindi translation. (1) The complete English sentence is available before translation begins. (2) The correct rendering of a word often depends on words that come *after* it — resolving "bank" needs the later "river" or "loan". The forward RNN encodes left-context, the backward RNN encodes right-context, and the concatenated $[\overrightarrow{h_t}; \overleftarrow{h_t}]$ gives a fully contextualised representation.

### Q23. In a Bidirectional RNN, do the forward and backward passes run concurrently?
Yes. The two directions are **independent computation graphs** over the same input, so they can be evaluated in parallel on a GPU. Within each direction the computation remains strictly sequential. Their outputs are merged only afterwards, usually by concatenation (sometimes sum or average).

### Q24. How does the reverse RNN get its input in a BiRNN?
It reads the *same* sequence, just traversed from the last element to the first. So the backward cell first sees $x_T$, then $x_{T-1}$, down to $x_1$, building $\overleftarrow{h_t}$. This is why BiRNNs cannot be used for real-time/streaming tasks — the full sequence must exist before the backward pass can start.

### Q25. How is the loss computed for a Bidirectional RNN?
The two directional hidden states are first merged into a single representation per position, which is projected to one prediction. The loss is then computed exactly as usual between that prediction and the target. During backpropagation the gradient flows into **both** directional sub-networks, each unrolled along its own time axis.

### Q26. Can some layers be unidirectional and others bidirectional?
Yes — this is architecturally legal and is used in practice (e.g. some speech models put bidirectional layers at the bottom and unidirectional ones on top to limit latency). The only hard requirement is dimensional compatibility, since a bidirectional layer outputs $2\times$ the hidden size. That said, uniform stacks are far more common, so treat mixing as a deliberate design choice.

---

## 4. Training RNNs — BPTT

### Q27. Does backpropagation work for RNNs? Is their accuracy lower because of it?
Backpropagation absolutely works for RNNs — the variant used is **Backpropagation Through Time (BPTT)**. The network is unrolled across time steps into an equivalent deep feedforward graph and standard chain-rule differentiation is applied. Because the same weights appear at every step, their gradients from all time steps are **summed** before the update. Accuracy limits come from vanishing gradients, not from any absence of backpropagation.

### Q28. What is the purpose of Backpropagation Through Time?
The forward pass builds a dependency chain across time, so the loss at step $t$ depends on every earlier step. BPTT walks that chain backwards, accumulating $\partial L / \partial W$ contributions from each step. In practice **Truncated BPTT** is used, cutting the backward pass after $k$ steps to bound memory and compute. Without it, credit could never be assigned to earlier inputs.

### Q29. What is an objective function?
It is the quantity being optimised — used interchangeably with loss or cost function. For sequence classification it is typically cross-entropy; for sequence regression, MSE. In sequence tasks the per-step losses are summed or averaged over time steps before backpropagation.

### Q30. If weights keep changing during training, do they also keep updating at test time?
No. At test/inference time the parameters are **frozen**. You call `model.eval()`, wrap the pass in `torch.no_grad()`, and never invoke `loss.backward()` or `optimizer.step()`. `eval()` additionally switches dropout off and makes batch-norm use running statistics. Hidden and cell states still update — they are activations, not parameters.

### Q31. In the CNN practical, `forward()` was never called explicitly. How is it used?
Because the model subclasses `nn.Module`, calling `model(input)` invokes `__call__`, which dispatches to `forward()` after running registered hooks. So `output = model(x)` *is* the forward pass. You should never call `model.forward(x)` directly, as that bypasses those hooks.

---

## 5. Limitations of Vanilla RNNs

### Q32. Does the hidden state cache most of a long book it was trained on?
No — this conflates two things. **Training** adjusts weights; the hidden state is a *runtime* activation, recomputed from scratch for every input sequence. It is a fixed-size tensor holding a compressed summary up to the current step only, not a retrievable archive of the corpus. Its finite capacity is exactly why very long sequences degrade.

### Q33. What happens with very long sequences? Can an RNN still remember the beginning?
Poorly. Gradients must traverse a product of many Jacobians involving $W_h$; when the relevant singular values are $<1$ the signal decays exponentially (**vanishing gradient**), and when $>1$ it explodes. Practically, vanilla RNNs struggle beyond roughly 10–20 steps of dependency. Exploding gradients are patched with gradient clipping; vanishing gradients require architectural change — LSTM or GRU.

### Q34. How far back can an RNN remember? Can we define a fixed limit?
There is no clean number. Effective memory depends on hidden dimension, the spectral radius of $W_h$, the activation, sequence statistics and the task itself. Empirically vanilla RNNs handle short spans well and degrade smoothly rather than cutting off sharply. LSTMs/GRUs push this to a few hundred steps; attention removes the recurrence bottleneck entirely.

### Q35. Why do recent inputs overwrite the past when memory is abundant? What is the real constraint?
The constraint is **representational, not storage**. "Memory" here is a fixed-dimensional encoded vector in GPU RAM, not bytes on a disk. Each step applies a contractive non-linear transform to that vector, so older contributions are progressively squeezed out — capacity is bounded by the hidden dimension, not by available hardware. Storing raw history on disk would defeat the purpose of a learned compressed representation.

### Q36. Is there an explicit mechanism in an RNN deciding what to keep or discard?
In a vanilla RNN, no — retention is an **implicit** side-effect of the learned $W_h$ and the $\tanh$ non-linearity. There is no dedicated gate, so useful information can be overwritten simply because the update rule is uniform. LSTMs make this explicit with forget and input gates, but note those gates are still *learned*, not hand-specified.

### Q37. Given these drawbacks, are vanilla RNNs still used or fully replaced?
LSTMs and GRUs have largely superseded them for serious sequence modelling, and Transformers dominate NLP. Vanilla RNNs are still useful for **short sequences, tiny models and embedded/low-power settings**, where fewer parameters and lower latency matter more than long-range memory. They also remain the pedagogical foundation for understanding every gated variant.

---

## 6. LSTM — Cell State and Gates

### Q38. What is the difference between an RNN cell and an LSTM cell?
An RNN cell maintains **one** state ($h_t$) and one $\tanh$ update — compact, but with short effective memory. An LSTM maintains **two** states: $h_t$ (short-term/output) and $c_t$ (long-term memory), regulated by three gates. The cell state's largely additive update path lets gradients flow across many steps without repeated squashing. Cost: roughly 4× the parameters of a comparable RNN.

### Q39. What is the cell state, and what data structure is it?
It is a **tensor** — typically shaped `(batch, hidden_dim)` per layer — held in GPU/CPU RAM, not an object with methods or a persistent store. Conceptually it is a memory highway running horizontally through time, modified only by element-wise operations. That is why gradients survive along it far better than through $h_t$ alone.

### Q40. Where are the hidden state and cell state stored — inside neurons?
Neither is stored inside a neuron. Both are **activations** — intermediate tensors produced by the cell's equations and held in RAM/VRAM during the forward pass. They are retained during training only because autograd needs them for the backward pass, then freed. An LSTM cell is nothing more than the concrete implementation of its equations.

### Q41. What is a candidate cell, and why does it matter?
The candidate $\tilde{c}_t = \tanh(W_c h_{t-1} + U_c x_t + b_c)$ is the **proposed new content** for memory at this step. It is what *could* be written; the input gate $i_t$ decides how much actually is. Separating proposal from admission is what lets the LSTM add new information without wiping existing memory. Note there is no "candidate gate" — the candidate is a state, gated by $i_t$.

### Q42. Why do gates use sigmoid while the candidate and hidden state use $\tanh$?
The two serve different roles. Gates use **sigmoid** to produce values in $[0,1]$ that act as multiplicative valves — 0 blocks completely, 1 passes fully, 0.5 passes half. Content uses **$\tanh$** to produce values in $[-1,1]$, so information is signed and zero-centred and can both add to and subtract from memory. Squashing a gate with $\tanh$ would allow negative "amounts", which is meaningless.

### Q43. Both the forget gate and input gate are sigmoid layers followed by a Hadamard product — so what makes one "forget"?
The difference is **what each multiplies**, not their internal form. $f_t$ multiplies the *previous* cell state $c_{t-1}$, so a value near 0 erases old memory. $i_t$ multiplies the *candidate* $\tilde{c}_t$, so a value near 0 blocks new information. Identical mechanics, opposite targets:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

### Q44. Are the gates applied after multiplication, or on the values?
Order matters. Each of $f_t$, $i_t$, $o_t$ and $\tilde{c}_t$ is first computed from $h_{t-1}$ and $x_t$ via its own affine transform plus non-linearity. Only then is the element-wise product taken: $f_t \odot c_{t-1}$ and $i_t \odot \tilde{c}_t$. Those two products are **added** to give $c_t$, and finally $h_t = o_t \odot \tanh(c_t)$.

### Q45. How does an LSTM decide what to forget or keep? Who sets a threshold like 0.7?
Nobody sets it — the gate values are **outputs of learned affine layers**, produced fresh at every time step from $h_{t-1}$ and $x_t$. Values such as 0.7 in slides are illustrative, not hard-coded. The gate weights are trained by gradient descent on the task loss, so "what to remember" is whatever reduces that loss. Relevance is therefore defined entirely by the objective.

### Q46. What is the difference between $W_f$ and $U_f$ in the forget gate equation?
Each gate has **two** weight matrices: one for the recurrent input and one for the current input. Convention here: $W$ multiplies $h_{t-1}$, $U$ multiplies $x_t$.

$$f_t = \sigma(W_f h_{t-1} + U_f x_t + b_f)$$

The same pairing applies to $(W_i, U_i)$, $(W_o, U_o)$ and $(W_c, U_c)$ — 8 matrices plus 4 biases in total.

### Q47. What does the dot symbol mean in the cell-state update?
It denotes the **Hadamard product** — element-wise multiplication, written $\odot$. It is *not* a dot product or matrix multiplication. Both operands share the same shape and the output preserves it, which is exactly what makes per-element gating possible.

### Q48. Does an LSTM store every prediction it has made, unlike an RNN which overwrites?
No. The LSTM does not accumulate an ever-growing log; $c_t$ is still a **fixed-size tensor**. The difference is *how* it is updated: the RNN fully rewrites $h_t$ each step, while the LSTM applies a gated, largely additive update that can preserve selected content nearly unchanged over long spans. That selective retention — not infinite capacity — is the real advantage.

### Q49. What is the first step of learning if we start at time step 0?
The first step processes the first element of the sequence — the first word, frame or sample. States are initialised to zero ($h_{-1} = c_{-1} = 0$), and the forward pass runs to the end of the sequence. Only *after* the full unrolled forward pass is the loss computed and BPTT run. Weights therefore update once per batch, not once per time step.

### Q50. Why is an LSTM less interpretable than a vanilla RNN?
An RNN cell has a single equation and one state, so the mapping from input to state change is relatively traceable. An LSTM has four interacting sub-networks and two states, roughly quadrupling the parameters. What a given gate dimension has learned to represent is rarely human-meaningful, and observed behaviour emerges from the *interaction* of gates rather than any one of them.

---

## 7. Applications & Practical Notes

### Q51. Are RNNs used for prediction, recognition, or both?
Both. The recurrent stack is a **feature extractor** over sequences; the head decides the task. A softmax head gives classification (sentiment, intent, POS tags); a linear head gives regression (price, temperature). The same architecture also serves generation, where each step's output feeds the next step's input.

### Q52. Can RNNs be used for OCR?
Yes — the standard approach is a **CRNN**: a CNN extracts visual features from image slices, and an RNN (usually a BiLSTM) reads that feature sequence left-to-right to output characters, typically trained with CTC loss. It is best described as **many-to-many**, since a sequence of image slices maps to a sequence of characters. Bidirectionality helps because neighbouring glyphs disambiguate each other.

### Q53. Does Google use RNNs for search auto-suggestions?
Auto-suggestion systems combine several techniques — query logs, prefix tries, ranking models and neural sequence models. RNNs/LSTMs were widely used for the neural completion component historically; production systems today lean heavily on Transformer-based models. It is best described as an ensemble of retrieval and neural ranking, not a single architecture.

### Q54. LSTMs can lose memory on long sequences — how does that affect video, which has far more input?
The same bound applies, and video is harsher because each frame is high-dimensional. Long videos are therefore handled by **chunking**: sample or segment frames, encode each with a CNN, and run the recurrent model over that reduced feature sequence. Short clips may fit entirely in the recurrent state; long ones are summarised hierarchically or handled with temporal attention.

### Q55. Can we debug and inspect these internals in real time as input is fed?
Yes. In PyTorch, request all step-wise states (`output, (h_n, c_n) = lstm(x)`), or register `forward_hook`s on the module to capture intermediate tensors. Gate activations can be logged and plotted as heatmaps over time steps to see what is retained versus erased. TensorBoard, Weights & Biases and Netron help visualise activations and graph structure.

### Q56. Why does an AI chat tool "lose context" after many prompts, and how can it be mitigated?
Two distinct causes should not be conflated. Modern chat models are Transformers with a **hard, finite context window** — once the token budget is exhausted, older turns are truncated or compressed away; this is a fixed limit, not gradual RNN-style decay. Mitigations: periodic summarisation (e.g. a `/compact` step), retrieval over stored notes, and carrying forward explicit artefacts such as specs or code. Reformulating a summary changes the prompt, so some drift in output is expected.

### Q57. What is dropout, and why exclude it from the last layer?
Dropout is a regulariser that randomly zeroes a fraction of activations during training, preventing co-adaptation of units and improving generalisation. It is active only in training and disabled by `model.eval()` at inference. It is normally omitted on the **final output layer**, since dropping logits directly injects noise into the prediction itself. In stacked RNNs it is applied between recurrent layers.

### Q58. What is meant by "synthetic" in the demo?
It refers to **synthetic data** — sequences generated programmatically from a known rule rather than collected from the real world. Because ground truth is exactly known, it is ideal for verifying that a model can learn a specific dependency (e.g. a fixed-lag pattern). Success on synthetic data validates the implementation; it does not guarantee real-world performance.

---

## Summary — Vanilla RNN vs LSTM

| Aspect | Vanilla RNN | LSTM |
|---|---|---|
| States maintained | $h_t$ only | $h_t$ and $c_t$ |
| Gates | None | Forget, Input, Output |
| Weight matrices per layer | 2 ($W_x, W_h$) | 8 (4 gates × 2) |
| Effective memory | ~10–20 steps | Hundreds of steps |
| Gradient path | Multiplicative (vanishes) | Additive along $c_t$ |
| Parameters | Fewest | ~4× an RNN |
| Interpretability | Higher | Lower |
| Best suited for | Short sequences, low-resource | Long-range dependencies |

---

## Key Equations Reference

**Vanilla RNN**

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$$
$$\hat{y}_t = \phi_y(W_y h_t + b_y)$$

**LSTM**

$$f_t = \sigma(W_f h_{t-1} + U_f x_t + b_f) \quad \text{(forget)}$$
$$i_t = \sigma(W_i h_{t-1} + U_i x_t + b_i) \quad \text{(input)}$$
$$o_t = \sigma(W_o h_{t-1} + U_o x_t + b_o) \quad \text{(output)}$$
$$\tilde{c}_t = \tanh(W_c h_{t-1} + U_c x_t + b_c) \quad \text{(candidate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

**Bidirectional RNN**

$$h_t = [\overrightarrow{h_t} \; ; \; \overleftarrow{h_t}]$$

---

*Compiled and technically validated from a lecture Q&A session. Duplicate questions merged; attributions removed.*
