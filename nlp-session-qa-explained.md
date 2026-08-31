## Part 1 — Text Cleaning and Normalization

### Q1. If punctuation defines the emotion of text (confused, excited), aren't we destroying information by removing it?

**Short answer:** Yes, you are. That's why removal is a task-dependent decision, not a default step.

**Explanation:** Punctuation is genuine signal. "great" and "great!!!" carry different intensity; "Let's eat, grandma" and "Let's eat grandma" carry different meanings entirely. Question marks mark interrogatives, ellipses mark hesitation, and repeated exclamation marks correlate strongly with sentiment intensity in social media text.

The rule of thumb is: **punctuation is signal when the task depends on tone, structure, or intent; it is noise when the task depends only on topic.**

- **Keep it** — sentiment analysis, sarcasm detection, machine translation, question classification, anything fed to a Transformer.
- **Drop it** — bag-of-words topic modelling (LDA), keyword extraction, TF-IDF document clustering, where "budget!" and "budget" should collapse to the same feature.

Modern Transformer pipelines almost always preserve punctuation, because the subword tokenizer has token slots for `!`, `?`, `.` and the model learns their weight from data rather than having a human decide it in advance. Aggressive cleaning was a necessity of the sparse-feature era; with dense contextual models it is usually a liability.

---

### Q2. Can you explain NFC and NFD with more examples?

**Short answer:** Two Unicode normalization forms. NFC composes a character into the fewest code points possible; NFD decomposes it into a base character plus separate combining marks.

**Explanation:** Unicode often allows the *same visible character* to be encoded in more than one way. To a human they look identical; to a tokenizer they are different byte sequences and therefore different tokens. Normalization forces one canonical spelling before tokenization.

| Character | NFC (composed) | NFD (decomposed) |
|---|---|---|
| ñ | U+00F1 (1 code point) | n (U+006E) + ◌̃ (U+0303) — 2 code points |
| é | U+00E9 | e (U+0065) + ◌́ (U+0301) |
| ç | U+00E7 | c (U+0063) + ◌̧ (U+0327) |
| ü | U+00FC | u (U+0075) + ◌̈ (U+0308) |
| Å | U+00C5 | A (U+0041) + ◌̊ (U+030A) |
| 한 (Hangul) | U+D55C | ᄒ + ᅡ + ᆫ — 3 jamo |

**Why it matters in practice:** if a user types "café" from a Mac (often NFD) and your training corpus stored "café" as NFC, a string equality check fails, your vocabulary lookup misses, and the word may fragment into odd subwords or hit `[UNK]`. Standard practice is to normalize the entire corpus to NFC before tokenizing — one canonical form in, one out.

There are also NFKC and NFKD ("compatibility" forms) which go further and collapse visually similar but semantically distinct characters — for example "ﬁ" (ligature) → "fi", "²" → "2", and full-width "Ａ" → "A". Those are lossier and should be used deliberately.

---

### Q3. Is text cleaning done according to the required output?

**Short answer:** Yes. There is no universally correct cleaning pipeline — cleaning is defined by the downstream task.

**Explanation:** Every cleaning operation is a trade: you remove variance to make the signal easier to find, and you risk removing the signal itself. The correct question is never "is this step good practice?" but "does this step remove information my task needs?"

| Operation | Helps | Hurts |
|---|---|---|
| Lowercasing | Topic modelling, search | NER ("Apple" vs "apple"), acronyms ("US" vs "us") |
| Stopword removal | TF-IDF, keyword extraction | Sentiment ("not good"), translation, any sequence model |
| Punctuation removal | Bag-of-words classification | Sentiment, sarcasm, question detection |
| Stemming/lemmatization | Search recall, sparse models | Modern Transformers (subwords handle it better) |
| Number removal | Some topic models | Financial, medical, legal text |

For LLM-era pipelines, the safe default is *minimal cleaning*: strip HTML tags, URLs, boilerplate, control characters, and duplicated whitespace, then normalize Unicode — and leave the actual language alone.

---

### Q4. What is a "lemmatization rule" in simple words?

**Short answer:** A rule that maps an inflected word back to its dictionary form, using the word's meaning and grammatical role.

**Explanation:** A lemma is the headword you would look up in a dictionary. A lemmatization rule encodes how to get there, and it consults both a dictionary and the part of speech.

- "running" (verb) → "run"
- "ran" (verb, irregular) → "run"
- "better" (adjective) → "good"
- "mice" (noun, irregular plural) → "mouse"
- "studies" (verb) → "study"; "studies" (noun) → "study"

Notice the last pair: the same surface form resolves through a different rule depending on the POS tag. That's the defining feature of lemmatization — it is *linguistically informed*, which makes it slower and dependency-heavy (it needs a lexicon and often a POS tagger), but it always returns a real word.

---

### Q5. What is stemming in simple terms?

**Short answer:** Chopping affixes off a word with pattern rules, without checking whether the result is a real word.

**Explanation:** Stemming is a crude, fast, purely mechanical truncation. The Porter stemmer, for example, applies an ordered list of suffix-stripping rules.

- "running" → "run"
- "studies" → "studi" ← not a word, and that's fine
- "happiness" → "happi"
- "universal", "university", "universe" → "univers" ← over-stemming; three distinct meanings collapsed
- "ran" → "ran" ← under-stemming; irregular form missed

**Stemming vs lemmatization:** stemming is a fast heuristic that may produce non-words; lemmatization is a dictionary lookup that always produces valid words but costs more. Stemming is still useful in search engines where recall matters more than precision. In modern deep-learning pipelines both are largely obsolete — subword tokenization already lets the model see "run" inside "running".

---

### Q6. What is stop word removal in simple terms?

**Short answer:** Deleting extremely common words that carry little discriminative information for the task at hand.

**Explanation:** In a bag-of-words model, "the" appears in essentially every document, so its presence tells you nothing about which document is which. It inflates the feature matrix and dilutes the terms that actually distinguish documents. Removing such words sharpens the signal-to-noise ratio and shrinks the vocabulary.

The crucial caveat: **stop words are only stop words relative to a task.** "not", "no", "never" appear on many default stop lists and are catastrophic to remove for sentiment analysis — "not good" becomes "good". Similarly, prepositions matter enormously for relation extraction ("drug *for* disease" vs "drug *from* disease").

---

### Q7. Examples of stop words?

**Short answer:** a, an, the, in, on, at, with, by, from.

**Explanation:** Typical English stop lists cover four families:

- **Articles:** a, an, the
- **Prepositions:** in, on, at, with, by, from, of, to, for
- **Pronouns:** i, you, he, she, it, they, we
- **Auxiliaries/copulas:** is, am, are, was, were, be, been, has, have, do, does

Lists are not standardized — NLTK ships ~179 English stop words, spaCy ~326, scikit-learn ~318, and they disagree with each other. A more principled alternative to a fixed list is to derive it from your own corpus: drop terms above a document-frequency threshold (say, present in >90% of documents), which adapts automatically to domain-specific filler.

---

### Q8. Does less text mean fewer tokens for a model like GPT?

**Short answer:** Generally yes, but not proportionally, and not reliably — token count depends on the tokenizer, not on word or character count.

**Explanation:** Subword tokenizers allocate tokens by *frequency*, not by length. Common words get one token; rare words get shredded.

- "the" → 1 token (3 characters)
- "unbelievable" → possibly 3 tokens ("un", "bel", "ievable")
- "antidisestablishmentarianism" → 8+ tokens
- "hello" → 1 token, but "hEllO" → 4 tokens (casing breaks the merge)

So removing nine short stop words might save nine tokens, while removing one rare technical term saves five. Worse, aggressive cleaning can *increase* token count: strip the space before a word and "the cat" (2 tokens) can become "thecat" (3+ tokens), because leading-space variants are themselves distinct tokens in BPE vocabularies.

A useful rough figure for English with GPT-family tokenizers: ~0.75 words per token, or ~4 characters per token. Non-English text is far less efficient — the same meaning in Hindi, Thai, or Tamil can cost 2–5× more tokens.

---

### Q9. In the text cleaning stage, what is typically stripped from raw text? (missed question)

**Short answer:** Option (c) — HTML tags, URLs, emails, and control characters.

**Explanation:** The logic is that these four categories are *non-linguistic artifacts of how the text was stored or transmitted*, not part of the language the model needs to learn. `<div class="row">`, `https://x.com/abc123`, `user@company.com`, and `\x00`/`\r` contribute no semantic content, appear in near-random patterns, and would otherwise consume vocabulary slots and token budget.

The other options fail because each removes real linguistic information:

- **Removing all digits** destroys quantities, dates, dosages, prices, versions — often the most important content in the sentence.
- **Removing all vowels** destroys the words themselves; "cat"/"cot"/"cut" all collapse to "ct".
- **Removing every occurrence of "the"** is arbitrary, single-word stopword removal that damages grammatical structure without any principled benefit.

---

## Part 2 — Tokenization

### Q10. What is tokenization?

**Short answer:** Splitting a text string into the smallest units the model actually reads — words, subwords, or characters.

**Explanation:** A neural network cannot consume a string. Tokenization is the bridge from human text to a finite, indexable set of symbols. It is a *segmentation* decision, and the granularity you choose determines the fundamental trade-off between vocabulary size and sequence length:

| Level | "unhappiness" becomes | Vocab size | Sequence length | Out-of-vocabulary problem |
|---|---|---|---|---|
| Character | u, n, h, a, p, p, i, n, e, s, s | ~100 | Very long | None |
| Subword | un, happi, ness | 30k–100k | Moderate | Effectively none |
| Word | unhappiness | 100k–1M+ | Short | Severe |

Subword tokenization is the modern default precisely because it sits at the sweet spot: a manageable vocabulary, reasonable sequence lengths, and no unknown-word cliff.

---

### Q11. Can different parts of the text use different tokenization methods, or must the whole text use one?

**Short answer:** One method for the entire text. The tokenizer and its vocabulary are fixed at training time and frozen for inference.

**Explanation:** This is non-negotiable, because token ID 4500 means whatever the model *learned* it means during training. If you switched algorithms mid-document, the same ID would suddenly refer to something else, and the model's learned embedding for that ID would be meaningless. Tokenizer and model weights are a matched pair; swapping one invalidates the other.

What *does* vary — and this is likely what prompted the question — is how the single fixed algorithm treats different words. A subword tokenizer is adaptive by construction:

- "the", "cat", "running" → kept whole (frequent in training data)
- "photosynthesis" → possibly "photo" + "synthesis"
- "Pranavjeet" → split into several fragments (rare)
- "🚀" → split into raw UTF-8 bytes

So the *rules* are uniform; the *outcome per word* differs by frequency. That is one algorithm behaving differently, not several algorithms in use.

---

### Q12. What is lemmatization?

**Short answer:** Converting a word to its dictionary base form (lemma) using its meaning and grammatical role.

**Explanation:** See Q4 for the rule mechanics. The key architectural point is *why* you'd bother: in sparse models, "run", "runs", "ran", and "running" are four unrelated columns in your feature matrix, splitting the statistical evidence for a single concept across four features. Lemmatization merges them, so a model with limited data sees one strong feature instead of four weak ones.

In dense contextual models this problem solves itself — the embeddings for "run" and "running" end up close in vector space because they occur in similar contexts, and the model gets to keep the tense information rather than having it thrown away. This is why lemmatization has quietly disappeared from Transformer pipelines.

---

### Q13. Can you briefly explain Byte Pair Encoding, WordPiece, and Unigram?

**Short answer:** Three ways to build a subword vocabulary — BPE merges by frequency, WordPiece merges by likelihood gain, Unigram prunes down from a huge candidate set.

**Explanation:**

**BPE (Byte-Pair Encoding)** — *bottom-up, greedy, frequency-driven.*
Start with a vocabulary of individual characters. Count every adjacent pair in the corpus, merge the most frequent pair into a new single token, then recount and repeat until you hit the target vocabulary size. If "e"+"s" is the most frequent pair, "es" becomes a token; later "es"+"t" might merge into "est". The stored artifact is an ordered list of merge rules, replayed in the same order at inference. Used by GPT-2/3/4, RoBERTa, Llama.

**WordPiece** — *bottom-up, but merges by likelihood, not raw count.*
Same skeleton as BPE, different selection criterion. Instead of merging the most frequent pair, it merges the pair that most increases the likelihood of the training corpus under a unigram language model — roughly, it maximizes `count(AB) / (count(A) × count(B))`. This favors pairs that occur together *more than chance would predict*, so it avoids merging a rare token with a very common one just because the combination is superficially frequent. Used by BERT, DistilBERT, ELECTRA.

**Unigram** — *top-down, probabilistic pruning.*
Runs in the opposite direction. Start with a deliberately over-complete vocabulary of every plausible substring, fit a unigram probability to each, then iteratively compute how much total corpus likelihood you'd lose by deleting each token, and delete the least useful 10–20%. Repeat until you reach the target size. Because it retains probabilities, it can score multiple valid segmentations of the same word and pick the best — or sample among them during training (subword regularization). Used by SentencePiece, T5, ALBERT, XLNet.

**Summary:** BPE and WordPiece build up; Unigram cuts down. BPE is deterministic and simple; WordPiece is statistically smarter about merges; Unigram is probabilistic and supports multiple segmentations.

---

### Q14. What does "subword-level" mean? A small example?

**Short answer:** Breaking a word into smaller meaningful pieces instead of storing the whole word as one token.

**Explanation:** "unhappiness" → `["un", "happi", "ness"]`.

This buys three concrete things:

1. **No out-of-vocabulary cliff.** Any word, including one invented yesterday, can be built from pieces. "cryptobro" → "crypto" + "bro".
2. **Morphological generalization.** The model sees the same "un" prefix in "unhappy", "unlikely", "unfair" and can learn that it signals negation, which transfers to words it has never seen.
3. **Bounded vocabulary.** English has hundreds of thousands of word forms but only ~30k–50k useful subword pieces are needed to cover them.

More examples: "tokenization" → `["token", "ization"]`; "playing" → `["play", "ing"]`; "ChatGPT" → `["Chat", "G", "PT"]`.

---

### Q15. Does subword tokenization work equally well for languages like Japanese or Cantonese?

**Short answer:** It works, but not equally well. Scriptless-boundary and logographic languages pay a real efficiency penalty.

**Explanation:** Subword algorithms were designed around a hidden assumption: that whitespace pre-segments text into word-like units, and the algorithm only has to decide how to split *within* those units. Japanese, Chinese, Thai, and Khmer have no such spaces, so the tokenizer must discover boundaries as well as sub-boundaries.

Three specific difficulties:

1. **No whitespace anchor.** 「東京都に住んでいます」 has no delimiters. Systems typically pre-segment with a morphological analyzer (MeCab, Juman) or use SentencePiece, which treats the raw stream — spaces included — as just another character and learns boundaries from data.
2. **Huge character inventory.** Chinese and Japanese kanji run to thousands of distinct characters, so the "start from characters" step of BPE begins with a much larger base than the ~100 characters of English.
3. **Token inflation.** Because these scripts are underrepresented in the training corpora used to *fit* most tokenizers, their characters often fall back to raw UTF-8 bytes — 3 bytes per character. The same sentence can cost 2–4× more tokens in Japanese than in English, which directly translates to higher API cost and less content fitting in a context window.

Japanese is also agglutinative, with long verb-suffix chains (食べさせられたくなかった), which fragment heavily. This is an active area of work; newer multilingual tokenizers deliberately rebalance their training mix to reduce the penalty.

---

### Q16. Does the tokenization pattern remain the same for all languages?

**Short answer:** No. Writing systems, word boundaries, and morphology differ enough that patterns diverge significantly.

**Explanation:** A single multilingual tokenizer can *process* all languages, but its behavior on each is very different, driven by how much of that language it saw when its vocabulary was fitted:

- **English** — dense coverage; ~4 characters per token.
- **German** — compounding ("Donaudampfschifffahrtsgesellschaft") means long words split into many pieces, though the pieces are usually meaningful morphemes.
- **Turkish/Finnish** — agglutinative; a single word can encode a whole English clause, so one word may become 6–8 tokens.
- **Arabic/Hebrew** — root-and-pattern morphology plus optional diacritics; consonantal roots don't align well with linear subword merging.
- **Hindi/Tamil/Thai** — often heavily byte-fragmented due to underrepresentation in tokenizer training data.

The practical consequence is *tokenizer fairness*: users writing in low-resource languages pay more tokens for the same message and get less usable context window, purely as an artifact of vocabulary construction.

---

## Part 3 — Token IDs and Vocabulary

### Q17. Where do token IDs come from? Where do we get them?

**Short answer:** From the tokenizer's vocabulary file, which maps every token string to a unique integer.

**Explanation:** The vocabulary is a plain dictionary shipped alongside the model — `vocab.json`, `tokenizer.json`, or `tokenizer.model` depending on the library. It looks like:

```json
{"the": 262, "cat": 3797, "sat": 3332, "Ġthe": 262, "<|endoftext|>": 50256}
```

At inference, tokenization is a segmentation step followed by a hash-map lookup. You never generate IDs yourself; you load the tokenizer that was distributed with the model and it returns them. In practice:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
tok.encode("The cat sat")   # → [464, 3797, 3332]
```

---

### Q18. How are these token IDs assigned dynamically?

**Short answer:** They aren't dynamic. The vocabulary is a fixed lookup table built once, at tokenizer training time, and frozen thereafter.

**Explanation:** The ID assignment is essentially *arbitrary* — it's an index, not a measurement. Ordering typically follows one of: merge order (the sequence in which BPE discovered each merge), corpus frequency (most common token gets the lowest index), or plain alphabetical order.

This is why numerically adjacent IDs mean nothing semantically. "cat" and "catapult" can land on consecutive indices simply because they sort next to each other, despite being unrelated. Conversely "cat" and "dog" can be thousands apart.

The critical property is *stability*: once built, the same string always yields the same ID, during training and at inference, forever. If it didn't, every embedding the model learned would point at the wrong row.

---

### Q19. By which process are tokens converted to token IDs?

**Short answer:** A dictionary lookup. At this stage the number carries no meaning at all — it's just an address.

**Explanation:** This is worth sitting with, because it's the point where most confusion starts. The pipeline is:

```
"The cat sat"  →  ["The", "cat", "sat"]  →  [464, 3797, 3332]  →  embedding vectors
     text              tokens                   IDs (addresses)        meaning
```

The ID is a *row number in the embedding matrix*, nothing more. ID 3797 says "go fetch row 3797 of the embedding table". All the semantic content lives in that row's 768 (or 1024, or 4096) learned floating-point numbers — not in the integer 3797. The integer is a pointer; the vector is the meaning.

---

### Q20. If neural networks ultimately need numbers, is tokenization just the first step in converting text to numbers?

**Short answer:** Exactly right. It's step one of three.

**Explanation:** The full conversion chain:

1. **Tokenization** — text → discrete tokens (`"The cat sat"` → `["The", "cat", "sat"]`)
2. **Numericalization** — tokens → integer IDs via vocabulary lookup (`[464, 3797, 3332]`)
3. **Embedding** — IDs → dense vectors via table lookup (`[3, 768]` matrix of floats)

Only after step 3 does the network have something it can multiply, add, and differentiate. Steps 1 and 2 are discrete and non-differentiable — you can't backpropagate through a dictionary lookup — which is precisely why the embedding matrix exists: it's the first *learnable* thing in the stack, and gradients flow into it and stop there.

---

### Q21. Do we have reserved tokens, like keywords in a programming language?

**Short answer:** Yes — special tokens with fixed roles and reserved IDs.

**Explanation:**

| Token | Purpose |
|---|---|
| `[CLS]` | Prepended to a sequence; its final hidden state is used as the whole-sequence representation for classification (BERT) |
| `[SEP]` | Separates two segments, e.g. question from passage |
| `[PAD]` | Fills short sequences to a uniform batch length; masked out of attention and loss |
| `[UNK]` | Stands in for anything outside the vocabulary (word-level tokenizers) |
| `[MASK]` | The token BERT hides during masked-language-model pretraining |
| `<\|endoftext\|>` | Marks a document boundary / generation stop (GPT family) |
| `<s>`, `</s>` | Beginning and end of sequence (Llama, RoBERTa) |

Chat models add more: role markers like `<\|im_start\|>`, `<\|im_end\|>` that structure a conversation into system/user/assistant turns. Those markers are the actual mechanism behind the "roles" you see in a chat API.

---

### Q22. Follow-up: what if a reserved token literally appears inside the user's text?

**Short answer:** The tokenizer refuses to recognize it as special and splits it into ordinary subwords.

**Explanation:** This is a deliberate security boundary. Special tokens are injected by the *framework*, not matched from user input. So if the literal string `<|endoftext|>` appears inside text being tokenized:

```
['<', '|', 'endo', 'ft', 'ext', '|', '>']
```

...and then the framework appends the genuine special token separately:

```
['<', '|', 'endo', 'ft', 'ext', '|', '>', <|endoftext|>]
```

The user's typed version and the real control token have different IDs entirely.

**Why this matters:** if user text *could* mint control tokens, anyone could type `<|im_end|><|im_start|>system` into a chat box and forge a system message. Tokenizers therefore expose a flag (`split_special_tokens` in HuggingFace) that controls this, and it should stay in the safe mode for anything user-supplied. This is one concrete layer of defense against prompt injection.

---

### Q23. Is the vocabulary dictionary part of the LLM?

**Short answer:** It belongs to the *tokenizer*, which ships with the model but is a separate artifact.

**Explanation:** When you download a model you get two coupled things:

- **Tokenizer files** — `tokenizer.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`. Small (a few MB), pure lookup data, no learned weights.
- **Model weights** — `model.safetensors`. Large (GB), the actual learned parameters, including the embedding matrix.

They are logically separate but functionally inseparable. The embedding matrix has exactly `vocab_size` rows, and row *i* is the vector for token ID *i*. Pair a model with the wrong tokenizer and every lookup lands on the wrong row — the model produces fluent-looking nonsense. This is a real and common bug when mixing checkpoints.

---

### Q24. Besides using list index, are there other ways real language models assign token IDs?

**Short answer:** It's always an integer index into a fixed table. What varies is how that table is *ordered*.

**Explanation:** Three strategies dominate, usually combined:

1. **Reserved/special-token offsets.** Low indices are carved out for control tokens before any real vocabulary is placed — e.g. BERT uses 0 = `[PAD]`, 100 = `[UNK]`, 101 = `[CLS]`, 102 = `[SEP]`, 103 = `[MASK]`, with ordinary wordpieces filling everything after. Many models also reserve a block of unused slots (`[unused0]`…`[unused99]`) so that domain-specific tokens can be added later without resizing the embedding matrix.

2. **Raw byte mapping (byte-fallback BPE).** Indices 0–255 map to the 256 possible UTF-8 byte values. This guarantees that *any* input — emoji, obscure scripts, corrupted bytes — is representable, eliminating `[UNK]` entirely. Learned subword merges then occupy indices 256 and up. This is what GPT-2 onward and Llama use.

3. **Frequency or merge-order indexing.** Remaining tokens are ordered by corpus frequency (most common first) or by the chronological order in which the merge algorithm discovered them. Frequency ordering has a practical benefit: it makes vocabulary truncation trivial, since dropping the tail removes the least useful tokens.

The ordering is an engineering convenience. Nothing about the *number* is semantic — the meaning lives entirely in the embedding row it points to.

---

### Q25. How does a spelling mistake affect encoding? Are there side effects on prediction?

**Short answer:** It changes the token sequence — either to `[UNK]` or to an unusual fragment chain — and yes, prediction quality degrades.

**Explanation:** The effect depends on the tokenizer:

**Word-level:** "recieve" isn't in the vocabulary, so it becomes `[UNK]`. Every misspelling in the document collapses to the same token, meaning all typos become indistinguishable from each other.

**Subword-level:** the word survives, but fragments oddly. "receive" might be a single clean token; "recieve" becomes `["rec", "ieve"]` or `["re", "ci", "eve"]` — a sequence the model has rarely seen in that arrangement.

**Downstream effects:**

1. **Loss of the pre-learned vector.** The model no longer retrieves the well-trained embedding for the intended word; it has to reconstruct meaning from fragment vectors that were learned in unrelated contexts.
2. **Increased reliance on context.** With the token-level signal degraded, the model leans harder on surrounding words. In a rich sentence ("please recieve the package") it usually recovers. In a sparse one ("recieve?") it may not.
3. **Longer sequences.** Typos cost more tokens, consuming context budget.
4. **Compounding in generation.** A garbled input token distribution can push the output distribution off in subtle ways.

Modern LLMs are surprisingly robust here, largely because their training data contained plenty of real-world misspellings — the model has genuinely seen "recieve" many times. Robustness is learned from noisy data, not architecturally guaranteed.

---

## Part 4 — From IDs to Vectors

### Q26. "The cat sat" is 3 words → 3 tokens → 3 IDs → input `[x, y, z]`. What if there were 4 words? What does the input matrix look like?

**Short answer:** It's a 2-D matrix of shape `[sequence_length, embedding_dimension]` — 4 words simply makes it one row taller, not a padded second row.

**Explanation:** Each token becomes a *vector*, not a scalar. With an embedding dimension of 200:

- "The cat sat" → matrix of shape `[3, 200]`
- "The cat sat down" → matrix of shape `[4, 200]`

```
              ← 200 dimensions →
"The"    [ 0.21, -0.44, 0.83, ... ]
"cat"    [-0.11,  0.92, 0.05, ... ]
"sat"    [ 0.67,  0.13, -0.29, ... ]
"down"   [ 0.02, -0.31, 0.76, ... ]
```

**Where padding actually comes in:** padding is needed for *batching*, not for single sequences. If you process 32 sentences at once, they must share a length, so shorter ones are filled with `[PAD]` tokens up to the longest — giving a 3-D tensor `[batch_size, max_seq_len, embed_dim]`, e.g. `[32, 128, 200]`. An attention mask then tells the model to ignore the padded positions so they contribute nothing to the output or the loss.

---

### Q27. What does "embedding dimension" mean in the `[3, 200]` example?

**Short answer:** The number of numerical values used to represent each individual token.

**Explanation:** Embedding dimension = the length of one token's vector. With 3 words and dimension 200, each word is described by 200 numbers, giving a 3 × 200 matrix.

Conceptually, each dimension is a learned coordinate axis in a semantic space. No single axis has a clean human label — you can't point at dimension 47 and say "this one means plurality" — but *directions* through the space reliably encode relationships. The canonical demonstration:

```
vector("king") − vector("man") + vector("woman") ≈ vector("queen")
```

That works because gender is encoded as a consistent direction, learned purely from co-occurrence statistics.

Typical values: 50–300 for classic Word2Vec/GloVe; 768 for BERT-base; 1024 for BERT-large; 4096+ for large LLMs.

---

### Q28. For a large vocabulary of, say, 50k words, must the vector length be 50k?

**Short answer:** Only with one-hot encoding. Dense embeddings decouple vector length from vocabulary size entirely.

**Explanation:** With one-hot, yes — every word is a 50,000-length vector of zeros with a single 1. That's the definition, and it's why the representation is called *sparse*: 49,999 wasted slots per word, and the storage grows linearly with vocabulary.

Dense embeddings break that coupling. Vocabulary size determines the *number of rows* in the embedding matrix; embedding dimension determines the *width*. They're independent hyperparameters:

- One-hot: 50,000 words × 50,000 dims = 2.5 billion values, almost all zero
- Dense: 50,000 words × 768 dims = 38.4 million values, all meaningful

That's a ~65× reduction, and the dense version encodes similarity that the one-hot version structurally cannot. Common embedding dimensions: 768, 1024, 2048, 4096, 8192.

---

### Q29. Is the curse of dimensionality in one-hot encoding a computational problem, a statistical learning problem, or both?

**Short answer:** Both — but the statistical problem is the fatal one.

**Explanation:**

**Statistical (the real killer):** In one-hot space, every pair of distinct words is *equidistant and orthogonal*. The dot product of any two different one-hot vectors is exactly zero. So "cat"·"dog" = 0 and "cat"·"bulldozer" = 0 — the geometry says these pairs are equally unrelated. The representation contains *no similarity information whatsoever*, which means the model cannot generalize from "cat" to "dog"; it must learn every word independently, from scratch, requiring vastly more data. Worse, as dimensionality grows, data becomes exponentially sparse and any given region of the space contains almost no examples.

**Computational (annoying but tractable):** 50k-dimensional vectors mean enormous weight matrices, high memory footprint, and multiplications that are almost entirely by zero. Sparse matrix libraries mitigate a lot of this — you can represent a one-hot vector as a single index, and multiplying by the weight matrix is just a row lookup.

So: you can *engineer around* the computational cost. You cannot engineer around the fact that the representation encodes no relationships. Dense embeddings fix both, but they exist to fix the second.

---

### Q30. So should we use one-hot encoding?

**Short answer:** Not for text in modern pipelines. Dense contextual embeddings have replaced it.

**Explanation:** The historical progression:

1. **One-hot** — no similarity, no generalization
2. **Static dense embeddings** (Word2Vec, GloVe) — similarity captured, but one fixed vector per word regardless of context
3. **Contextual embeddings** (BERT, GPT) — the vector for a word is computed from its actual sentence, so "bank" in "river bank" and "bank account" get different vectors

Step 3 is what the attention mechanism enabled, and it's what production systems use today.

---

### Q31. Is there any remaining application for one-hot if dense is better?

**Short answer:** Yes — when categories are few and genuinely have no natural similarity structure.

**Explanation:** One-hot is the correct choice, not a fallback, when:

- **The category set is small.** 12 months, 7 weekdays, 5 product tiers, 50 US states. A 12-dimensional one-hot is cheap and perfectly readable.
- **Categories are genuinely unordered and unrelated.** Blood type, country code, POS tag. Here, "no similarity encoded" is an accurate reflection of reality, not a loss.
- **You want interpretability.** In a logistic regression or tree model, one-hot columns map directly to feature importances you can explain to a stakeholder. Dense embeddings do not.
- **As training targets.** Classification labels are one-hot vectors compared against a softmax output — that's how cross-entropy loss is defined.
- **Tabular ML in general.** Scikit-learn, XGBoost, and LightGBM pipelines still one-hot categorical features routinely, and it works well.

The rule: use one-hot when the vocabulary is small and flat; use embeddings when it's large or has internal structure worth learning.

---

### Q32. What is the need for the dot product between "cat" and "dog"?

**Short answer:** To measure how similar their vector representations are.

**Explanation:** The whole premise of embeddings is that *semantic similarity should equal geometric proximity*. The dot product is the arithmetic that tests whether that holds:

```
cat · dog     = 0.87   (high — both pets, similar contexts)
cat · car     = 0.12   (low — unrelated)
cat · kitten  = 0.94   (very high — near-synonyms)
```

If two words appear in similar contexts, a well-trained embedding places them in similar directions, and the dot product comes out large. If the representation *couldn't* produce that pattern — as with one-hot, where every such product is 0 — you'd know the representation had failed.

Downstream, this single operation powers semantic search, recommendation, clustering, RAG retrieval, and duplicate detection. It's also the mathematical core of attention itself: query·key scores are dot products.

---

### Q33. Why can't we just use token IDs as model inputs?

**Short answer:** Because IDs are arbitrary addresses, and feeding them in directly would make the model interpret meaningless numeric distance as meaningful similarity.

**Explanation:** Suppose "cat" = 4500 and "dog" = 8500. A neural network doing arithmetic on those inputs sees two numbers 4,000 apart and, having no other information, treats them as very different. Meanwhile "building" might be 4501 — one away from "cat" — and the network reads them as near-identical.

The IDs came from merge order or alphabetical sorting. They encode *nothing* about meaning. But a network can only do arithmetic, and arithmetic on an ID implies a magnitude relationship that doesn't exist.

The embedding layer solves this precisely: the ID is used only as a *lookup index*, never as a quantity. Row 4500 is fetched, and that row contains 768 learned values that *do* encode meaning — placed there by gradient descent over billions of tokens.

---

### Q34. But couldn't we design IDs by rule — say, all animals get 8XXXX?

**Short answer:** No. It fails on scale, on multi-membership, and on the fact that meaning is contextual rather than categorical.

**Explanation:** Say animals get 8xxxx and man-made objects 2xxxx. Then:

- cat (80001) and car (20001) are 60,000 apart
- cat (80001) and cow (80002) are 1 apart

The model reads that as "cat is 60,000× more different from a car than from a cow" — a completely arbitrary ratio you invented, now baked into the geometry as if it were fact.

Deeper problems:

1. **Multi-membership.** Is "horse" an animal or a vehicle? Both, contextually. Is "leather" animal or man-made? Both. Is "mouse" a rodent or a peripheral? One numeric slot cannot hold two answers.
2. **Similarity isn't one-dimensional.** "Cat" is similar to "dog" (both pets), to "tiger" (both felines), to "kitten" (life stage), to "meow" (associated sound). These are four *different* axes of similarity. A single number has one axis.
3. **It doesn't scale.** Someone must hand-classify 50,000 tokens — including subword fragments like "ing", "un", "##tion" which belong to no semantic category at all.
4. **It's already been done, better, automatically.** A 768-dimensional embedding has 768 axes, each learned from data. It discovers that "horse" sits near both "animal" and "transport" regions without anyone deciding in advance.

---

### Q35. What is the comparable characteristic in dense representation?

**Short answer:** Distance is meaningful. Cosine similarity between two dense vectors is a usable measure of semantic relatedness.

**Explanation:** This is the single property that makes dense representations useful. In one-hot space the similarity between any two distinct words is *exactly zero*, always — the space has no gradient of meaning. In dense space, similarity is continuous, and words with related meanings sit closer together.

Concretely, this enables:

- **Ranking:** "which of these 10,000 documents is most like my query?"
- **Analogy:** king − man + woman ≈ queen
- **Clustering:** related terms group naturally without labels
- **Transfer:** a model that learns something about "dog" partially generalizes to "puppy", because their vectors overlap

Cosine similarity is the standard measure because it compares *direction* while ignoring magnitude — and in embedding space, direction is where the meaning lives; magnitude tends to track word frequency, which you usually want to factor out.

---

### Q36. What are the numeric values in an embedding? Are they cosine similarity values? Do embeddings always carry semantic meaning?

**Short answer:** No, they're learned parameters, not similarity scores. And no — they only carry semantic meaning after training.

**Explanation:**

**What the numbers are:** each value is a learned weight, adjusted by gradient descent. They start as small random numbers and are updated the same way any other network parameter is. Cosine similarity is something you *compute from* two embeddings afterward — it's an output of the representation, not its content.

**Do they always carry meaning?** No, and the distinction matters:

- **At random initialization** — no meaning at all. Pure noise.
- **After Word2Vec/GloVe training** — static semantic meaning. One fixed vector per word, capturing its average sense across the corpus.
- **In a Transformer** — contextual meaning. The vector for "bank" is computed from the surrounding sentence, so it differs between "river bank" and "bank account".
- **Trained on a non-semantic objective** — the vectors encode whatever the objective rewarded. Embeddings trained to predict user clicks encode behavioral similarity, not meaning.

The general principle: **an embedding encodes whatever the training objective forced it to encode.** Semantic meaning emerges from semantic objectives (predicting neighbouring words), not from the embedding mechanism itself.

---

### Q37. How do we choose the value of the embedding dimension?

**Short answer:** It's a hyperparameter — chosen empirically, guided by vocabulary size, data volume, and compute budget.

**Explanation:** There's no closed-form answer, but there are reliable heuristics and conventions.

**Practical rules of thumb:**
- A common starting point for categorical embeddings: `dim ≈ min(50, (vocab_size + 1) / 2)`, or `dim ≈ vocab_size^0.25`
- Word2Vec/GloVe: 100–300
- BERT-base: 768; BERT-large: 1024
- Large LLMs: 4096–16384

**The trade-off:**
- **Too small** — underfitting. Not enough axes to separate distinct concepts; unrelated words get crowded into the same region.
- **Too large** — overfitting and waste. More parameters than the data can constrain, plus quadratic growth in downstream compute.

**Constraints that matter in practice:**
- Must be divisible by the number of attention heads (768 = 12 heads × 64 dims each)
- Powers of 2 or multiples of 64 for GPU efficiency
- Scaling laws suggest width should grow roughly in proportion to depth and dataset size

In the end it's chosen by validation performance. Start with an established value for your model family; only tune it if you have a specific reason.

---

### Q38. So the embedding vector size is a kind of hyperparameter?

**Short answer:** Yes. In Word2Vec specifically, the hidden-layer dimension *is* the embedding dimension.

**Explanation:** This is a nice structural observation. In CBOW/Skip-gram, the network is:

```
input (V-dim one-hot) → hidden layer (D-dim) → output (V-dim softmax)
```

The input-to-hidden weight matrix has shape `[V, D]`. After training, you discard the output layer and keep that matrix — **it is the embedding table**, with one D-dimensional row per vocabulary word. So the hidden layer width you chose as a hyperparameter is, by construction, the embedding dimension you end up with.

Other hyperparameters in the same family: context window size, negative sampling count, learning rate, minimum word frequency.

---

## Part 5 — Word2Vec: CBOW and Skip-Gram

### Q39. Can you explain CBOW once again?

**Short answer:** CBOW trains a shallow network to predict a missing centre word from its surrounding context words, and the embeddings are the by-product.

**Explanation:** Step by step, for the sentence "the cat sat on the mat" with a window of ±2 and "sat" as the target:

1. **Input.** Take the context words — "the", "cat", "on", "the" — as one-hot vectors.
2. **Lookup.** Multiply each by the input weight matrix `[V, D]`. Because the input is one-hot, this is mathematically just a row lookup, so each context word yields its D-dimensional vector.
3. **Average.** Average those context vectors into a single hidden-layer representation. This is the "bag" in Continuous Bag of Words — order is discarded.
4. **Score.** A softmax over the output matrix scores every word in the vocabulary as a candidate centre word.
5. **Backpropagate.** Compare against the true answer ("sat"), compute cross-entropy loss, and push gradients back to adjust both matrices.
6. **Repeat** across every position in the corpus.
7. **Discard and keep.** Throw away the output layer. Keep the input weight matrix — that's your embedding table.

**The key insight:** the prediction task is a *pretext*. Nobody wants a mediocre word-guesser. But to guess well, the network is forced to place words that appear in similar contexts near each other in the hidden space — and that arrangement is the thing you actually wanted.

**Why the averaging matters:** it smooths over the local neighbourhood, which makes training fast and gives strong representations for frequent words. The cost is that rare words get drowned out by their more common neighbours in the average.

---

### Q40. In CBOW we're still taking one-hot vectors for context words — but we said one-hot has many limitations?

**Short answer:** Conceptually yes, but neither limitation applies here, for two reasons.

**Explanation:**

**First — the one-hot vector is never actually constructed.** Multiplying a one-hot vector by the embedding matrix is mathematically identical to selecting one row of that matrix:

```
[0, 0, 1, 0, 0] × W  ≡  W[2]
```

Every real implementation does the lookup and skips the multiplication entirely. The one-hot exists only in the mathematical description, not in memory. So the computational objection — huge sparse vectors, wasted multiplications — evaporates.

**Second — the one-hot is input, not output.** The statistical limitation of one-hot is that it encodes no similarity. But CBOW's *entire purpose* is to produce a representation that does. The one-hot is merely the addressing mechanism for a "fake" prediction task; once training completes, the output layer is discarded and the learned dense matrix replaces the one-hot representation completely.

Put simply: one-hot is the scaffolding used to build the dense embedding, then removed.

---

### Q41. When do we use CBOW vs Skip-Gram?

**Short answer:** CBOW for speed and frequent words on large corpora; Skip-Gram for rare words and smaller corpora.

**Explanation:**

| | CBOW | Skip-Gram |
|---|---|---|
| Direction | Context → centre word | Centre word → context |
| Training passes per window | 1 (context averaged) | One per context word |
| Speed | Faster | Slower (several× ) |
| Frequent words | Strong | Strong |
| Rare words | Weaker — averaged away | Stronger — gets dedicated updates |
| Small datasets | Struggles | Better |

The mechanism behind the difference: CBOW averages the context into one prediction, so a rare word contributes a fraction of one training signal. Skip-Gram makes a *separate* prediction for each context word, so a rare word generates several distinct gradient updates every time it appears. More signal per occurrence — which is exactly what rare words need.

**Default choice:** Skip-Gram with negative sampling (SGNS), unless training time is the binding constraint. It's the configuration that has held up best empirically.

---

### Q42. If a word wasn't seen during training but has a similar meaning to a vocabulary word, can CBOW predict it?

**Short answer:** No. CBOW has a fixed vocabulary; there is no embedding row and no softmax slot for an unseen word, so it can never be produced.

**Explanation:** The architecture makes this impossible, not merely difficult. The output layer is a softmax over exactly V slots — the words seen during training. A word that wasn't there has no slot to receive probability mass.

And "similar meaning doesn't help" is the crucial part: **in Word2Vec, meaning comes only from observed co-occurrence.** The model has no dictionary, no morphology, no notion of synonymy beyond what it counted in the data. An unseen word has no counts, therefore no meaning, therefore no vector. All out-of-vocabulary words collapse to the same `UNK` representation — meaning "cryptocurrency" and "zygomorphic" become identical if neither appeared in training.

**How later methods fixed this:**
- **FastText** represents a word as the sum of its character n-grams, so "unhappiness" can be assembled from "un", "hap", "ppi", "ness" even if the whole word was never seen.
- **Subword tokenizers** (BPE, WordPiece) do the same at the tokenization stage — which is why modern LLMs have no OOV problem at all.

---

## Part 6 — Softmax

### Q43. What is a softmax probability? I understand probability, but what is the softmax function?

**Short answer:** A function that normalizes a vector of arbitrary real numbers into a probability distribution summing to 1.

**Explanation:** A neural network's final layer produces raw scores (logits) — any real numbers, positive or negative, on no fixed scale. Softmax converts them into probabilities:

```
Softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
```

Worked example with logits `[2.0, 1.0, 0.1]`:

1. Exponentiate: `e^2.0 = 7.39`, `e^1.0 = 2.72`, `e^0.1 = 1.11`
2. Sum: `7.39 + 2.72 + 1.11 = 11.22`
3. Divide each by the sum: `[0.659, 0.242, 0.099]` — which sums to 1.0

Two properties earn softmax its name:
- **Exponentiation makes everything positive**, which is required for probabilities and handles negative logits cleanly.
- **It amplifies differences.** A logit gap of 1.0 becomes a probability ratio of ~2.7×. It's a *soft* argmax — it picks a winner while retaining a graded distribution over the rest, which keeps the function differentiable.

---

### Q44. What kind of function is softmax?

**Short answer:** A vector-to-vector normalization function that maps arbitrary reals to a probability distribution.

**Explanation:** Formally, softmax maps ℝⁿ → ℝⁿ with two guarantees: every output is in (0, 1), and all outputs sum to exactly 1. In CBOW, it scores every word in the vocabulary for the likelihood of being the target centre word.

```
Softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
```

A practical footnote: implementations subtract the maximum logit before exponentiating (`e^(x_i − max(x))`), because `e^1000` overflows floating point. Subtracting a constant leaves the result mathematically unchanged but keeps it numerically stable — this is standard in every library.

---

### Q45. So softmax is an activation function — is my understanding correct?

**Short answer:** Yes, with an important distinction: it's a *vector-valued* activation applied to the whole output layer, not element-wise like ReLU or sigmoid.

**Explanation:** Softmax is classified as an activation function, and it appears in that slot in every framework. But it behaves differently from the others:

- **ReLU, sigmoid, tanh** — element-wise. Each output depends only on its own input. Change one logit, one output changes.
- **Softmax** — vector-wise. Every output depends on *all* inputs, because they share the denominator. Change one logit, every output shifts.

That coupling is the whole point: it's what makes the outputs compete for a fixed budget of probability. It also constrains where softmax can be used — it belongs on the final layer of a multi-class classifier, never in a hidden layer, because the coupling would destroy the independent feature detection you want from hidden units.

---

### Q46. Is the softmax function the same as the sigmoid function?

**Short answer:** No, though they're closely related — softmax is the multi-class generalization of sigmoid.

**Explanation:**

| | Sigmoid | Softmax |
|---|---|---|
| Formula | `1 / (1 + e^(−x))` | `e^(x_i) / Σ e^(x_j)` |
| Input | One scalar | A vector |
| Outputs | One value in (0,1) | n values summing to 1 |
| Independent? | Yes — outputs unrelated | No — outputs compete |
| Use case | Binary or multi-*label* | Multi-*class* (exactly one answer) |

**The relationship:** apply softmax to a 2-element vector `[x, 0]` and you get exactly `sigmoid(x)` for the first element. Softmax with n=2 *is* sigmoid.

**When the choice matters:** classifying an image as one of {cat, dog, bird} — exactly one is true — needs softmax. Tagging an article with any of {sports, politics, finance} — several can be true simultaneously — needs independent sigmoids, because softmax would force them to compete for a total of 1.0.

---

### Q47. How does the addition happen in softmax?

**Short answer:** The addition is the denominator — the sum of all exponentiated logits, which each numerator is then divided by.

**Explanation:** Tracing it precisely, for logits `[2.0, 1.0, 0.1]`:

**Step 1 — exponentiate each element independently:**
```
e^2.0 = 7.389
e^1.0 = 2.718
e^0.1 = 1.105
```

**Step 2 — add them all up (this is the addition):**
```
Σ = 7.389 + 2.718 + 1.105 = 11.212
```

**Step 3 — divide each by that sum:**
```
7.389 / 11.212 = 0.659
2.718 / 11.212 = 0.242
1.105 / 11.212 = 0.099
```

**Step 4 — verify:** `0.659 + 0.242 + 0.099 = 1.000` ✓

The sum is what guarantees the outputs total 1. It's also what makes softmax vector-valued: since every output shares that denominator, changing any single logit changes every probability.

---

## Part 7 — GloVe and Comparisons

### Q48. What does it mean that Word2Vec predicts locally and GloVe globally?

**Short answer:** Word2Vec learns from one small context window at a time; GloVe builds a corpus-wide co-occurrence matrix first, then factorizes it.

**Explanation:**

**Word2Vec — predicts locally.** It slides a window across the text and makes an incremental prediction at each position. In "the cat sat on", with a ±2 window, it tries to predict "cat" using only its immediate neighbours "the", "sat", "on" — then updates the vectors slightly and moves on. It never holds a global view; global structure emerges only as an accumulated side effect of millions of local updates.

**GloVe — counts globally.** Before any training begins, it builds a full V × V co-occurrence matrix by tallying how often each word appears near each other word across the *entire* corpus. Then it factorizes that matrix so that vector dot products reproduce the log co-occurrence counts.

**The worked example from the slides:** compare how often "solid" appears near "ice" versus near "steam".

| Word | P(w \| ice) | P(w \| steam) | Ratio |
|---|---|---|---|
| solid | high | low | ≫ 1 → relates to ice |
| gas | low | high | ≪ 1 → relates to steam |
| water | high | high | ≈ 1 → relates to both |
| fashion | low | low | ≈ 1 → relates to neither |

The *ratio* of co-occurrence probabilities is what carries meaning — it cleanly separates "relevant to ice" from "relevant to steam" while cancelling out words relevant to both or neither. GloVe's objective is built directly on that ratio, which is only computable with global counts.

**In practice:** the two methods perform comparably on most benchmarks. GloVe trains faster once the matrix is built (it's parallelizable and the counts are computed once); Word2Vec streams and needs no large matrix in memory.

---

### Q49. What is "fashion" in the example?

**Short answer:** The control row — a word deliberately unrelated to the entire ice/steam topic.

**Explanation:** It's there to demonstrate what a *non-signal* looks like. The table has four cases and needs all four to make the argument:

- "solid" → strongly ice-associated (ratio ≫ 1)
- "gas" → strongly steam-associated (ratio ≪ 1)
- "water" → associated with both (ratio ≈ 1)
- "fashion" → associated with neither (ratio ≈ 1)

The instructive part is that "water" and "fashion" produce the *same* ratio (≈1) for opposite reasons — one is relevant to both, one to neither. This shows that the ratio alone distinguishes ice-words from steam-words but cannot distinguish "relevant to both" from "irrelevant". That's why GloVe's actual loss function also weights by raw co-occurrence count, so high-frequency pairs like (water, ice) carry more influence than near-zero pairs like (fashion, ice).

---

### Q50. Is GloVe similar to self-attention?

**Short answer:** No. They both use dot products, but they solve different problems — GloVe produces static vectors, self-attention produces contextual ones.

**Explanation:**

| | GloVe | Self-attention |
|---|---|---|
| When it runs | Once, at training | Every forward pass, per input |
| Output | One fixed vector per word | A different vector per occurrence |
| Input to the operation | Global corpus counts | The current sequence |
| "bank" in two sentences | Identical vector | Different vectors |

GloVe gives "bank" a single vector that is a blurred average of *all* its senses across the corpus. Self-attention computes a fresh representation for each token by weighting every other token in the current sentence — so in "the river bank flooded", the vector for "bank" absorbs information from "river", while in "the bank approved the loan" it absorbs from "loan".

The superficial similarity is that both compute dot products between vectors. But GloVe's dot products approximate corpus-wide counts, while attention's dot products score relevance *within a specific sequence at inference time*. Self-attention captures context; GloVe cannot.

---

### Q51. Why the dot product instead of cosine similarity, which is more accurate semantically?

**Short answer:** They serve different stages. The dot product is GloVe's *training* objective; cosine similarity is the *evaluation* measure used afterward.

**Explanation:** Cosine similarity is indeed the correct measure for comparing dense vectors semantically, and it's what you use in production for search and retrieval.

But GloVe's training objective is specifically to make dot products reproduce log co-occurrence counts:

```
w_i · w_j + b_i + b_j ≈ log(X_ij)
```

where `X_ij` is how often words i and j co-occur. That's a *magnitude-sensitive* target: a pair that co-occurs 10,000 times must produce a larger dot product than a pair that co-occurs 10 times. Cosine similarity normalizes magnitude away, so it structurally cannot represent frequency and would make the objective unlearnable.

There's also a computational reason: the dot product is a single fused operation, while cosine requires computing two norms and dividing — meaningful overhead across billions of training pairs.

**The clean summary:** dot product is an internal training mechanism for fitting frequencies; cosine similarity is the post-training measure of semantic distance. Note that if you L2-normalize your vectors after training, the two become identical anyway — which is exactly what most vector databases do.

---

### Q52. What is Byte Pair Encoding? Can it replace Word2Vec?

**Short answer:** No — BPE is a tokenizer, Word2Vec is an embedding learner. They occupy different stages of the same pipeline.

**Explanation:** They aren't competitors; they're sequential.

```
raw text
   ↓  BPE            (split into subword tokens)
tokens
   ↓  vocabulary lookup   (tokens → IDs)
IDs
   ↓  Word2Vec / embedding layer   (IDs → dense vectors)
vectors → model
```

BPE answers *"what are the units?"* Word2Vec answers *"what does each unit mean?"* You need both. You could run BPE and then train Word2Vec on the resulting subword tokens — that's a perfectly coherent pipeline, and roughly what FastText approximates.

**What actually replaced Word2Vec** is not BPE but *contextual embeddings*. Modern Transformers use BPE for tokenization and then learn their embedding matrix jointly with the rest of the network — end to end, as part of language model pretraining — rather than as a separate Word2Vec pre-training step. The embedding table is still there; it's just trained differently, and refined by attention layers into context-dependent representations.

---

## Part 8 — Real-World LLMs

### Q53. Are Transformer architecture efficiencies different across Claude, OpenAI, and Gemini, or does everyone follow the same approach?

**Short answer:** All three share the same Transformer blueprint, but diverge significantly in efficiency-related design choices.

**Explanation:** The foundational components — self-attention, feed-forward blocks, residual connections, layer normalization — are common to all. The divergence is in the engineering around them, and it's substantial:

- **Mixture-of-Experts vs dense scaling.** MoE routes each token to a small subset of specialized expert sub-networks, so total parameter count can be enormous while per-token compute stays modest. Dense models activate every parameter for every token — simpler and often better per-parameter, but more expensive to scale.
- **Attention and KV-cache variants.** Multi-query and grouped-query attention share key/value projections across heads to shrink the memory needed for the cache during generation. Sliding-window, sparse, and linear attention variants trade full context access for lower cost.
- **Positional encoding.** RoPE, ALiBi, and learned positions differ in how well they extrapolate to sequences longer than those seen in training.
- **Multimodal fusion.** Early fusion (native multimodal training) versus late fusion (a separate vision encoder projected into the text space) leads to very different capability profiles.

This is genuinely a topic for the Transformers lecture later in the course — the terms above only click once attention itself is covered. The one-line takeaway: *same blueprint, different efficiency engineering*, and the differences are large enough to matter for cost, latency, and long-context behavior.

---

### Q54. Since theories build incrementally, can we assume commercial LLMs use all of these and choose a method at runtime based on context and task?

**Short answer:** They use many of these techniques together, but the architecture and tokenizer are fixed at training time. Nothing is selected at runtime.

**Explanation:** This is an important correction to a natural intuition. A deployed model does *not* decide "this input looks like sentiment analysis, so I'll switch to WordPiece and use static embeddings." Those choices were locked in before the model ever saw a user.

**Fixed at training time (immutable at inference):**
- Tokenizer algorithm and vocabulary
- Embedding dimension and matrix
- Number of layers, attention heads, hidden sizes
- Positional encoding scheme
- All learned weights

**Variable at inference:**
- The input itself, and therefore the attention patterns computed over it
- Sampling parameters (temperature, top-p)
- System prompt, tools, retrieved context
- In MoE models, *which experts a token is routed to* — this is genuinely dynamic per-token routing, but it's routing within a fixed architecture, not a choice among algorithms

So the model's *behavior* adapts enormously to input and task; its *machinery* does not change. What looks like adaptability is a single fixed architecture that was trained to be general enough to handle many tasks — plus, increasingly, orchestration *outside* the model (routing between different model sizes, tool use, retrieval) that a user might reasonably mistake for the model reconfiguring itself.

---

### Q55. How long does the training take?

**Short answer:** This needs context to answer usefully — training what, on what data, on what hardware?

**Explanation:** The question was flagged during the session as needing more context, and rightly so, because the range spans seven orders of magnitude:

| What | Data | Hardware | Rough time |
|---|---|---|---|
| Word2Vec (CBOW) | 100M words | 1 CPU, multithreaded | Minutes to a few hours |
| GloVe | 6B tokens | Multi-core CPU | Hours |
| BERT-base from scratch | 3.3B words | 16 TPUs | ~4 days |
| Fine-tuning BERT | 10k examples | 1 GPU | Minutes to hours |
| Frontier LLM pretraining | Trillions of tokens | Thousands of GPUs | Weeks to months |

For the CBOW/Skip-gram models discussed in this session, the honest answer is: fast. Word2Vec was designed in 2013 to run on commodity CPU hardware, and training on a corpus of a few hundred million words finishes in under an hour. That efficiency was a large part of why it was so influential.

---

## Quick Reference

**The pipeline, end to end:**
```
raw text
  → cleaning & Unicode normalization    (task-dependent, minimal for LLMs)
  → tokenization                        (BPE / WordPiece / Unigram)
  → token IDs                           (fixed vocabulary lookup — arbitrary integers)
  → embeddings                          (learned dense vectors — this is where meaning lives)
  → model                               (Transformer layers → contextual representations)
```

**The three things most worth remembering:**

1. **Token IDs are addresses, not values.** The integer means nothing; the embedding row it points to means everything.
2. **One-hot's fatal flaw is statistical, not computational.** Every distinct word pair is exactly equidistant, so no similarity can be represented at all.
3. **Cleaning decisions are task decisions.** There is no universally correct pipeline, and for modern Transformers the safe default is to clean as little as possible.
