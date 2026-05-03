# Evaluation Metrics Taxonomy

## Overall Configuration and Models

**Embedding models used:**
- IE and Coverage (Approach A): `Qwen/Qwen3-Embedding-8B`
- Coverage (Approach B): bi-encoder (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) + cross-encoder (`cross-encoder/nli-deberta-v3-base`)

**Judge models used:**
- Citation: `gpt-4o-mini` (or configurable)
- Coverage (Approach C): `gpt-5.4-mini` (or configurable)
- Mandatory questions: `gpt-4o-mini` (or configurable)

**Default thresholds:**
- IE similarity threshold: 0.55
- Coverage similarity threshold (Approach A): 0.75
- Coverage entailment threshold (Approach B): 0.5
- SUSWIR coverage sub-threshold for source sentences: 0.6
- SUSWIR redundancy sub-threshold: 0.5
- Mandatory question compliance threshold: 0.75

---

This project has four main evaluation points:
- Information Extraction (IE)
- Coverage
- Citation faithfulness/support
- Mandatory questions

Approach mapping by evaluation point:
- IE: embeddings + similarity, and SUSWIR
- Coverage: embedding + similarity, bi-encoder + similarity (with cross-encoder entailment scoring), and LLM as judge
- Citation: LLM as judge only
- Mandatory questions: LLM as judge only

## 1) Information Extraction Metrics

### Approach A: Embeddings + similarity

**Per-document calculation:**

$$
\text{Precision} = \frac{\text{matched\_extracted}}{\text{extracted\_count}}
$$

$$
\text{Recall} = \frac{\text{matched\_gt}}{\text{gt\_count}}
$$

$$
\text{F1} = \begin{cases}
\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} & \text{if } \text{Precision} + \text{Recall} > 0 \\
0 & \text{otherwise}
\end{cases}
$$

Where:
- $\text{matched\_extracted}$ = count of extracted statements where $\max_j \text{similarity}(e_i, gt_j) \geq \text{threshold}$
- $\text{matched\_gt}$ = count of GT facts where $\max_i \text{similarity}(e_i, gt_j) \geq \text{threshold}$
- Similarity is computed as cosine similarity: $\text{similarity}(a, b) = \mathbf{emb}(a) \cdot \mathbf{emb}(b)$ (L2-normalized)

**Aggregated IE scores (method/model level):**

$$
\text{mean\_precision} = \frac{1}{D} \sum_{d=1}^{D} \text{Precision}_d
$$

$$
\text{mean\_recall} = \frac{1}{D} \sum_{d=1}^{D} \text{Recall}_d
$$

$$
\text{mean\_f1} = \frac{1}{D} \sum_{d=1}^{D} \text{F1}_d
$$

Where $D$ is the number of documents evaluated.

Note:
- These aggregated IE scores are defined in the codebase as a helper summary function.
- The current main evaluation run stores per-document outputs by default.
- Embedding model used: `Qwen/Qwen3-Embedding-8B`
- Default threshold: 0.55

### Approach B: SUSWIR

**Component definitions:**

$$
\text{SSF (Semantic Similarity Factor)} = \frac{1}{n_e} \sum_{i=1}^{n_e} \mathbf{emb}(e_i) \cdot \mathbf{emb}(\text{source\_full})
$$

Where $n_e$ is the number of extracted statements and $\mathbf{emb}(\text{source\_full})$ is the embedding of the entire source text.

$$
\text{RDF (Redundancy Factor)} = \begin{cases}
1.0 & \text{if } n_e \leq 1 \\
\frac{1}{\binom{n_e}{2}} \sum_{i < j} \mathbb{1}[\text{similarity}(e_i, e_j) < 0.5] & \text{otherwise}
\end{cases}
$$

Where the indicator $\mathbb{1}[\cdot]$ counts statement pairs with similarity below 0.5 (lower redundancy is better).

$$
\text{REF (Relevance Factor)} = \frac{\text{covered\_source\_sentences}}{\text{total\_source\_sentences}}
$$

Where a source sentence $s_k$ is covered if $\max_i \text{similarity}(e_i, s_k) > 0.6$.

$$
\text{BAF (Bias Avoidance Factor)} = \frac{1}{n_e} \sum_{i=1}^{n_e} \max_j \text{similarity}(e_i, s_j)
$$

Where the maximum is taken over all source sentences $s_j$.

**Final SUSWIR score (equal weighting):**

$$
\text{SUSWIR} = \frac{\text{SSF} + \text{RDF} + \text{REF} + \text{BAF}}{4}
$$

Note:
- Source text is split into sentences at boundaries of $\geq 10$ characters.
- Embeddings are L2-normalized.
- All similarities are computed as dot products of normalized embeddings.
- Embedding model used: `Qwen/Qwen3-Embedding-8B`

---

## 2) Coverage Metrics (Ground-Truth Topic Coverage)

Coverage is evaluated with three approaches:
- Approach A: embedding + similarity with Hungarian maximum matching
- Approach B: bi-encoder + similarity candidate retrieval, then cross-encoder entailment scoring
- Approach C: LLM as judge (binary covered/not covered per fact)

### Approach A: Embedding + Similarity with Hungarian Matching

**Matching setup:**

For each conversation, compute the similarity matrix between ground-truth facts and chatbot utterances:

$$
S_{ij} = \text{similarity}(\text{gt}_i, \text{utterance}_j) = \mathbf{emb}(\text{gt}_i) \cdot \mathbf{emb}(\text{utterance}_j)
$$

Apply Hungarian maximum matching to find the one-to-one assignment that maximizes total similarity.

**Per-fact hit determination:**

$$
\text{hit}_i = \begin{cases}
1 & \text{if } S_{\text{matched}(i)} \geq \text{threshold} \\
0 & \text{otherwise}
\end{cases}
$$

Where $\text{matched}(i)$ is the utterance index assigned to fact $i$ by the Hungarian algorithm.

**Metrics:**

$$
\text{hit\_rate} = \frac{\sum_{i=1}^{n_{\text{gt}}} \text{hit}_i}{n_{\text{gt}}}
$$

$$
\text{weighted\_critical\_recall} = \frac{\sum_{i=1}^{n_{\text{gt}}} w_i \cdot \text{hit}_i}{\sum_{i=1}^{n_{\text{gt}}} w_i}
$$

Where $w_i$ is the importance weight for fact $i$:
- Critical = 4.0
- High = 3.0
- Medium = 2.0
- Low = 1.0

**Embedding and threshold:**
- Embedding model: `Qwen/Qwen3-Embedding-8B`
- L2 normalization applied
- Default threshold: 0.75
- Utterance extraction: chatbot responses are split by sentence boundaries and bullet points, with fragments < 10 characters filtered out

### Approach B: Bi-encoder + Cross-encoder Entailment

**Candidate retrieval (bi-encoder):**

For each ground-truth fact, retrieve the top `K=3` matching chatbot utterances using cosine similarity of bi-encoder embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).

**Entailment scoring (cross-encoder):**

For the top 3 candidates per fact, compute entailment scores using a cross-encoder (`cross-encoder/nli-deberta-v3-base`).

$$
\text{hit}_i = \begin{cases}
1 & \text{if } \text{entailment\_score}_{\text{best}_i} \geq \text{threshold} \\
0 & \text{otherwise}
\end{cases}
$$

**Metrics:**

Same formulas as Approach A (hit_rate and weighted_critical_recall), but using entailment scores instead of embedding similarity.

### Approach C: LLM as Judge

An LLM is prompted to judge whether each ground-truth fact is covered by the conversation:

$$
\text{hit}_i = \begin{cases}
1 & \text{if LLM judges fact } i \text{ as covered} \\
0 & \text{otherwise}
\end{cases}
$$

**Metrics:**

Same formulas as Approach A (hit_rate and weighted_critical_recall), but using LLM judgments as the hit function.

---

## 3) Citation Faithfulness/Support Metrics

**Approach:** LLM as judge only.

### Claim and Citation Extraction

**Factual claim identification:**

Chatbot responses are split into sentences, then filtered to extract factual claims:
1. Sentence must end with `.`, `!`, or `?` (question-ending sentences are filtered out).
2. Sentence length (after stripping citations) must be $\geq 20$ characters.
3. Conversational filler is filtered (e.g., "Ich verstehe", "Okay", "Gerne").

**Citation extraction:**

For each claim, extract all citations matching the pattern `[Quelle X]` where $X \in \{1, 2, 3\}$.

$$
\text{citations per claim} = \{ X : \text{pattern } [Quelle X] \text{ found in sentence}\}
$$

### LLM Judging

For each (claim, source_chunk) pair, an LLM is prompted to assign one of three labels:

$$
\text{support\_label} \in \{\text{full\_support}, \text{partial\_support}, \text{no\_support}\}
$$

### Metrics

**Citation precision (strict, full-support only):**

$$
\text{strict\_full\_only} = \frac{\text{full\_support\_count}}{\text{total\_citations}}
$$

**Citation precision (relaxed, full + partial):**

$$
\text{relaxed\_full\_plus\_partial} = \frac{\text{full\_support\_count} + \text{partial\_support\_count}}{\text{total\_citations}}
$$

**Support coverage (claims with citations):**

$$
\text{support\_coverage} = \frac{\text{claims\_with\_citations}}{\text{total\_factual\_claims}}
$$

**Support distribution:**

$$
\text{support\_distribution\_full} = \frac{\text{full\_support\_count}}{\text{total\_citations}}
$$

$$
\text{support\_distribution\_partial} = \frac{\text{partial\_support\_count}}{\text{total\_citations}}
$$

$$
\text{support\_distribution\_none} = \frac{\text{no\_support\_count}}{\text{total\_citations}}
$$

Where:
- $\text{total\_citations}$ = total number of citation pairs (claim, source) across all turns
- $\text{total\_factual\_claims}$ = number of factual claims extracted (after filtering filler and questions)
- $\text{claims\_with\_citations}$ = count of factual claims that have $\geq 1$ citation

---

## 4) Mandatory Question Compliance Metrics

**Approach:** LLM as judge only.

### Per-Conversation Metrics

For each conversation, an LLM judges whether each mandatory question was explicitly asked by the chatbot (not just mentioned, but actively requested with clear question or request structure).

**Question recall (per conversation):**

$$
\text{question\_recall} = \frac{\text{mandatory\_questions\_asked}}{\text{mandatory\_questions\_total}}
$$

Where:
- $\text{mandatory\_questions\_total}$ = number of mandatory questions defined for the procedure
- $\text{mandatory\_questions\_asked}$ = count of mandatory questions the LLM judge marked as "is_asked = true"

**Compliance indicators (per conversation):**

$$
\text{strict\_compliance} = \begin{cases}
1 & \text{if } \text{mandatory\_questions\_total} > 0 \text{ and } \text{question\_recall} = 1.0 \\
0 & \text{otherwise}
\end{cases}
$$

$$
\text{acceptable\_compliance} = \begin{cases}
1 & \text{if } \text{question\_recall} \geq \text{compliance\_threshold} \\
0 & \text{otherwise}
\end{cases}
$$

**First question turn:**

$$
\text{first\_mandatory\_question\_turn} = \min\{\text{turn} : \text{question is asked at turn}\}
$$

Or $\text{null}$ if no mandatory questions were asked.

### Global/Mode Aggregates

**Mean question recall (across conversations in a mode):**

$$
\text{mean\_question\_recall} = \frac{1}{n} \sum_{c=1}^{n} \text{question\_recall}_c
$$

**Strict compliance rate:**

$$
\text{strict\_compliance\_rate} = \frac{1}{n} \sum_{c=1}^{n} \text{strict\_compliance}_c
$$

**Acceptable compliance rate:**

$$
\text{acceptable\_compliance\_rate} = \frac{1}{n} \sum_{c=1}^{n} \text{acceptable\_compliance}_c
$$

**Mean first turn (over non-null cases):**

$$
\text{mean\_first\_mandatory\_question\_turn} = \frac{1}{m} \sum_{c \in \text{non-null}} \text{first\_mandatory\_question\_turn}_c
$$

Where $m$ is the count of conversations with $\text{first\_mandatory\_question\_turn} \neq \text{null}$.

**Judge failure rate:**

$$
\text{judge\_failure\_rate} = \frac{\text{failed\_judge\_conversations}}{n}
$$

Where $n$ is the total number of conversations evaluated and failed conversations are those where the LLM judge batch call failed or returned invalid output.

**Parameters:**
- Default acceptable compliance threshold: 0.75
- Judge model: `gpt-4o-mini` (or configurable)
