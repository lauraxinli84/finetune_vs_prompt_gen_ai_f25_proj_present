# Prompt Engineering vs. Fine-Tuning for Sentiment Classification

**Laura Li** | Generative AI Models in Theory and Practice | Vanderbilt University | December 4, 2025

---

## The Central Question

When adapting LLMs to domain-specific tasks, should you use prompt engineering or fine-tuning? This project compares three strategies for 5-star sentiment classification:

1. **Zero-shot prompting** - baseline with clear instructions
2. **Few-shot prompting** - 4 examples provided in context  
3. **LoRA fine-tuning** - parameter-efficient training

We implement algorithms from the Formal Algorithms for Transformers paper [Phuong & Hutter, 2022]: Algorithm 14 (inference-time prompting) and Algorithm 13 (training-time parameter updates).

---

## Results Summary

| Dataset | Zero-Shot | Few-Shot | Fine-Tuned | Improvement |
|---------|-----------|----------|------------|-------------|
| **Yelp** | 52.9% | 44.7% ❌ | **67.3%** | +14.4% |
| **Amazon** | 42.7% | 24.3% ❌ | **60.4%** | +17.7% |

**Key findings:**
- Few-shot prompting degraded performance by 8-18 percentage points
- Fine-tuning achieved substantial gains over zero-shot baseline
- Model trained only on Yelp transferred well to Amazon (60.4% vs. 42.7% baseline)
- Failed parse rates: Zero-shot ~0.3%, Few-shot 44-91%, Fine-tuned ~5%

The central finding: **For Gemma-2-2B, fine-tuning vastly outperforms prompting strategies, and few-shot prompting actively hurts performance.**

---

## Methodology

### Model Architecture

**Gemma-2-2B-it** (2.5B parameters)
- Decoder-only transformer (GPT-style)
- Causal masking, autoregressive prediction
- Instruction-tuned for prompt following
- Small enough for consumer GPU training (Google Colab L4)

### Datasets and Splits

Two domains for cross-domain evaluation:
- **Yelp**: Restaurant reviews
- **Amazon**: Product reviews  
- Both use 1-5 star ratings

**Stratified splits per dataset:**
- Training: 5,000 (1,000 per class)
- Validation: 1,000 (200 per class)
- Test: 3,000 (~600 per class)

### Approach 1: Zero-Shot Prompting

Clear instruction with rating definitions:

```
Classify the sentiment of the following review on a scale of 1 to 5:
1 = Very Negative
2 = Negative
3 = Neutral
4 = Positive
5 = Very Positive

Review: {text}

Sentiment (1-5):
```

**Configuration:**
- Parameters updated: 0
- Temperature: 0.1 (nearly deterministic)
- Max new tokens: 10 (force direct answers)
- Context length: ~50 tokens

### Approach 2: Few-Shot Prompting

Extends zero-shot with 4 stratified examples (one per class) before the test case:

```
Classify the sentiment of reviews on a scale of 1 to 5:
[rating definitions]

Here are some examples:

Review: {example_1_text}
Sentiment: {example_1_label}

[...3 more examples...]

Now classify this review:
Review: {text}
Sentiment:
```

**Configuration:**
- Parameters updated: 0
- Examples: Stratified sampling from training set
- Context length: 800-1,000 tokens (16-20× longer than zero-shot)
- Theoretical basis: In-context learning [Brown et al., 2020]

### Approach 3: LoRA Fine-Tuning

Low-Rank Adaptation adds trainable matrices to attention layers without updating the base model.

**Mathematical formulation:**
```
h = W₀x + BAx
where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), rank r=8
```

**Configuration:**
- Trainable parameters: 3.2M (0.12% of total)
- Target modules: Query and Value projections in attention
- Rank: r=8, alpha=16
- Training: 2 epochs, batch size 8 (via gradient accumulation)
- Optimizer: AdamW, lr=2×10⁻⁴
- Time: ~90 minutes on L4 GPU

**Memory optimizations:**
- FP16 mixed precision
- Gradient checkpointing
- Sequence truncation to 128 tokens
- Per-device batch size of 1 with 8 gradient accumulation steps

**Key insight from LoRA paper:** "The change in weights during model adaptation has a low intrinsic rank" - we validate this by achieving strong performance with only rank 8.

---

## Results and Analysis

### Yelp Performance (Restaurant Reviews)

| Method | Accuracy | F1 Macro | F1 Weighted | Failed Parses |
|--------|----------|----------|-------------|---------------|
| Zero-shot | 52.93% | 48.72% | 49.13% | 6 / 3,000 (0.2%) |
| Few-shot | 44.67% | 45.53% | 45.64% | 1,331 / 3,000 (44%) |
| **Fine-tuned** | **67.33%** | **67.39%** | **67.41%** | 146 / 3,000 (5%) |

**Fine-tuned per-class performance:**

```
Class          Precision  Recall  F1    Support
1-star (⭐)       0.82     0.75   0.78    645
2-star (⭐⭐)      0.64     0.65   0.64    583
3-star (⭐⭐⭐)     0.58     0.64   0.61    579
4-star (⭐⭐⭐⭐)    0.64     0.59   0.61    650
5-star (⭐⭐⭐⭐⭐)   0.70     0.76   0.73    543
```

**Observations:**
- Strongest performance on extreme sentiments (1-star, 5-star)
- Moderate performance on neutral class (3-star)
- Balanced precision-recall across all classes

### Amazon Performance (Cross-Domain Transfer)

| Method | Accuracy | F1 Macro | F1 Weighted | Failed Parses |
|--------|----------|----------|-------------|---------------|
| Zero-shot | 42.70% | 36.75% | 36.55% | 10 / 3,000 (0.3%) |
| Few-shot | 24.33% | 16.11% | 16.20% | 2,718 / 3,000 (91%) |
| **Fine-tuned** | **60.43%** | **60.19%** | **60.33%** | 1 / 3,000 (0.03%) |

The model trained only on Yelp generalizes well to Amazon despite domain shift (restaurant → product reviews). The near-perfect parsing rate (only 1 failure) demonstrates robust output formatting.

### Why Few-Shot Failed: Analysis

Few-shot prompting catastrophically failed on both datasets. Three main causes:

**1. Cognitive Overload**
- Context expanded from 50 tokens (zero-shot) to 800-1,000 tokens (few-shot)
- Small models (2B params) struggle to process multiple examples while focusing on the task
- Attention mechanism must attend to all examples simultaneously

**2. Output Format Chaos**
- Zero-shot: Direct outputs like "3" or "4"
- Few-shot: Verbose explanations like "Based on the review, I would rate this as positive because..."
- Parser couldn't extract ratings from these explanations (44-91% failure rate)

**3. Attention Dilution**
- Attention spreads across four example reviews instead of focusing on test case
- Model tries to understand examples and classify simultaneously
- Small models lack capacity for this divided attention

**4. Model Size Dependency**
- GPT-3 (175B params) excels at few-shot learning [Brown et al., 2020]
- Gemma-2B has 87× fewer parameters
- Hypothesis: Model size threshold (~7B params?) determines few-shot viability

The takeaway: Techniques that work on massive models don't necessarily transfer to smaller, deployable models.

---

## Key Insights

### 1. Model Size Determines Optimal Strategy

**Large models (>100B params):** Few-shot prompting works well  
**Small models (<7B params):** Few-shot fails, fine-tuning essential  
**Implication:** Don't generalize findings across model scales

### 2. LoRA Efficiency Validated

- Updated only 0.12% of parameters (3.2M / 2.5B)
- Achieved 67% accuracy (vs. 53% zero-shot)
- Rank r=8 sufficient for strong performance
- Confirms low-rank hypothesis: Adaptation requires few degrees of freedom

### 3. Cross-Domain Transfer Success

- Single adapter trained on Yelp → 60% on Amazon
- +17.7% over zero-shot baseline
- Suggests LoRA captures domain-agnostic sentiment patterns
- Nearly perfect parsing (1/3,000 failures) despite never seeing product reviews

### 4. Practical Trade-offs

**Fine-tuning cost:**
- One-time: 90 minutes on L4 GPU
- Adapter size: 12.8 MB
- Same inference speed as base model (merge adapters)

**Prompt engineering cost:**
- Zero training time
- 14-17% lower accuracy
- Unreliable output formatting with few-shot

For production systems, 90 minutes of training is trivial compared to ongoing accuracy loss.

---

## Implementation Highlights

### Technologies
- **Framework**: PyTorch 2.0 + Hugging Face Transformers
- **Training**: PEFT library (LoRA)
- **Hardware**: Google Colab L4 GPU

### Memory Optimizations for 16GB GPU

```python
# Fitting 2.5B model on consumer hardware
per_device_train_batch_size=1      # Minimal batch
gradient_accumulation_steps=8      # Effective batch=8
fp16=True                          # Mixed precision (2× speedup)
gradient_checkpointing=True        # Trade compute for memory
max_length=128                     # Truncate long sequences
```

### Generation Configuration

```python
# Force short, parseable outputs
max_new_tokens=10        # Direct answers only
temperature=0.1          # Nearly deterministic
do_sample=True           # Temperature sampling
```

Limiting generation to 10 tokens improved parsing from 44% failure → 5% failure for fine-tuned model.

### Robust Output Parsing

Hierarchical parser to handle format variation:
1. Pattern match: "Sentiment: X" or "Rating: X"
2. Find any digit 1-5 in response
3. Check first 100 characters as fallback

This engineering detail made the difference between a partially-working and fully-working system.

---

## Theoretical Connections

### Formal Algorithms for Transformers [Phuong & Hutter, 2022]

Our implementation directly follows three core algorithms:

**Algorithm 10: DTransformer** (Decoder-only architecture)
- Unidirectional masked self-attention
- Causal masking: token t only attends to tokens 1:t
- Autoregressive next-token prediction

**Algorithm 13: DTraining** (Training procedure)
- Minimize cross-entropy loss on next-token prediction
- Update parameters via stochastic gradient descent
- We update only LoRA parameters (Θ ⊂ Φ₀)

**Algorithm 14: DInference** (Prompting)
- Inference without parameter updates
- Temperature τ controls sampling: 0→deterministic, 1→faithful, ∞→uniform
- Used for both zero-shot and few-shot approaches

### Low-Rank Adaptation Theory [Hu et al., 2021]

**Hypothesis:** "The change in weights during model adaptation has a low intrinsic rank"

**Our validation:**
- Rank r=8 achieved 67% accuracy (+14% over baseline)
- Cross-domain transfer supports domain-agnostic learning
- Suggests adaptation amplifies existing features rather than learning new ones

If adaptation required high-rank updates, we'd need much larger rank values for good performance.

---

## Practical Recommendations

### When to Use Zero-Shot
- Quick prototyping without training
- Well-defined tasks with clear instructions
- Accept limited accuracy for small models (~50%)

### When to Use Fine-Tuning
- Need maximum accuracy
- Have labeled training data (5,000+ samples)
- Can afford one-time training cost (~90 minutes)
- Domain-specific terminology or specialized tasks
- Critical: Consistent output formatting required

**For models <7B parameters, default to fine-tuning over prompt engineering.**

### When to Avoid Few-Shot
- Using models <7B parameters
- Model tends to generate verbose explanations
- Parsing outputs reliably is important

For small models, our advice is clear: **Skip few-shot prompting entirely.** Use zero-shot for quick tests, then move directly to fine-tuning.

### Deployment Workflow

1. **Validate** with zero-shot to establish baseline
2. **Collect** 5,000+ labeled examples with stratified sampling
3. **Fine-tune** using LoRA (few hours on consumer GPU)
4. **Deploy** small adapter (13MB) alongside base model
5. **Swap** adapters for multi-task systems without reloading base model

---

## Limitations and Future Work

### Current Limitations

**Model coverage:**
- Only tested Gemma-2-2B (one model size)
- Need systematic experiments across 2B, 7B, 13B, 70B to find few-shot threshold

**LoRA configuration:**
- Only tested rank r=8
- No ablation study on ranks 4, 16, 32

**Prompt engineering:**
- Single template design
- Better prompts might improve few-shot (though unlikely to close 18% gap)

**Task coverage:**
- Only sentiment classification
- May not generalize to QA, summarization, code generation

### Future Directions

**Immediate extensions:**
1. **Model scaling study**: Test 7B, 13B, 70B models with same setup to identify few-shot threshold
2. **Rank ablation**: Compare r=4,8,16,32 for efficiency-performance trade-off
3. **Full fine-tuning comparison**: Quantify LoRA vs. full parameter updates
4. **Other PEFT methods**: Test prefix tuning, adapter layers, (IA)³

**Broader research:**
- Extend to other task types (NER, QA, summarization)
- Test instruction-tuned vs. base models
- Investigate hybrid approaches combining prompting and fine-tuning

---

## Conclusions

### Main Takeaways

1. **Fine-tuning wins decisively** for small models on domain-specific tasks
   - +14-17% absolute accuracy over zero-shot
   - Reliable, parseable outputs
   - Modest computational cost (90 minutes)

2. **Few-shot prompting can harm** small model performance
   - -8 to -18% accuracy degradation
   - 44-91% output parsing failures
   - Model size matters for in-context learning

3. **LoRA is efficient and practical**
   - 0.12% parameters, 67% accuracy
   - Strong cross-domain transfer (60% on unseen domain)
   - Adapter size: 13MB, no inference latency

4. **Adaptation strategy depends on model size**
   - <7B params: Fine-tune
   - >100B params: Prompting viable
   - No universal best practice

### Answering the Central Question

For Gemma-2-2B and similar small models: **Fine-tuning with LoRA is the clear winner.** The computational cost is justified by significant accuracy gains and reliable output formatting.

However, this answer is model-scale dependent. Findings from massive models like GPT-3 don't transfer to smaller, more practical models. Practitioners must validate techniques at their deployment scale rather than assuming frontier model results will transfer downward.

---

## References

1. Phuong, M., & Hutter, M. (2022). Formal Algorithms for Transformers. *arXiv preprint arXiv:2207.09238*.

2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

3. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS 2020*.

4. Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS 2017*.

5. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *ACL 2019*.

---

**Questions?**
