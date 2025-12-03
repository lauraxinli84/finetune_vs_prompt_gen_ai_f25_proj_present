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

**Central finding:** For Gemma-2-2B, fine-tuning vastly outperforms prompting strategies, and few-shot prompting actively hurts performance.

---

## Methodology

### Model Architecture

**Gemma-2-2B-it** (2.6B parameters)
- Decoder-only transformer (GPT-style)
- Causal masking, autoregressive prediction
- Instruction-tuned for prompt following
- Small enough for consumer GPU training (Google Colab Pro L4)

### Datasets and Splits

Two domains for cross-domain evaluation:
- **Yelp**: Restaurant reviews (from Hugging Face)
- **Amazon**: Product reviews (from Kaggle; subsampled from 3M train/650K test to 200K train/50K test)
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

Extends zero-shot with 4 examples selected through stratified sampling:

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
- Example selection: `n_shots // num_classes = 4 // 5 = 0` samples per class, plus `4 % 5 = 4` extra samples
  - Result: 4 examples selected from first 4 classes (labels 0-3), none from class 4
- Context length: 800-1,000 tokens (16-20× longer than zero-shot)
- Theoretical basis: In-context learning [Brown et al., 2020]

The stratified sampling function divides shots evenly across classes, then distributes remainder to first classes. With 4 shots and 5 classes, this means one example each from classes 0, 1, 2, and 3. **Notably, 5-star reviews were not represented in the few-shot examples**, which may have contributed to the poor performance.

### Approach 3: LoRA Fine-Tuning

Low-Rank Adaptation adds trainable matrices to attention layers without updating the base model.

**Mathematical formulation:**
```
h = W₀x + BAx
where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), rank r=8
```

**Configuration:**
- Trainable parameters: 3.2M (0.12% of total)
- Target modules: All attention projections (q_proj, k_proj, v_proj, o_proj)
- Rank: r=8, alpha=16, dropout=0.05
- Training: 2 epochs, effective batch size 8 (1×8 gradient accumulation)
- Optimizer: AdamW, lr=2×10⁻⁴, 50 warmup steps
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

**Key observations:**
- Strongest performance on extreme sentiments (1-star, 5-star): clear linguistic signals
- Moderate performance on neutral class (3-star): inherently ambiguous
- Balanced precision-recall across all classes indicates well-calibrated model
- Fine-tuning improved accuracy by 14.4 percentage points while maintaining reasonable parse rates

### Amazon Performance (Cross-Domain Transfer)

| Method | Accuracy | F1 Macro | F1 Weighted | Failed Parses |
|--------|----------|----------|-------------|---------------|
| Zero-shot | 42.70% | 36.75% | 36.55% | 10 / 3,000 (0.3%) |
| Few-shot | 24.33% | 16.11% | 16.20% | 2,718 / 3,000 (91%) |
| **Fine-tuned** | **60.43%** | **60.19%** | **60.33%** | 1 / 3,000 (0.03%) |

**Cross-domain transfer analysis:**
- Model trained only on Yelp generalizes well to Amazon despite domain shift
- 17.7 percentage point improvement over zero-shot demonstrates robust learning
- Near-perfect parsing (only 1 failure) shows stable output formatting
- Performance drop from 67.3% (Yelp) to 60.4% (Amazon) reflects domain differences:
  - Restaurant reviews emphasize service and experience
  - Product reviews emphasize features and value

The strong transfer validates that LoRA captures domain-agnostic sentiment patterns rather than restaurant-specific language.

---

## Why Few-Shot Failed: Deep Dive

Few-shot prompting catastrophically failed on both datasets, contradicting findings from large models like GPT-3. We identify four root causes:

### 1. Cognitive Overload from Extended Context

Context length comparison:
- Zero-shot: ~50 tokens (instruction + review)
- Few-shot: 800-1,000 tokens (instruction + 4 examples + review)

For a 2B parameter model, the 16-20× context expansion creates processing bottlenecks:
- Attention mechanism must compute relationships across all tokens
- Limited model capacity struggles to maintain focus on the actual task
- Working memory effectively "overflows" with example content

### 2. Output Format Instability

Zero-shot outputs were concise and parseable:
```
Response: "4"
```

Few-shot outputs became verbose and unparseable:
```
Response: "Based on the review's positive tone and mention of excellent food, 
I would rate this 5. The customer clearly enjoyed their experience..."
```

**Root cause:** The examples may have signaled to the model that this is a "complex reasoning task" requiring explanation rather than a simple classification. The model attempts to justify its prediction rather than provide a direct answer.

**Mitigation attempt:** We implemented an improved extraction function with multiple strategies:
1. Pattern matching for "Sentiment: X" or "Rating: X"
2. Digit extraction (finding any 1-5 in the response)
3. Searching first 100 characters only

Even with improved parsing, few-shot still significantly underperformed zero-shot, confirming the fundamental issue.

### 3. Attention Dilution Across Examples

In few-shot prompting, the attention mechanism must:
1. Understand each of the 4 example reviews
2. Extract the pattern relating reviews to ratings
3. Apply this pattern to the test review
4. Generate an appropriate response

For small models, this divided attention reduces focus on the actual query. The model may be "distracted" by example content, unable to clearly distinguish "these are examples to learn from" versus "this is the query to answer."

### 4. Model Size Dependency

**Stark contrast with GPT-3:**
- GPT-3 175B: Few-shot prompting works excellently [Brown et al., 2020]
- Gemma-2B: Few-shot prompting fails catastrophically

**Hypothesis:** There exists a model size threshold (~7B parameters?) below which in-context learning becomes unreliable. Models below this threshold lack the capacity for robust few-shot learning, while models above it can effectively leverage examples.

**Missing 5-star examples:** Our stratified sampling selected examples from classes 0-3 only. The absence of 5-star review examples may have further degraded performance, especially on highly positive reviews.

---

## Key Insights and Contributions

### 1. Model Size Fundamentally Determines Adaptation Strategy

This work reveals a critical interaction between model scale and optimal adaptation approach:

**Small models (<7B params):**
- Few-shot prompting: Fails or degrades performance
- Fine-tuning: Essential for good performance
- Implication: Budget LoRA fine-tuning time into deployment

**Large models (>100B params):**
- Few-shot prompting: Highly effective
- Fine-tuning: May be overkill
- Implication: Prompt engineering can suffice

**This finding contradicts the assumption that techniques scale uniformly across model sizes.** Practitioners must validate approaches at their deployment scale.

### 2. LoRA Validates Low-Rank Hypothesis

- Updated only 0.12% of parameters (3.2M / 2.6B)
- Achieved 67% accuracy (14.4% gain over zero-shot)
- Rank r=8 sufficient - suggests most adaptation information lies in low-dimensional subspace
- Efficient enough for consumer GPU (90 minutes, 12.8 MB adapter)

This confirms the LoRA paper's hypothesis that weight changes during adaptation have low intrinsic rank. The pre-trained model already contains the necessary features; adaptation primarily amplifies and redirects them.

### 3. Cross-Domain Transfer Demonstrates Generalization

Single Yelp-trained adapter achieved:
- 60.4% on Amazon (vs. 42.7% zero-shot)
- Only 1 failed parse in 3,000 examples
- Graceful 6.9% accuracy degradation from in-domain

This suggests LoRA learns generalizable sentiment representations rather than domain-specific shortcuts. The adapter captures linguistic patterns that transfer across review types.

### 4. Practical Deployment Considerations

**Cost-benefit analysis:**
- Fine-tuning: 90 minutes one-time, +14-17% accuracy, 12.8 MB storage
- Prompting: Zero training, -14-17% accuracy, unreliable outputs

For production systems requiring reliable sentiment analysis, the fine-tuning investment is trivial compared to ongoing accuracy loss from prompting approaches.

---

## Implementation Details

### Technologies Stack
- **Framework**: PyTorch 2.0 + Hugging Face Transformers
- **Fine-tuning**: PEFT library (LoRA implementation)
- **Hardware**: Google Colab Pro L4 GPU (22.5GB VRAM)
- **Data**: Hugging Face Datasets library

### Memory Optimization Techniques

Fitting a 2.6B parameter model on consumer GPU required several optimizations:

```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,      # Minimal batch to fit in memory
    gradient_accumulation_steps=8,      # Effective batch size = 8
    fp16=True,                          # Mixed precision (2× speedup)
    gradient_checkpointing=True,        # Recompute activations (saves memory)
    learning_rate=2e-4,
    warmup_steps=50,
    num_train_epochs=2,
)

# Tokenization settings
max_length=128  # Truncate long reviews to fit memory
```

**Gradient checkpointing** trades computation for memory by recomputing activations during backward pass instead of storing them. This increases training time by ~20% but dramatically reduces memory requirements.

### Generation Configuration for Reliable Parsing

```python
# Optimized for parseable outputs
max_new_tokens=10        # Force concise answers
temperature=0.1          # Nearly deterministic sampling
do_sample=True           # Enable temperature control
```

Limiting generation to 10 tokens was crucial - it prevented the model from generating lengthy explanations and forced direct numerical outputs. This single change improved parsing success from 56% to 95% for the fine-tuned model.

### Robust Output Parsing Strategy

Implemented hierarchical fallback approach:

1. **Pattern matching**: Look for "Sentiment: X" or "Rating: X" format
2. **Digit extraction**: Find any digit 1-5 anywhere in response
3. **Positional search**: Check first 100 characters only (where answers typically appear)
4. **Default fallback**: If all fail, default to class 2 (neutral)

This engineering improved few-shot parsing from complete failure (91% on Amazon) to marginal viability, though performance still lagged significantly behind zero-shot.

---

## Theoretical Grounding

### Formal Algorithms for Transformers [Phuong & Hutter, 2022]

Our implementation directly follows three core algorithms from this paper:

**Algorithm 10: DTransformer** (Decoder-only architecture)
- Implements unidirectional masked self-attention
- Causal masking ensures token t only attends to tokens 1:t
- Enables autoregressive next-token prediction

**Algorithm 13: DTraining** (Training procedure)
- Minimizes cross-entropy loss: `loss = -Σ log P(x[t+1] | x[1:t])`
- Updates parameters via stochastic gradient descent
- Our implementation: Update only LoRA parameters (Θ ⊂ Φ₀), freeze base model

**Algorithm 14: DInference** (Prompting)
- Inference without parameter updates
- Temperature τ controls sampling distribution: `P ∝ p^(1/τ)`
  - τ=0.1 (our setting): Nearly deterministic
  - τ=1.0: Faithful to model distribution
  - τ→∞: Uniform sampling

These formal specifications enable precise reproduction and provide theoretical foundation for our empirical findings.

### Low-Rank Adaptation Theory [Hu et al., 2021]

**Core hypothesis:** "The change in weights during model adaptation has a low intrinsic rank"

**Our validation:**
- Rank r=8 captures sufficient adaptation information
- 67% accuracy represents 72% of theoretical maximum improvement (from 53% to 100%)
- Cross-domain transfer confirms learned features are generalizable, not dataset-specific

**Interpretation:** Pre-trained models already contain rich feature representations from large-scale training. Task adaptation doesn't require learning fundamentally new features - instead, it amplifies and recombines existing ones. This recombination can be expressed as a low-rank perturbation to the original weights.

The mathematical intuition: If W₀ ∈ ℝ^(d×k) is the original weight matrix, adaptation seeks ΔW such that W₀ + ΔW performs well on the new task. LoRA parameterizes ΔW = BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and r << min(d,k). This constrains ΔW to have rank ≤ r, dramatically reducing trainable parameters.

---

## Practical Recommendations

### When to Use Each Approach

**Zero-shot prompting:**
- ✅ Rapid prototyping and task validation
- ✅ No training data available
- ✅ Tasks with clear natural language instructions
- ✅ Acceptable accuracy ~50% for initial testing
- ❌ Production systems requiring high accuracy

**Few-shot prompting:**
- ❌ Small models (<7B params) - avoid entirely
- ✅ Large models (>100B params) - effective
- ❌ When reliable output formatting is critical
- ❌ Domains with subtle linguistic patterns

**Fine-tuning with LoRA:**
- ✅ Need maximum accuracy (production systems)
- ✅ Have 5,000+ labeled examples
- ✅ Can afford 1-2 hours training time
- ✅ Domain-specific terminology or patterns
- ✅ Require consistent, parseable outputs
- ✅ Models <7B parameters (essential, not optional)

### Deployment Workflow

**Phase 1: Validation (Day 1)**
1. Implement zero-shot prompting
2. Evaluate on small sample (100-500 examples)
3. Establish baseline accuracy
4. Identify error patterns

**Phase 2: Data Collection (Week 1)**
1. Collect domain-specific labeled data
2. Target: 5,000+ samples with stratified classes
3. Split: 5k train / 1k val / remaining test
4. Verify class balance

**Phase 3: Fine-Tuning (Week 1-2)**
1. Configure LoRA (r=8, target attention layers)
2. Train 2 epochs (~90 minutes on L4)
3. Monitor validation loss for convergence
4. Save adapter (12.8 MB)

**Phase 4: Deployment (Week 2)**
1. Merge adapter into base model (optional, for speed)
2. Deploy with same inference code as zero-shot
3. Monitor parse failure rates
4. Collect edge cases for future iteration

**Multi-task strategy:** For systems handling multiple tasks, train separate LoRA adapters per task. Keep one base model, swap adapters as needed. This is far more efficient than maintaining multiple full fine-tuned models.

---

## Limitations and Future Work

### Current Study Limitations

**Model coverage:**
- Only evaluated one model size (2B parameters)
- Need systematic experiments: 2B → 7B → 13B → 70B → 175B
- Would identify precise threshold where few-shot becomes viable

**LoRA configuration:**
- Only tested rank r=8
- Ablation study needed: r ∈ {4, 8, 16, 32, 64}
- Could optimize efficiency-performance trade-off

**Few-shot design:**
- Single example selection strategy (stratified)
- Missing 5-star examples in 4-shot setup
- Alternative strategies: diversity sampling, semantic similarity, active learning

**Task scope:**
- Limited to sentiment classification (5-class)
- Findings may not generalize to:
  - Question answering
  - Summarization
  - Code generation
  - Named entity recognition

### Priority Future Directions

**1. Model scaling study:**
Systematically test adaptation strategies across model scales to identify phase transitions. Expected finding: few-shot viability emerges around 7-13B parameter range.

**2. Rank optimization:**
Test whether r=4 achieves similar results (50% fewer parameters) or if r=16 meaningfully improves accuracy (more capacity).

**3. Full fine-tuning comparison:**
Quantify performance gap between LoRA and full parameter updates. Hypothesis: LoRA achieves 90-95% of full fine-tuning gains at <1% of cost.

**4. Alternative PEFT methods:**
Compare LoRA against prefix tuning, adapter layers, and (IA)³. Different methods may suit different model architectures or tasks.

**5. Hybrid approaches:**
Investigate combining prompting and fine-tuning - e.g., fine-tune on general task, then use few-shot for domain adaptation. Could this enable fast task switching without training separate adapters?

**6. Theoretical analysis:**
Develop mathematical framework for predicting when few-shot will succeed based on model capacity, context length, and task complexity.

---

## Conclusions

### Main Contributions

1. **Empirical evidence for model size dependence:** Few-shot prompting fails catastrophically on small models (2B params) despite success on large models (175B params). This contradicts assumptions about uniform technique scaling.

2. **LoRA efficiency validation:** Updating 0.12% of parameters achieves 67% accuracy (+14% over baseline), confirming low-rank adaptation hypothesis for sentiment classification.

3. **Cross-domain transfer demonstration:** Single adapter trained on restaurant reviews achieves 60% accuracy on product reviews (+18% over baseline), showing generalizable learning.

4. **Practical deployment guidance:** For models <7B parameters, fine-tuning is essential - prompt engineering is insufficient. Adaptation strategy must match model scale.

### Answering the Central Question

**For Gemma-2-2B and similar small models:** Fine-tuning with LoRA is the clear winner. The computational cost (90 minutes, one-time) is justified by:
- 14-17% absolute accuracy improvement
- Reliable, parseable outputs
- Cross-domain generalization
- Minimal storage overhead (12.8 MB)

**However, this answer is model-scale dependent.** Larger models may achieve comparable results with prompt engineering alone, eliminating fine-tuning overhead. The field needs more research characterizing these transitions.

### Broader Impact

This work provides actionable guidance for practitioners deploying LLMs in resource-constrained environments. Key takeaways:

1. **Don't assume large-model findings transfer downward** - Always validate techniques at your deployment scale
2. **Parameter-efficient fine-tuning is accessible** - Consumer GPUs can train billion-parameter models
3. **Small models can achieve strong performance** - With proper adaptation, 2B models reach 67% accuracy
4. **Adaptation strategy is not universal** - Match technique to model size, task complexity, and data availability

As language models proliferate across diverse applications and model scales continue to fragment (from billions to hundreds of billions of parameters), understanding scale-dependent behavior becomes increasingly critical. This work contributes empirical evidence toward that understanding.

---

## References

1. Phuong, M., & Hutter, M. (2022). Formal Algorithms for Transformers. *arXiv preprint arXiv:2207.09238*. https://arxiv.org/abs/2207.09238

2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR) 2022*. https://arxiv.org/abs/2106.09685

3. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems (NeurIPS) 2020*. https://arxiv.org/abs/2005.14165

4. Vaswani, A., et al. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS) 2017*. https://arxiv.org/abs/1706.03762

5. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Association for Computational Linguistics (ACL) 2019*. https://arxiv.org/abs/1810.04805

---

## Resources and Links

### Model
- **Gemma-2-2B-it**: https://huggingface.co/google/gemma-2-2b-it
- Model card with architecture details and usage guidelines
- Requires Hugging Face account and agreement to terms

### Datasets
- **Yelp Review Full**: https://huggingface.co/datasets/Yelp/yelp_review_full
  - 650,000 train samples, 50,000 test samples
  - 5-class sentiment (1-5 stars)
  
- **Amazon Reviews (Kaggle)**: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
  - Original: 3,000,000 train, 650,000 test
  - Subsampled to 200,000 train, 50,000 test in this study

### Libraries and Tools
- **Hugging Face Transformers**: https://github.com/huggingface/transformers
- **PEFT (Parameter-Efficient Fine-Tuning)**: https://github.com/huggingface/peft
- **PyTorch**: https://pytorch.org
- **Google Colab**: https://colab.research.google.com

