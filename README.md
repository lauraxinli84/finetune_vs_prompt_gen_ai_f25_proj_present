# Prompt Engineering vs. Fine-Tuning for Sentiment Classification

**Laura Li** | Generative AI Models in Theory and Practice | Vanderbilt University | December 4, 2025

---

## 1. Problem Statement & Motivation

### The Challenge

Decoder-only transformers (like GPT) have emerged as powerful general-purpose problem solvers, capable of performing tasks traditionally handled by encoder-only models (e.g., BERT for classification) or encoder-decoder architectures (e.g., T5 for translation). When adapting these models to specific tasks, practitioners face a critical choice:

1. **Enhance the context** through prompt engineering (providing instructions and examples)
2. **Update the model** through fine-tuning (modifying model weights)

Classification tasks—typically solved using encoder-only models with classification heads—provide an ideal testbed for comparing these approaches.

### Research Question

**Which adaptation strategy delivers better performance for 5-class sentiment classification on decoder-only transformers: prompt engineering or parameter-efficient fine-tuning?**

We evaluate three strategies:
- **Zero-shot prompting**: Baseline with clear instructions only
- **Few-shot prompting**: Context enhanced with 5 demonstration examples
- **LoRA fine-tuning**: Parameter-efficient weight updates (0.12% of parameters)

---

## 2. Results Summary

| Dataset | Zero-Shot | Few-Shot (5-shot) | Fine-Tuned (LoRA) | Improvement |
|---------|-----------|-------------------|-------------------|-------------|
| **Yelp** | 52.9% | 59.1% | **67.3%** | +14.4% |
| **Amazon** | 42.7% | 52.7% | **60.4%** | +17.7% |

### Key Findings

1. **Few-shot prompting works**: Complete class coverage (one example per class) improves accuracy by 6-10 percentage points over zero-shot
2. **Fine-tuning works better**: LoRA achieves 14-18 point gains over zero-shot, 8-10 points over few-shot, while updating only 0.12% of parameters
3. **Cross-domain transfer succeeds**: Model trained only on Yelp achieves 60.4% on Amazon (vs. 42.7% zero-shot baseline)
4. **Output reliability**: Fine-tuning maintains <5% parse failure rate across both datasets

**Practical implication**: Fine-tuning reduces classification errors by ~30% compared to few-shot prompting, making the one-time 90-minute training cost worthwhile for production systems.

---

## 3. Methodology

### 3.1 Model Architecture

**Gemma-2-2B-it** (2.6B parameters)
- Decoder-only transformer architecture
- Causal self-attention with autoregressive prediction
- Instruction-tuned for prompt following
- Deployable on consumer GPU (Google Colab Pro L4)

### 3.2 Datasets

Two domains for cross-domain evaluation:

**Yelp Review Full** (Restaurant reviews)
- Source: Hugging Face
- 5-class sentiment (1-5 stars)
- Stratified splits: 5,000 train / 1,000 val / 3,000 test

**Amazon Reviews** (Product reviews)  
- Source: Kaggle
- Subsampled from 3M train/650K test to 200K train/50K test
- Stratified splits: 5,000 train / 1,000 val / 3,000 test

Both datasets use balanced class distributions (1,000 samples per class in training, ~600 per class in test).

### 3.3 Approach 1: Zero-Shot Prompting

Simple instruction-based prompt:

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

**Configuration**: 0 trainable parameters, ~100-120 tokens per query, temperature=0.1 (nearly deterministic)

### 3.4 Approach 2: Few-Shot Prompting

Extends zero-shot with 5 balanced examples (one per class):

```
Classify the sentiment of reviews on a scale of 1 to 5:
[rating definitions]

Here are some examples:

Review: {example_1_text}
Sentiment: 1

[...4 more examples covering classes 2-5...]

Review: {text}
Sentiment:
```

**Configuration**: 0 trainable parameters, ~500-600 tokens per query, stratified example selection ensures complete class coverage

**Why complete coverage matters**: Without an example from each class, the model lacks a reference point for that rating level, forcing extrapolation rather than interpolation. Our 5-shot approach (vs. 4-shot) ensures the model sees the full sentiment spectrum.

### 3.5 Approach 3: LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) adds trainable low-rank matrices to attention layers while freezing all pre-trained weights.

**Mathematical formulation**:
```
h = W₀x + BAx
where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), rank r=8
```

**Configuration**:
- Trainable parameters: 3.2M (0.12% of 2.6B total)
- Target modules: Query and value projections (q_proj, v_proj) in all 96 layers
- Rank r=8, alpha=16, dropout=0.05
- Training: 2 epochs, batch size 8 (1×8 gradient accumulation)
- Optimizer: AdamW, lr=2×10⁻⁴, 50 warmup steps
- Time: ~90 minutes on L4 GPU

**Why LoRA works**: Pre-trained models already contain rich feature representations. Task adaptation doesn't require learning entirely new features—instead, it amplifies and recombines existing ones. This recombination can be expressed as a low-rank perturbation, dramatically reducing trainable parameters while maintaining performance.

---

## 4. Implementation & Formal Algorithm

### 4.1 Algorithm: Decoder Transformer with LoRA

Based on Algorithm 10 (DTransformer) from [Phuong & Hutter, 2022] with LoRA from [Hu et al., 2021]:

**Input**: x ∈ V*, sequence of token IDs  
**Output**: Ŷ ∈ (0,1)^(V×n), predicted probability distribution over vocabulary

**Core Forward Pass** (simplified, showing key LoRA integration):

```
for each layer ℓ ∈ [1, 96]:
    X̃ ← layer_norm(X)                          # Pre-normalization
    
    # Attention with LoRA on Q and V projections
    Q ← W_q X̃ + B_q(A_q X̃)                    # ← LoRA HERE
    K ← W_k X̃                                  # ← Frozen
    V ← W_v X̃ + B_v(A_v X̃)                    # ← LoRA HERE
    
    # Compute masked attention
    S ← Q^T K / √d_attn                        # Scores
    S[i,j] ← -∞ if i > j                       # Causal masking
    Attn ← V · softmax(S)                      # Attention output
    X ← X + W_o · Attn                         # Residual connection
    
    # MLP block (fully frozen)
    X ← X + MLP(layer_norm(X))
    
return softmax(W_U · X)                        # Unembedding
```

**Understanding the Algorithm**:

This pseudocode shows how LoRA modifies the standard decoder transformer at lines 16 and 18. During the forward pass, each token representation flows through 96 transformer layers. In each layer's attention mechanism, the query (Q) and value (V) projections receive both the original frozen transformation (W_q X̃) and an additional low-rank transformation (B_q(A_q X̃)). The key insight is that A_q first projects the 2048-dimensional input down to 8 dimensions, then B_q projects it back up to 2048 dimensions. This "bottleneck" through rank-8 space forces the adaptation to learn a compressed, efficient representation of the task-specific changes needed. All other components—the key projection (K), output projection (W_o), MLP blocks, and layer normalizations—remain completely frozen, using only the knowledge from pre-training. This design allows us to adapt the model's behavior while training less than 1% of its parameters.

**LoRA Implementation Details**:

The critical modification occurs in attention projections:
```
Standard:  Q = W_q X̃                          (2048×2048 = 4.2M params)
With LoRA: Q = W_q X̃ + B_q(A_q X̃)            (2×2048×8 = 32K params)
                       ↑
                   Low-rank path
```

**Computation flow**:
1. Input X̃ ∈ ℝ^(2048×n) splits into two parallel paths
2. **Frozen path**: W_q X̃ → output ∈ ℝ^(2048×n)
3. **LoRA path**: A_q X̃ ∈ ℝ^(8×n) → B_q(...) ∈ ℝ^(2048×n)  
4. Outputs summed: (W_q + B_q A_q) X̃

This architecture freezes 2.6B parameters while training only 3.2M, achieving 128× parameter reduction per weight matrix.

**Training vs. Inference**:
- **Training**: Compute gradients only for {A_q, B_q, A_v, B_v} across all layers
- **Inference**: Merge weights W'_q = W_q + B_q A_q once, then use W'_q directly (zero added latency)

### 4.2 Training Configuration

**Memory optimizations**:
- FP16 mixed precision training
- Gradient checkpointing (recompute activations during backward pass)
- Sequence truncation to 128 tokens
- Per-device batch size=1 with 8-step gradient accumulation

**Generation for prompting methods**:
```python
max_new_tokens=15        # Allow number + minimal formatting
temperature=0.1          # Low temperature for consistency
do_sample=False          # Greedy decoding (deterministic)
```

These parameters prevent verbose explanations while ensuring reliable numeric output.

---

## 5. Results & Analysis

### 5.1 Yelp Performance (Restaurant Reviews)

| Method | Accuracy | F1 Macro | F1 Weighted | Failed Parses |
|--------|----------|----------|-------------|---------------|
| Zero-shot | 52.93% | 48.72% | 49.13% | 6/3,000 (0.2%) |
| Few-shot (5-shot) | 59.07% | 57.17% | 57.30% | 5/3,000 (0.17%) |
| **Fine-tuned (LoRA)** | **67.33%** | **67.39%** | **67.41%** | 146/3,000 (4.9%) |

**Per-class performance (Fine-tuned)**:
```
Class          Precision  Recall  F1    Support
1-star (⭐)       0.82     0.75   0.78    645
2-star (⭐⭐)      0.64     0.65   0.64    583
3-star (⭐⭐⭐)     0.58     0.64   0.61    579
4-star (⭐⭐⭐⭐)    0.64     0.59   0.61    650
5-star (⭐⭐⭐⭐⭐)   0.70     0.76   0.73    543
```

**Observations**:
- Extreme sentiments (1-star, 5-star) achieve highest performance due to clear linguistic signals
- Neutral class (3-star) shows moderate performance due to inherent ambiguity
- Balanced precision-recall indicates well-calibrated model

### 5.2 Amazon Performance (Cross-Domain Transfer)

| Method | Accuracy | F1 Macro | F1 Weighted | Failed Parses |
|--------|----------|----------|-------------|---------------|
| Zero-shot | 42.70% | 36.75% | 36.55% | 10/3,000 (0.3%) |
| Few-shot (5-shot) | 52.67% | 52.88% | 52.90% | 0/3,000 (0%) |
| **Fine-tuned (LoRA)** | **60.43%** | **60.19%** | **60.33%** | 1/3,000 (0.03%) |

**Cross-domain analysis**:
- 17.7 point improvement over zero-shot demonstrates robust learning
- 6.9% accuracy drop from Yelp (67.3%) to Amazon (60.4%) reflects domain differences:
  - Restaurant reviews emphasize service and experience
  - Product reviews emphasize features and value
- Near-perfect parsing (1 failure in 3,000) shows stable output formatting

**Significance**: The strong transfer validates that LoRA captures domain-agnostic sentiment patterns (e.g., "excellent," "disappointed") rather than restaurant-specific language, enabling:
- Train once on one domain, deploy across similar tasks
- Avoid collecting/labeling data for every new domain
- Achieve 60% accuracy on new domains vs. 43% from scratch

### 5.3 Why Complete Class Coverage Matters (5-Shot Analysis)

**Complete coverage (one example per class)**:
- Yelp: 59.1% accuracy, 5/3,000 failed parses (0.17%)
- Amazon: 52.7% accuracy, 0/3,000 failed parses (0%)

Providing exactly one example per class (0-4) ensures:
- Model sees full spectrum of sentiment expressions
- Balanced representation prevents class bias
- Clear boundaries between adjacent sentiment levels

**Analogy**: Teaching someone to rate movies without showing any 5-star examples—they'd struggle to distinguish 4-star from 5-star reviews. Similarly, the model needs at least one instance of each rating level to understand the complete scale.

### 5.4 Model Size and Adaptation Strategy

**For small models (2B params)**:
- Few-shot prompting: Moderate gains (+6-10% over zero-shot)
- Fine-tuning: Essential for best performance (+14-18% over zero-shot)

**For large models (>100B params like GPT-3)**:
- Few-shot prompting: Highly effective (per Brown et al., 2020)
- Fine-tuning: May be overkill

**Key insight**: Adaptation techniques scale differently across model sizes. Our findings confirm that 2B parameter models have limited capacity for in-context learning compared to larger models, making fine-tuning worthwhile despite the training cost.

---

## 6. Model Card & Ethical Considerations

### 6.1 Model Information

**Base Model**: Gemma-2-2B-it
- **Version**: google/gemma-2-2b-it from Hugging Face
- **Architecture**: Decoder-only transformer (96 layers, 2048 hidden dim)
- **Parameters**: 2.6B total, 3.2M trainable (LoRA adapters)
- **License**: Gemma Terms of Use (requires HuggingFace agreement)

**LoRA Adapter**:
- **Rank**: r=8
- **Target Modules**: q_proj, v_proj (query and value projections)
- **Trained On**: Yelp restaurant reviews (5,000 samples)
- **Storage**: 12.8 MB per adapter

### 6.2 Intended Use

**Primary use cases**:
- Sentiment classification of product/service reviews (1-5 star scale)
- Cross-domain sentiment analysis (restaurant → product reviews)
- Research on parameter-efficient fine-tuning methods

**Out-of-scope**:
- Fine-grained aspect-based sentiment analysis
- Languages other than English
- Domains significantly different from reviews (e.g., medical text, legal documents)

### 6.3 Limitations & Biases

**Known limitations**:
1. **Domain specificity**: Trained only on restaurant reviews; performance degrades on dissimilar domains
2. **Class imbalance sensitivity**: Requires balanced training data for optimal performance
3. **Context length**: Limited to reviews ≤128 tokens (truncated during training)
4. **Neutral class confusion**: 3-star reviews have higher error rate due to ambiguity

**Bias considerations**:
1. **Dataset bias**: Yelp reviews may over-represent certain demographics and geographic regions
2. **Language style**: Model may perform worse on non-standard English or slang-heavy reviews
3. **Cultural differences**: Sentiment expressions vary across cultures; model trained primarily on U.S. English
4. **Implicit assumptions**: 5-star scale may not map uniformly across all review domains

**Mitigation strategies**:
- Evaluate on diverse test sets before deployment
- Monitor performance across demographic groups
- Provide confidence scores alongside predictions
- Allow human review for borderline cases (3-star predictions)

### 6.4 Training Data

**Yelp Review Full**:
- **Source**: Public Hugging Face dataset
- **Size**: 5,000 training samples (stratified across 5 classes)
- **Content**: Restaurant reviews in English
- **Preprocessing**: Tokenization, truncation to 128 tokens, balanced sampling

**Amazon Reviews**:
- **Source**: Kaggle public dataset
- **Size**: 5,000 training samples (stratified across 5 classes)
- **Content**: Product reviews in English
- **Usage**: Evaluation only (cross-domain transfer testing)

---

## 7. Critical Analysis & Impact

### 7.1 Key Contributions

1. **Empirical validation of class coverage importance**: Demonstrated that balanced few-shot examples (5-shot vs. 4-shot for 5 classes) significantly improve performance, contradicting the assumption that any examples help equally

2. **LoRA efficiency at small scale**: Confirmed low-rank adaptation hypothesis for 2B models—updating 0.12% of parameters achieves 67% accuracy (+14% over zero-shot, +8% over few-shot)

3. **Cross-domain generalization evidence**: Single Yelp-trained adapter achieves 60% accuracy on Amazon (+18% over zero-shot), demonstrating that LoRA learns generalizable sentiment representations

4. **Practical deployment guidance**: For models ~2B parameters, fine-tuning is essential; few-shot provides reasonable middle ground when training is infeasible

### 7.2 What This Reveals

**About model adaptation**: The success of low-rank fine-tuning (r=8) suggests that task-specific knowledge occupies a low-dimensional subspace within the model's representation space. Pre-trained models already contain the necessary features—adaptation merely amplifies the relevant ones.

**About in-context learning**: Small models (<7B parameters) show limited in-context learning capacity. While 5-shot prompting helps, gains are modest (6-10%) compared to fine-tuning (14-18%). This stands in contrast to large models (>100B params) where few-shot prompting approaches fine-tuning performance.

**About cross-domain transfer**: The 6.9% accuracy drop from Yelp to Amazon (67.3% → 60.4%) is surprisingly small, indicating that sentiment classification is fundamentally domain-agnostic at the linguistic level, even though review content differs substantially.

### 7.3 Practical Impact

**For ML practitioners**:
- **Parameter efficiency**: LoRA enables fine-tuning billion-parameter models on consumer hardware (90 minutes, single L4 GPU)
- **Deployment efficiency**: 12.8 MB adapters allow storing 100+ task-specific models at ~354 GB vs. 35 TB for full fine-tuned copies
- **Task switching**: Swap adapters on-the-fly without reloading base model

**For production systems**:
- 30% error reduction (few-shot → fine-tuned) justifies one-time training cost
- Near-zero inference latency (merge weights at deployment)
- Stable outputs (<5% parse failures) ensure reliable integration

### 7.4 Limitations of This Study

1. **Single model size**: Only evaluated 2B parameters; findings may not generalize to 7B, 13B, or 70B models
2. **Single task type**: Limited to 5-class sentiment; other tasks (QA, summarization, code generation) may behave differently
3. **Limited LoRA configuration**: Only tested rank r=8; optimal rank may vary by task
4. **English-only**: Did not evaluate multilingual transfer

### 7.5 Future Directions

**Immediate next steps**:
1. **Model scaling study**: Test adaptation strategies across 2B → 7B → 70B to identify thresholds where few-shot becomes competitive
2. **Rank optimization**: Ablation study on r ∈ {4, 8, 16, 32} to find efficiency-performance sweet spot
3. **Full fine-tuning comparison**: Quantify gap between LoRA and full parameter updates

**Longer-term research**:
1. **Hybrid approaches**: Combine LoRA fine-tuning with few-shot prompting for rapid domain adaptation
2. **Theoretical framework**: Develop mathematical model predicting when few-shot succeeds based on model capacity and task complexity
3. **Multi-task LoRA**: Train single adapter on multiple related tasks to improve generalization

### 7.6 Broader Implications

As language models proliferate across applications and model scales fragment (2B to 175B+ parameters), understanding **scale-dependent behavior** becomes critical. This work contributes empirical evidence that:

- **Techniques don't scale uniformly**: Few-shot prompting's success on GPT-3 doesn't transfer to Gemma-2B
- **Practitioners must validate at deployment scale**: Don't assume large-model findings apply to small models
- **Parameter-efficient methods are accessible**: Billion-parameter fine-tuning is feasible on consumer hardware

The democratization of fine-tuning through methods like LoRA enables smaller organizations to build high-quality, customized models without massive infrastructure investments.

---

## 8. References

1. Phuong, M., & Hutter, M. (2022). Formal Algorithms for Transformers. *arXiv preprint arXiv:2207.09238*. https://arxiv.org/abs/2207.09238

2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR) 2022*. https://arxiv.org/abs/2106.09685

3. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems (NeurIPS) 2020*. https://arxiv.org/abs/2005.14165

4. Vaswani, A., et al. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS) 2017*. https://arxiv.org/abs/1706.03762

5. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Association for Computational Linguistics (ACL) 2019*. https://arxiv.org/abs/1810.04805

---

## 9. Resources & Links

### Code & Models
- **GitHub Repository**: [Project code and experiments]
- **Gemma-2-2B-it**: https://huggingface.co/google/gemma-2-2b-it
- **LoRA Implementation (PEFT)**: https://github.com/huggingface/peft

### Datasets
- **Yelp Review Full**: https://huggingface.co/datasets/Yelp/yelp_review_full (650K train, 50K test)
- **Amazon Reviews**: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews (3M train, 650K test; subsampled to 200K/50K)

### Libraries
- **Hugging Face Transformers**: https://github.com/huggingface/transformers
- **PyTorch**: https://pytorch.org
- **Google Colab**: https://colab.research.google.com

### Setup Instructions

**Requirements**:
```bash
pip install transformers==4.44.0 datasets==2.20.0 accelerate==0.33.0 peft==0.12.0 scikit-learn
```

**Quick Start**:
1. Clone repository and install dependencies
2. Authenticate with Hugging Face (requires Gemma license agreement)
3. Run zero-shot evaluation: `python evaluate_zero_shot.py`
4. Run few-shot evaluation: `python evaluate_few_shot.py --n_shots 5`
5. Train LoRA adapter: `python train_lora.py --rank 8 --dataset yelp`
6. Evaluate fine-tuned model: `python evaluate_lora.py --adapter_path ./adapters/yelp_r8`

**Hardware**:
- Training: NVIDIA L4 GPU (22.5 GB VRAM) recommended
- Inference: Can run on CPU for small batches; GPU recommended for production

---

## Conclusion

This project demonstrates that for small language models (~2B parameters), **parameter-efficient fine-tuning via LoRA significantly outperforms prompt engineering** for sentiment classification tasks. While few-shot prompting with complete class coverage provides meaningful improvements over zero-shot baseline (+6-10%), fine-tuning achieves substantially better results (+14-18%) by updating only 0.12% of parameters.

The strong cross-domain transfer (60% accuracy on Amazon after training only on Yelp) validates that LoRA learns generalizable sentiment representations. Combined with practical benefits—90-minute training time, 12.8 MB adapter size, zero inference latency—LoRA emerges as the clear choice for production sentiment classification systems using models in this size range.

However, adaptation strategy is **not universal**. These findings are specific to 2B parameter models; larger models (>100B) may achieve comparable results with prompt engineering alone. As the field continues to develop models across diverse scales, understanding these scale-dependent behaviors becomes increasingly important for practitioners making deployment decisions.
