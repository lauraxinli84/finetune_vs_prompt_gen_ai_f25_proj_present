# Prompt Engineering vs. Fine-Tuning for Sentiment Classification

**Laura Li** | Generative AI Models in Theory and Practice | Vanderbilt University | December 4, 2025

---

## 🎯 The Central Question

**When adapting LLMs to domain-specific tasks, should you use prompt engineering or fine-tuning?**

This project compares three adaptation strategies for 5-star sentiment classification:
- **Zero-shot prompting** (baseline)
- **Few-shot prompting** (4 examples)
- **LoRA fine-tuning** (parameter updates)

---

## 📊 Quick Results Summary

| Dataset | Zero-Shot | Few-Shot | Fine-Tuned | Improvement |
|---------|-----------|----------|------------|-------------|
| **Yelp** | 52.9% | 44.7% ❌ | **67.3%** ✅ | +14.4% |
| **Amazon** | 42.7% | 24.3% ❌ | **60.4%** ✅ | +17.7% |

**Key Finding**: For Gemma-2-2B, fine-tuning vastly outperforms prompting strategies, and few-shot prompting actually *hurts* performance.

---

## 🔬 Methodology

### Model & Architecture
- **Model**: Google Gemma-2-2B-it (2.5B parameters)
- **Architecture**: Decoder-only transformer (GPT-style)
- **Key features**: Causal masking, autoregressive prediction, unidirectional attention

### Datasets
- **Yelp Reviews**: Restaurant reviews (1-5 stars)
- **Amazon Reviews**: Product reviews (1-5 stars)
- **Scale**: 9,000 samples each (5k train / 1k val / 3k test)
- **Sampling**: Stratified to ensure balanced classes

### Three Adaptation Approaches

#### 1️⃣ Zero-Shot Prompting
```
Classify the sentiment (1-5):
1 = Very Negative
2 = Negative
3 = Neutral
4 = Positive
5 = Very Positive

Review: {text}
Sentiment (1-5):
```
- **Parameters updated**: 0
- **Inference**: Temperature τ=0.1 for deterministic outputs

#### 2️⃣ Few-Shot Prompting (4 examples)
```
Here are some examples:
Review: "Terrible service..." → Sentiment: 1
Review: "Food was okay..." → Sentiment: 3
Review: "Amazing experience!" → Sentiment: 5
[...4 total examples...]

Now classify this review:
Review: {text}
Sentiment:
```
- **Parameters updated**: 0
- **Strategy**: One example per class (stratified selection)

#### 3️⃣ LoRA Fine-Tuning
- **Method**: Low-Rank Adaptation (LoRA) [Hu et al., 2021]
- **Trainable parameters**: 3.2M (0.12% of total)
- **Target layers**: Query and Value projections in attention
- **Configuration**: rank r=8, α=16, dropout=0.05
- **Training**: 2 epochs, AdamW optimizer, lr=2e-4
- **Hardware**: Google Colab T4 GPU (~90 minutes)

**Mathematical formulation**:
```
h = W₀x + ΔWx = W₀x + BAx
where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r=8 << min(d,k)
```

---

## 📈 Detailed Results

### Yelp Performance

| Method | Accuracy | F1 Macro | Failed Parses |
|--------|----------|----------|---------------|
| Zero-shot | 52.93% | 48.72% | 6 / 3,000 (0.2%) |
| Few-shot | 44.67% | 45.53% | 1,331 / 3,000 (44%) |
| **Fine-tuned** | **67.33%** | **67.39%** | 146 / 3,000 (5%) |

**Per-class breakdown** (Fine-tuned):
```
Class 0 (⭐):      Precision 0.82 | Recall 0.75 | F1 0.78
Class 1 (⭐⭐):     Precision 0.64 | Recall 0.65 | F1 0.64
Class 2 (⭐⭐⭐):    Precision 0.58 | Recall 0.64 | F1 0.61
Class 3 (⭐⭐⭐⭐):   Precision 0.64 | Recall 0.59 | F1 0.61
Class 4 (⭐⭐⭐⭐⭐):  Precision 0.70 | Recall 0.76 | F1 0.73
```

**Observations**:
- Strongest on extreme sentiments (1-star, 5-star)
- Moderate on neutral (3-star)
- Balanced precision/recall across all classes

### Amazon Performance (Cross-Domain Transfer)

| Method | Accuracy | F1 Macro | Failed Parses |
|--------|----------|----------|---------------|
| Zero-shot | 42.70% | 36.75% | 10 / 3,000 (0.3%) |
| Few-shot | 24.33% | 16.11% | 2,718 / 3,000 (91%) |
| **Fine-tuned** | **60.43%** | **60.19%** | 1 / 3,000 (0.03%) |

**Key insight**: Model trained on Yelp generalizes well to Amazon despite domain shift (restaurant → product reviews).

---

## 🚨 Surprising Finding: Few-Shot Failure

Few-shot prompting **degraded performance** dramatically:
- Yelp: -8.3% accuracy
- Amazon: -18.4% accuracy
- Failed to parse 44-91% of outputs

### Why Did Few-Shot Fail?

**1. Cognitive Overload**
- Zero-shot prompt: ~50 tokens
- Few-shot prompt: ~800-1,000 tokens (4 examples)
- Small model (2B params) struggled with long context

**2. Output Format Chaos**
- Zero-shot: Direct output → "3" or "4"
- Few-shot: Verbose → "Based on the review, I would rate this as positive because..."
- Parser couldn't extract ratings from explanations

**3. Attention Dilution**
- Attention spread across examples instead of focusing on query
- Smaller models lack capacity to separate "examples" from "task"

**4. Contrast with GPT-3**
- GPT-3 175B excels at few-shot learning [Brown et al., 2020]
- Gemma-2B has 87× fewer parameters
- **Hypothesis**: Model size threshold (~7B+?) where few-shot becomes beneficial

**Conclusion**: For smaller models, zero-shot prompting is more reliable than few-shot.

---

## 💡 Key Insights

### 1. Model Size Determines Optimal Strategy
- **Large models (GPT-3 175B)**: Few-shot prompting works well
- **Small models (Gemma-2B)**: Few-shot fails, fine-tuning essential
- **Implication**: Can't generalize findings across model scales

### 2. LoRA Efficiency Validated
- Only 0.12% of parameters needed for adaptation
- 67% accuracy (Yelp) with r=8 rank
- Confirms "low intrinsic rank" hypothesis from LoRA paper

### 3. Cross-Domain Transfer Success
- Single adapter trained on Yelp → 60% on Amazon
- +17.7% over zero-shot baseline
- Suggests sentiment patterns are domain-agnostic

### 4. Practical Trade-offs
- **Fine-tuning cost**: ~90 minutes on T4 GPU, one-time
- **Prompt engineering cost**: Zero training, but lower accuracy
- **Deployment**: LoRA adapters are small (12.8 MB), swappable

---

## 🔧 Implementation Details

### Technologies
- **Framework**: PyTorch + Hugging Face Transformers
- **Training**: PEFT library (LoRA implementation)
- **Hardware**: Google Colab T4 GPU (16GB VRAM)
- **Optimizations**: FP16 precision, gradient checkpointing, gradient accumulation

### Memory Efficiency Tricks
```python
# Fitting 2.5B model on 16GB GPU
per_device_train_batch_size=1      # Minimal batch
gradient_accumulation_steps=8      # Effective batch=8
fp16=True                          # Mixed precision (2× speedup)
gradient_checkpointing=True        # Trade compute for memory
max_length=128                     # Truncate long reviews
```

### Generation Settings
```python
# Deterministic, short outputs for reliable parsing
max_new_tokens=10        # Force direct answers
temperature=0.1          # Nearly deterministic
do_sample=True           # Temperature sampling
```

### Robust Output Parsing
```python
def extract_rating_improved(text):
    # 1. Pattern match: "Sentiment: X"
    # 2. Find any digit 1-5
    # 3. Check first 100 chars
    # Returns: rating or None
```
This improved parser reduced failed parses from 44% → 5% for fine-tuned model.

---

## 📚 Theoretical Connections

### Formal Algorithms for Transformers [Phuong & Hutter, 2022]

**Algorithm 10**: Decoder-only transformer architecture
- Gemma implements unidirectional masked self-attention
- Causal masking: token t only attends to tokens 1:t

**Algorithm 13**: DTraining (parameter updates)
- LoRA fine-tuning updates subset Θ ⊂ Φ₀ via gradient descent
- Loss: Cross-entropy on next-token prediction

**Algorithm 14**: DInference (prompting)
- Zero-shot and few-shot use inference without parameter updates
- Temperature τ=0.1 for deterministic sampling

### Low-Rank Hypothesis [Hu et al., 2021]

> "The change in weights during model adaptation has a low intrinsic rank"

**Validated by our results**:
- Rank r=8 sufficient for 67% accuracy (vs 53% baseline)
- Cross-domain transfer confirms low-rank updates capture generalizable features
- Suggests adaptation amplifies existing features rather than learning new ones

---

## 🎓 Practical Implications

### For Practitioners

**When to use Zero-Shot**:
- Quick prototyping, no training budget
- Small models (<7B params)
- Tasks with clear instructions

**When to use Fine-Tuning**:
- Domain-specific terminology
- Need consistent output format
- Have 50-100 minutes for training
- Want maximum accuracy

**When to avoid Few-Shot** (for small models):
- Limited by model capacity
- Need reliable output parsing
- Better to use zero-shot + fine-tuning pipeline

### Cost-Benefit Analysis
- **LoRA training**: 90 minutes, 16GB GPU, one-time cost
- **Accuracy gain**: +14-17% absolute improvement
- **Deployment**: Same inference speed as base model (merge adapters)
- **Storage**: 12.8 MB per adapter (minimal overhead)

### Deployment Strategy
1. Start with zero-shot for rapid iteration
2. Collect domain-specific data (5,000 samples)
3. Fine-tune with LoRA (one afternoon)
4. Deploy adapter alongside base model
5. Swap adapters for different domains/tasks

---

## 🔮 Future Work

### Immediate Extensions
- **Larger models**: Test Gemma-7B, Llama-3-8B to find few-shot threshold
- **Rank ablation**: Compare r=4, 16, 32 to optimize efficiency
- **Full fine-tuning**: Measure marginal gains vs. LoRA
- **Other tasks**: Classification, NER, summarization

### Research Questions
- What model size enables effective few-shot learning?
- Can prompt engineering be salvaged with better templates?
- Do findings transfer to instruction-tuned vs. base models?
- How does domain similarity affect transfer learning?

---

## 📦 Repository Structure

```
notebook.ipynb                  # Single Colab notebook
├── Setup & Dependencies
├── Data Loading (Yelp, Amazon)
├── Stratified Sampling (5k/1k/3k splits)
├── Zero-Shot Evaluation
├── Few-Shot Evaluation
├── LoRA Fine-Tuning (~90 min)
└── Evaluation & Results Export
```

**All code in one place** for reproducibility. Dataset splits and trained adapters saved to Google Drive.

---

## 🏁 Conclusions

### Main Takeaways

1. **Fine-tuning wins decisively** for small models on domain-specific tasks
   - +14-17% accuracy over zero-shot
   - Consistent, parseable outputs
   
2. **Few-shot prompting can hurt** performance on smaller models
   - Model size matters for in-context learning
   - Don't assume GPT-3 findings generalize
   
3. **LoRA is efficient and practical**
   - 0.12% parameters, 90 minutes training
   - Strong cross-domain transfer
   
4. **Adaptation strategy depends on model size**
   - <7B params: Fine-tune
   - >100B params: Few-shot prompting
   - No universal best practice

### The Answer to Our Central Question

**For Gemma-2-2B and similar small models**: Fine-tuning with LoRA is the clear winner. The computational cost is justified by significant accuracy gains and reliable output formatting.

**But**: This answer changes with model scale. Larger models may benefit more from prompt engineering. Always consider the size-strategy interaction.

---

## 📖 References

1. Phuong, M., & Hutter, M. (2022). Formal Algorithms for Transformers. *arXiv preprint arXiv:2207.09238*.
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
3. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS 2020*.

---

**Questions?**
