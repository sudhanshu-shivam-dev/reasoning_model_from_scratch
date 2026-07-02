# 00 · The Big Picture — How a Reasoning Model Gets Built

> **Read this first.** Every other module zooms into one box of the diagrams on
> this page. If you ever feel lost later, come back here.

---

## 1. The one-paragraph version

A **reasoning model** is a large language model (LLM) that has been encouraged —
by prompting, by inference-time tricks, or by training — to produce
**intermediate steps** before its final answer. The book builds one from scratch
in three stages: **(1)** get a small pretrained LLM (Qwen3 0.6B) generating text
and **measure** how well it solves math problems whose answers can be checked
automatically; **(2)** improve accuracy **without training** by spending more
compute at inference time (sampling many answers, voting, self-refining);
**(3)** improve the model itself **with training**, using reinforcement learning
(GRPO) driven by those same automatic answer checks, and alternatively by
distilling a stronger teacher's reasoning traces into the student.

---

## 2. Where reasoning models come from (the full LLM pipeline)

Before this book starts, a lot has already happened. Here is the standard LLM
life cycle — the book picks up at the ★:

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#e0edfb","primaryTextColor":"#0d366b","primaryBorderColor":"#2a78d6","lineColor":"#898781","textColor":"#52514e","edgeLabelBackground":"#f0efec","clusterBkg":"rgba(137,135,129,0.07)","clusterBorder":"#a9a7a0","titleColor":"#898781"},"flowchart":{"nodeSpacing":36,"rankSpacing":44,"curve":"basis","padding":10}}}%%
flowchart LR
    subgraph PRE["Pretraining (not in this book)"]
        D["🌐 Internet-scale text"] --> PT["Next-token prediction<br/>on trillions of tokens"]
        PT --> BASE["Base model<br/>(e.g. Qwen3-0.6B-Base) ★"]
    end

    subgraph POST["Post-training (this book!)"]
        BASE --> ITS["⚡ Inference-time scaling<br/>Ch 4–5 (no weight updates)"]
        BASE --> RL["🏋️ RL with verifiable rewards<br/>GRPO — Ch 6–7"]
        BASE --> SFT["📖 Distillation / SFT<br/>on reasoning traces — Ch 8"]
        ITS --> RM["🧠 Reasoning model"]
        RL --> RM
        SFT --> RM
    end

    EVAL["📏 Evaluation with a math verifier — Ch 3"]
    RM <-->|measure improvement| EVAL
    BASE <-->|measure baseline| EVAL
    classDef data fill:#d9f4e9,stroke:#1baf7a,color:#0b4a33
    classDef model fill:#e6e3f7,stroke:#4a3aa7,color:#251d54
    classDef dec fill:#fdf0d1,stroke:#eda100,color:#6b4a00
    class D data
    class BASE,RM model
    class EVAL dec
```

**Key insight:** the three improvement arrows are *independent and composable*.
You can apply inference-time scaling to an RL-trained model, distill a model and
then run RL on it, and so on. Real systems (DeepSeek-R1, OpenAI o-series) combine
several.

---

## 3. The three levers, compared

| | ⚡ Inference-time scaling (Ch 4–5) | 🏋️ RL / GRPO (Ch 6–7) | 📖 Distillation (Ch 8) |
|---|---|---|---|
| **Changes model weights?** | ❌ No | ✅ Yes | ✅ Yes |
| **Needs training data?** | ❌ No | Questions + checkable answers | Teacher's reasoning traces |
| **Cost paid at…** | Every inference (forever) | Training time (once) | Training time (once) |
| **Core mechanism** | Sample more / think longer / revise | Trial & error + reward for correct answers | Imitate a stronger model |
| **Ceiling** | Limited by what the base model *can* sample | Can discover *new* behaviors | Limited by the teacher |
| **Fragility** | Robust, simple | Hyperparameter-sensitive, can collapse | Robust, simple |

```mermaid
%%{init: {"theme":"base","themeVariables":{"quadrant1Fill":"#dcf3dc","quadrant2Fill":"#e0edfb","quadrant3Fill":"#f0efec","quadrant4Fill":"#fdf0d1","quadrant1TextFill":"#52514e","quadrant2TextFill":"#52514e","quadrant3TextFill":"#52514e","quadrant4TextFill":"#52514e","quadrantPointFill":"#2a78d6","quadrantPointTextFill":"#52514e","quadrantXAxisTextFill":"#898781","quadrantYAxisTextFill":"#898781","quadrantTitleFill":"#898781","quadrantInternalBorderStrokeFill":"#c3c2b7","quadrantExternalBorderStrokeFill":"#a9a7a0"}}}%%
quadrantChart
    title Cost vs. capability gain of each method
    x-axis Low implementation effort --> High implementation effort
    y-axis Small gain --> Large gain
    "CoT prompting (Ch 4)": [0.15, 0.35]
    "Majority voting (Ch 4)": [0.3, 0.55]
    "Self-refinement (Ch 5)": [0.4, 0.45]
    "Distillation (Ch 8)": [0.55, 0.7]
    "GRPO (Ch 6)": [0.8, 0.75]
    "GRPO + fixes (Ch 7)": [0.9, 0.85]
```

---

## 4. The experimental loop you'll repeat all book long

Everything in the book is one instance of this loop:

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#e0edfb","primaryTextColor":"#0d366b","primaryBorderColor":"#2a78d6","lineColor":"#898781","textColor":"#52514e","edgeLabelBackground":"#f0efec","clusterBkg":"rgba(137,135,129,0.07)","clusterBorder":"#a9a7a0","titleColor":"#898781"},"flowchart":{"nodeSpacing":36,"rankSpacing":44,"curve":"basis","padding":10}}}%%
flowchart TB
    A["1️⃣ Take a model"] --> B["2️⃣ Generate answers to<br/>math questions (Ch 2)"]
    B --> C["3️⃣ Extract final answer<br/>and verify it (Ch 3)"]
    C --> D["4️⃣ Compute accuracy"]
    D --> E{"Try an improvement"}
    E -->|"prompt / sample differently (Ch 4–5)"| B
    E -->|"update weights (Ch 6–8)"| A
    classDef model fill:#e6e3f7,stroke:#4a3aa7,color:#251d54
    classDef dec fill:#fdf0d1,stroke:#eda100,color:#6b4a00
    class A model
    class C,E dec
```

This is why **Chapter 3 (evaluation) is the keystone chapter**: without a
trustworthy, automatic verifier you can neither measure progress *nor* train
with reinforcement learning (the verifier literally *is* the reward function in
Ch 6).

---

## 5. The cast of characters

| Thing | Role in the book |
|---|---|
| **Qwen3-0.6B (base)** | The small pretrained LLM everything is built on — small enough to run on a laptop |
| **Qwen3-0.6B (reasoning/"thinking")** | The official post-trained variant, used for comparison |
| **MATH-500-style problems** | Math word problems with a single checkable numeric/symbolic answer |
| **The verifier** | Code that extracts the final answer (e.g. from `\boxed{...}`) and checks equality with the reference |
| **The generate function** | The token-by-token sampling loop from Ch 2 — reused *everywhere* |
| **GRPO** | The RL algorithm (Group Relative Policy Optimization) that turns verifier output into weight updates |

---

## 6. Mental models to carry with you

### 🎰 An LLM is a next-token slot machine
The model outputs a probability for every token in the vocabulary; *decoding
strategy* decides how to pick one. Greedy = always take the most likely.
Sampling = spin the wheel. Everything in Ch 4 is about pulling this lever
multiple times and aggregating.

### 🧗 Reasoning tokens are footholds
Each intermediate step conditions the next prediction. Writing "Let me compute
17 × 23 step by step" literally changes the probability distribution of every
subsequent token toward correct arithmetic. That's why *more tokens can mean
more accuracy* — the model is buying itself easier subproblems.

### 🏫 RL is a teacher who only grades the final exam
GRPO never says *which step* was wrong. It only rewards a whole answer that
ended correctly. The gradient math (Ch 6) spreads that single scalar across all
tokens of the answer. This weak-but-honest signal is enough — that's the
surprising empirical result behind DeepSeek-R1-Zero and this book's Part 3.

### 👨‍🏫 Distillation is a teacher who shows worked solutions
Instead of grading, the teacher writes out full reasoning traces and the student
imitates them token by token (plain cross-entropy). Simpler signal, faster
learning, but the student can't exceed the teacher.

---

## 7. Notation used across all modules

| Symbol | Meaning |
|---|---|
| `x` | prompt / question tokens |
| `y = (y₁ … y_T)` | generated response tokens |
| `π_θ(y_t \| x, y_<t)` | probability the model (policy) with weights θ assigns to token `y_t` given everything before it |
| `r` or `R` | reward (1 = correct answer, 0 = wrong, in the simplest setup) |
| `A` | advantage — how much better a response is than a baseline |
| `G` | group size in GRPO (number of sampled answers per question) |
| `ε` | PPO/GRPO clipping range |
| `β` | KL-penalty coefficient |

---

## ✅ Self-check

<details><summary>1. Why does the book implement evaluation (Ch 3) before any improvement method?</summary>

Because every improvement — prompting tricks, RL, distillation — is judged by
the same accuracy metric, and because the RL reward in Ch 6 *is* the verifier.
No verifier → no measurement → no training signal.
</details>

<details><summary>2. Name the key trade-off between inference-time scaling and RL training.</summary>

Inference-time scaling needs no training and is easy to implement, but you pay
extra compute on *every single query, forever*, and you can only surface
abilities the base model already has. RL costs a lot once (training) but the
improved weights make every future query cheaper/better, and it can create new
behaviors.
</details>

<details><summary>3. What does "verifiable reward" mean and why is math the ideal domain for it?</summary>

A reward that can be computed automatically by checking the model's final
answer against a known ground truth. Math (and code, via unit tests) is ideal
because equality of a final numeric/symbolic answer is cheap and objective to
check — no human judgment or learned reward model needed.
</details>

---

## 🔗 Where to go next

- New to transformers → [App C+D · Qwen3 Architecture](appendix_qwen3_architecture.md) first, then [Ch 1](ch01_understanding_reasoning_models.md).
- Otherwise → [Ch 1 · Understanding Reasoning Models](ch01_understanding_reasoning_models.md).
