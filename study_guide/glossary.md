# Glossary — Every Term, One Breath Each

> Sorted by theme, not alphabet, so neighboring terms reinforce each other.
> Chapter references point to the module where the term is developed.

---

## Reasoning fundamentals

| Term | Definition | Where |
|---|---|---|
| **Reasoning (operational def.)** | Generating intermediate steps before a final answer such that the answer is more likely correct. | Ch 1 |
| **Chain of thought (CoT)** | The intermediate-step text itself; also the prompting technique that elicits it. | Ch 1, 4 |
| **Thinking tokens / `<think>` block** | Delimited region of a response holding the deliberation, separated from the user-facing answer. | Ch 2, 8 |
| **Overthinking** | Producing far more reasoning tokens than a problem needs, wasting cost and sometimes flipping correct answers. | Ch 1, 8 |
| **Emergent reasoning behaviors** | Self-checking, backtracking, "aha" moments arising from RL without being demonstrated in training data (R1-Zero result). | Ch 1, 6 |

## Generation & decoding

| Term | Definition | Where |
|---|---|---|
| **Token / tokenization (BPE)** | Subword units from byte-pair encoding; the model's alphabet (~151k for Qwen3). | Ch 2 |
| **Autoregressive generation** | Predict a token, append it, repeat — each output becomes input. | Ch 2 |
| **Logits** | Pre-softmax scores over the vocabulary, one per possible next token. | Ch 2 |
| **Greedy decoding** | Always pick the argmax token; deterministic. | Ch 2 |
| **Temperature (T)** | Logit divisor before softmax: <1 sharpens, >1 flattens the distribution. | Ch 4 |
| **Top-k / top-p sampling** | Restrict sampling to the k best tokens / the smallest set with cumulative probability p. | Ch 4 |
| **Chat template** | The exact special-token format (`<\|im_start\|>` …) a chat model was trained on; mandatory at inference. | Ch 2 |
| **EOS / stop token** | Token whose emission ends generation (`<\|im_end\|>` for Qwen3 turns). | Ch 2 |

## Evaluation

| Term | Definition | Where |
|---|---|---|
| **Verifier** | Program that extracts a model's final answer and checks it against ground truth. | Ch 3 |
| **Verifiable reward** | Reward computed by such a program — objective, free, automatic. | Ch 3, 6 |
| **Answer extraction** | Parsing the completion for the final answer (last `\boxed{…}`, brace-matched). | Ch 3 |
| **maj@N** | Accuracy of the majority-voted answer over N samples (deployable). | Ch 4 |
| **pass@N** | Fraction of problems where ≥1 of N samples is correct (capability upper bound). | Ch 4 |
| **Contamination** | Benchmark items leaking into training data, inflating scores. | App F |
| **LLM-as-a-judge** | Using a strong LLM with a rubric to grade open-ended outputs. | App F |
| **Elo / arena rating** | Pairwise human-preference ranking aggregated across many battles. | App F |
| **Perplexity** | exp(mean next-token cross-entropy); "effective branching factor" of the model's uncertainty. | App F |
| **Goodhart's law / reward hacking** | Optimizing a proxy metric until it stops measuring what you meant; exploiting verifier gaps. | Ch 3, 6 |

## Inference-time scaling

| Term | Definition | Where |
|---|---|---|
| **Inference-time (test-time) scaling** | Buying accuracy with more compute at answer time, weights frozen. | Ch 1, 4, 5 |
| **Parallel scaling** | Independent samples aggregated at the end (voting). | Ch 4 |
| **Sequential scaling** | Later compute conditioned on earlier output (refinement). | Ch 5 |
| **Self-consistency** | Sample N CoTs with temperature, majority-vote the final answers. | Ch 4 |
| **Self-refinement** | Generate → critique → revise loop with the same model. | Ch 5 |
| **Intrinsic self-correction** | Refinement using only the model's own judgment — empirically weak. | Ch 5 |

## Reinforcement learning

| Term | Definition | Where |
|---|---|---|
| **Policy (π_θ)** | The LLM viewed as an action-chooser: a distribution over next tokens. | Ch 6 |
| **Rollout / trajectory** | One sampled complete response (an RL episode). | Ch 6 |
| **RLVR** | Reinforcement Learning with Verifiable Rewards — RL where the grader is a program. | Ch 6 |
| **RLHF** | RL from Human Feedback — reward comes from a *learned* preference model; aligns style, not correctness. | Ch 1, 6 |
| **Policy gradient / REINFORCE** | Increase log-probability of sampled outputs proportionally to their reward. | Ch 6 |
| **Baseline** | Quantity subtracted from reward to reduce variance without bias. | Ch 6 |
| **Advantage (A)** | Reward minus baseline: "better than expected?" — sign decides push up vs. down. | Ch 6 |
| **Critic / value model** | Learned baseline network in PPO; what GRPO eliminates. | Ch 6 |
| **GRPO** | Group Relative Policy Optimization: advantages from normalizing rewards within a G-sample group per question. | Ch 6 |
| **Group (G)** | The set of responses sampled per question whose reward statistics form the baseline. | Ch 6 |
| **Probability ratio (ρ)** | π_new/π_old per token; measures how far the update has already moved. | Ch 6 |
| **Clipped surrogate objective** | min(ρA, clip(ρ)A) — zeroes gradients once ρ leaves [1−ε, 1+ε]. | Ch 6 |
| **Reference model / KL penalty (β)** | Frozen initial model + divergence penalty anchoring the policy; often dropped in RLVR. | Ch 6, 7 |
| **Format reward** | Small bonus for correct output structure (e.g. emitting `\boxed{}`) that bootstraps learning. | Ch 6 |
| **Zero-variance group** | All-correct or all-wrong group → std=0 → zero advantages → no gradient. | Ch 6, 7 |
| **Dynamic sampling** | Discarding zero-variance groups and resampling until the batch is informative (DAPO). | Ch 7 |
| **Clip-higher** | Asymmetric clipping (ε_high > ε_low) giving low-probability tokens room to grow; fights entropy collapse. | Ch 7 |
| **Entropy collapse** | Policy distribution sharpening until exploration dies. | Ch 7 |
| **Token-level loss** | Normalizing loss over all batch tokens (not per response) to remove length bias. | Ch 7 |
| **Overlong filtering / soft punishment** | Masking or gently penalizing truncated responses instead of grading them as failures. | Ch 7 |
| **Policy collapse** | Degenerate optimization outcome: repetitive/broken outputs, exploded KL. | Ch 6, 7 |

## Distillation & SFT

| Term | Definition | Where |
|---|---|---|
| **Supervised fine-tuning (SFT)** | Next-token cross-entropy training on curated input→output pairs. | Ch 8 |
| **Distillation (sequence-level)** | Student trained on teacher-*generated* text; the reasoning-model sense of the word. | Ch 8 |
| **Knowledge distillation (classic)** | Student matches teacher's output *distribution* (soft logits); the original Hinton sense. | Ch 8 |
| **Teacher / student** | Strong source model / small model being trained. | Ch 8 |
| **Reasoning trace** | Full teacher output (thinking + answer) used as training target. | Ch 8 |
| **Rejection sampling** | Keep only verifier-approved traces for the dataset. | Ch 8 |
| **Loss masking** | Computing loss only on response tokens, never prompt tokens (also in GRPO). | Ch 8, 6 |
| **Catastrophic forgetting** | Narrow fine-tuning eroding general abilities. | Ch 8 |

## Architecture (App C+D)

| Term | Definition |
|---|---|
| **Residual stream** | The d-dim vector per token that blocks read from and additively write to. |
| **Pre-norm** | Normalization *before* each sublayer; stabilizes deep training. |
| **RMSNorm** | LayerNorm minus mean-centering: rescale by root-mean-square, learned gain. |
| **RoPE** | Rotary position embedding — rotate Q/K by position-dependent angles; encodes *relative* offsets. |
| **QK-norm** | RMSNorm on queries/keys before RoPE; bounds attention logits (Qwen3 tweak). |
| **Causal mask** | −∞ on future positions so token t attends only backwards. |
| **MHA / GQA** | Multi-head attention / grouped-query attention (several Q heads share one KV head; shrinks KV cache). |
| **SwiGLU** | Gated FFN: `down(silu(gate(x)) ⊙ up(x))`; the modern MLP block. |
| **Weight tying** | Output head reuses the embedding matrix (saves ~155M params at 0.6B). |

## Systems (App E, G)

| Term | Definition |
|---|---|
| **KV cache** | Stored keys/values of past tokens so each decode step processes only the new token. |
| **Prefill vs. decode** | Prompt processed in parallel (compute-bound) vs. token-by-token generation (bandwidth-bound). |
| **Continuous batching** | Refilling finished batch slots with new requests mid-flight (serving systems). |
| **PagedAttention** | Paged, non-contiguous KV-cache memory management (vLLM); "virtual memory for the cache". |
| **Streaming / SSE** | Emitting tokens to the client as generated; Server-Sent Events is the usual transport. |
