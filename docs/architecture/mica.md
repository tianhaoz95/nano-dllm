# Minor Component Adaptation (MiCA)

**MiCA** is the primary Parameter-Efficient Fine-Tuning (PEFT) strategy used in this project. Unlike LoRA, which focuses on the dominant singular directions (largest singular values), MiCA targets the **minor components** (smallest singular values) of the weight matrices.

## Why MiCA?

In a pretrained AR model, the major singular directions are already heavily optimized for causal next-token prediction. By adapting only the minor directions:
1.  **Preservation**: We avoid interfering with the model's core causal capabilities.
2.  **Specialization**: We use the "untapped" capacity of the weights to learn the new bidirectional patterns required by diffusion.

## Implementation Details

*   **Target Modules**: Currently applied to `q_proj` and `v_proj` in the attention layers.
*   **Rank ($r$)**: Default value is **32**.
*   **Alpha ($\alpha$)**: Default value is **32.0**.
*   **Trainable Parameters**: Approximately **0.36%** of the base model.

### SVD Precomputation
Before training starts, we perform a one-time SVD on the base model weights to identify the minor singular vectors:
$$ W = U \Sigma V^T $$
The MiCA adapter is then initialized using the tail of $U$ and $V$.
