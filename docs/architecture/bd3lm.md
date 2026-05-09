# Block Diffusion (BD3LM)

**BD3LM** (Block Discrete Diffusion Language Model) is the core training objective of this project. It interpolates between standard Autoregressive modeling and full sequence Diffusion.

## The Objective

Instead of predicting one token at a time or the entire sequence at once, BD3LM operates on **blocks** of tokens. 

### Masking Strategy
1.  A block size $K$ is selected.
2.  A diffusion timestep $t \in (0, 1]$ is sampled.
3.  Within each block of size $K$, tokens are stochastically masked based on $t$.
4.  The model is trained to recover the original tokens given the partially masked block and the full causal context of previous blocks.

## Advantages

*   **Efficiency**: Much faster than token-by-token diffusion.
*   **Quality**: Allows for non-autoregressive refinement within a block, improving local coherence.
*   **Flexibility**: By setting $K=1$, the model behaves like an AR model. By setting $K=L$ (sequence length), it behaves like a standard MDLM.
