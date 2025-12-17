# rq_vae

Experiments on applying RQ-VAE to fast text decoding in latent space.

## Main Idea

We use ideas from two papers:
- [Kaiser 2018: Fast Decoding in Sequence Models using Discrete Latent Variables](https://charleslow.github.io/notebook/book/papers/kaiser_2018.html)
    - Applies the idea of representing tokens in a shorter latent space, and then doing autoregressive text translation in the latent space, then upsample back to token space
    - Still uses old VQ-VAE discretization which has issues
- [Lee 2022: Autoregressive Image Generation using Residual Quantization](https://charleslow.github.io/notebook/book/papers/lee_2022.html)
    - Better way of doing discretization, using a codebook with multiple levels instead of a flat codebook
    - Some tricks of using a specialized transformer for decoding that is faster

## Experiment Goal

Two main goals:
- Verify if we can achieve significant speedup by decoding in latent space without losing accuracy.
- Verify if the latent representations are useful for e.g. search or recommendation

