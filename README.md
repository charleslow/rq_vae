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

Verify if we can achieve significant speedup by decoding in latent space without losing accuracy
- Check perplexity on held out data
- Check codebook usage
- Compare against various base models (including qwen0.6B) on standard LLM tasks

Other goals:
- Test if RQ-transformer matches standard transformer decoding for the same compute
- Test if using a pretrained decoder backbone is necessary
- Test performance - inference speed trade-off as we increase compression factor
- Test codebook vocabulary size to codebook depth tradeoff
- Test if unsloth speeds up inference significantly


