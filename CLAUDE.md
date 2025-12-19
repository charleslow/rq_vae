# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RQ-VAE (Residual Quantized Variational Autoencoder) experiments for fast text decoding. This project explores applying RQ-VAE techniques to accelerate text generation/decoding in language models.

## Test configuration

Maintain a minimal set of tests that tests only the critical logic of the modules. Aim for readable and essential tests that are easier to maintain rather than exhausting all possibilities.

Before running tests, make sure to `source .venv/bin/activate` before running python scripts.


## External Libraries

Some code from external libraries is copied and pasted in `src/external/`. Do NOT modify these files, but can import from them.

## Library Usage

Use established implementations when they are available:
- For residual quantization:
    - Use `ResidualVQ` from `src/external/residual_vq.py`
- For optimized pytorch transformer modules:
    - Use `x_transformers` package
    - e.g. Use `x_transformers.Decoder` for a standard implementation of the transformer decoder
- For the RQ-transformer:
    - Use `src/external/rq_transformer.py`. This is copied from an established source by `lucidrains`, do not modify this file

