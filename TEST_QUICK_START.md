# RQ-VAE Tests - Quick Start

## Run All Tests

```bash
python run_tests.py
```

## Common Commands

```bash
# Run on CPU only (faster for development)
python run_tests.py --cpu

# Run specific component
python run_tests.py tests/test_quantizer.py
python run_tests.py tests/test_encoder.py
python run_tests.py tests/test_decoder.py
python run_tests.py tests/test_rq_vae.py

# Run with coverage report
python run_tests.py --coverage

# Verbose mode
python run_tests.py -v

# Stop on first failure
python run_tests.py --failfast
```

## Test Structure

```
tests/
├── test_quantizer.py        # ResidualQuantizer (15 tests)
├── test_layers.py           # SwiGLU & layers (20+ tests)
├── test_encoder.py          # TextEncoder (14 tests)
├── test_decoder.py          # TextDecoder (14 tests)
├── test_rq_vae.py          # RQVAE model (18 tests)
├── test_rq_transformer.py   # RQTransformer (25+ tests)
└── test_dataset.py          # Dataset utilities (mock tests)
```

## Key Features

✅ **CPU/CUDA Compatible** - All tests work on both CPU and CUDA
✅ **Automatic Device Detection** - No manual configuration needed
✅ **Comprehensive Coverage** - 100+ tests covering all components
✅ **Modular Design** - Each component tested independently
✅ **Easy to Run** - Single command to run all tests

## Verify Installation

```bash
# Check if pytest is installed
pytest --version

# Run a quick test
python run_tests.py tests/test_quantizer.py::TestResidualQuantizer::test_initialization -v
```

## Device Selection

Tests automatically use CUDA if available, or CPU otherwise.

Force CPU mode:
```bash
python run_tests.py --cpu
```

Check what device will be used:
```python
import torch
print(torch.cuda.is_available())  # True if CUDA available
```

## Need Help?

- Full documentation: [tests/README.md](tests/README.md)
- Testing guide: [TESTING.md](TESTING.md)
- Run `python run_tests.py --help` for all options
