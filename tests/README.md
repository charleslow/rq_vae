# RQ-VAE Test Suite

Comprehensive test suite for the RQ-VAE project. All tests are designed to run on both CPU and CUDA devices.

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Or use pytest directly
pytest tests/
```

### Test Options

```bash
# Run on CPU only (even if CUDA is available)
python run_tests.py --cpu

# Run specific test file
python run_tests.py tests/test_quantizer.py

# Run specific test
python run_tests.py tests/test_quantizer.py::TestResidualQuantizer::test_forward_shape

# Run with coverage report
python run_tests.py --coverage

# Run in verbose mode
python run_tests.py -v

# Run with output capture disabled (useful for debugging)
python run_tests.py -s

# Run only fast tests (skip slow ones)
python run_tests.py -m "not slow"

# Stop on first failure
python run_tests.py --failfast

# Run tests matching keyword
python run_tests.py -k "gradient"
```

## Test Structure

```
tests/
├── __init__.py              # Test package init
├── conftest.py              # Shared fixtures and configuration
├── README.md                # This file
├── test_quantizer.py        # Tests for ResidualQuantizer
├── test_layers.py           # Tests for SwiGLU and transformer layers
├── test_encoder.py          # Tests for TextEncoder
├── test_decoder.py          # Tests for TextDecoder
├── test_rq_vae.py          # Tests for RQVAE model
├── test_rq_transformer.py   # Tests for RQTransformer
└── test_dataset.py          # Tests for dataset utilities
```

## Test Coverage

### Components Tested

- ✅ **ResidualQuantizer** ([test_quantizer.py](test_quantizer.py))
  - Forward pass and output shapes
  - Codebook initialization and EMA updates
  - Decode indices functionality
  - Straight-through estimator
  - Commitment loss
  - Codebook usage statistics
  - CPU/CUDA consistency

- ✅ **Neural Network Layers** ([test_layers.py](test_layers.py))
  - SwiGLU activation
  - SwiGLUTransformerLayer
  - Gradient flow
  - Different model dimensions

- ✅ **TextEncoder** ([test_encoder.py](test_encoder.py))
  - Forward pass with compression
  - Different compression factors
  - Latent refinement layers
  - Freeze/unfreeze backbone
  - Gradient flow

- ✅ **TextDecoder** ([test_decoder.py](test_decoder.py))
  - Forward pass with upsampling
  - Target length specification
  - Different compression factors
  - Latent refinement layers
  - Output logits validation

- ✅ **RQVAE** ([test_rq_vae.py](test_rq_vae.py))
  - Full encode-decode cycle
  - Reconstruction loss
  - Save and load checkpoints
  - Different model configurations
  - Gradient flow

- ✅ **RQTransformer** ([test_rq_transformer.py](test_rq_transformer.py))
  - Spatial and depth transformers
  - Forward pass with teacher forcing
  - Autoregressive generation
  - Top-k and nucleus sampling
  - Different temperatures

- ✅ **Dataset Utilities** ([test_dataset.py](test_dataset.py))
  - TextDataset creation
  - StreamingTextDataset
  - DataLoader creation
  - Mock testing for HuggingFace datasets

## CPU/CUDA Compatibility

All tests are designed to work on both CPU and CUDA:

- **Automatic Device Selection**: Tests automatically detect and use CUDA if available
- **Forced CPU Mode**: Use `--cpu` flag to run tests on CPU even when CUDA is available
- **CUDA-specific Tests**: Some tests verify CPU/CUDA consistency (skipped if CUDA unavailable)

### Testing Device Handling

```python
def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

All test modules include this helper to ensure tests run on the appropriate device.

## Test Markers

Tests can be marked with custom markers:

- `@pytest.mark.slow` - Slow tests (can skip with `-m "not slow"`)
- `@pytest.mark.cuda` - Tests requiring CUDA

## Writing New Tests

When adding new tests:

1. Use the `get_device()` helper for device selection
2. Ensure tests work on both CPU and CUDA
3. Add appropriate markers if test is slow or requires CUDA
4. Include docstrings explaining what is being tested
5. Test edge cases and different configurations

Example:

```python
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_my_feature():
    """Test my new feature."""
    device = get_device()
    model = MyModel().to(device)

    x = torch.randn(2, 10, 128, device=device)
    output = model(x)

    assert output.shape == (2, 10, 128)
```

## Dependencies

Required packages for running tests:

```
pytest>=7.0
torch
transformers
datasets
einops
unsloth
```

Optional (for coverage):

```
pytest-cov
```

## Continuous Integration

These tests are designed to run in CI/CD environments:

- Tests automatically adapt to CPU-only environments
- Mock testing for datasets to avoid downloading large files
- Reasonable timeouts for model tests
- Clear pass/fail criteria

## Troubleshooting

### Tests fail with CUDA OOM

```bash
# Run on CPU instead
python run_tests.py --cpu
```

### Tests are too slow

```bash
# Skip slow tests
python run_tests.py -m "not slow"
```

### Need to debug a specific test

```bash
# Run with output capture disabled and verbose mode
python run_tests.py -sv tests/test_quantizer.py::TestResidualQuantizer::test_forward_shape
```

### Want to see coverage

```bash
python run_tests.py --coverage
# Open htmlcov/index.html in browser
```
