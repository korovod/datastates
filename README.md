# DataStates

Efficient asynchronous checkpointing engine.

For a detailed description about design principles, implementation, and performance evaluation against state-of-the-art checkpointing engines, please refer to the [HPDC'24 paper](https://hal.science/hal-04614247).

## Usage

### Requirements

- Python
- pybind11
- PyTorch

### Installation

Using Spack:

```
git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git
git clone https://github.com/korovod/korovod-spack-packages.git
cd spack/bin
./spack repo add korovod-spack-packages
./spack install py-datastates
```

Using pip:

```
git clone https://github.com/korovod/datastates.git
cd datastates

# Install the CPP/Python binding
pip install . -v
```

### Using DataStates in your Python project

```python
from datastates import CkptEngine
```

## Tests

```
# Test with a simple PyTorch code, DeepSpeed not required.
python tests/test_ckpt_engine.py   

# Test with a simple DeepSpeed code.
python tests/test_datastates_llm.py   
```

## Citation

> Avinash Maurya, Robert Underwood, M. Mustafa Rafique, Franck Cappello, and Bogdan Nicolae. "DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models". HPDC'24: The 33rd International Symposium on High-Performance Parallel and Distributed Computing (Pisa, Italy, 2024).
