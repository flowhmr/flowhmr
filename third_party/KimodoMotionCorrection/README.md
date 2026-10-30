# motion_correction

Standalone `correct_motion` implementation packaged as a small C++ motion trajectory correction library with Python bindings.

## Installation Guide

### Prerequisites

Ensure you have a C++17 compatible compiler (GCC 7.0+, Clang 5.0+, or MSVC 2017+) and CMake 3.15+. On Windows, install MinGW-w64 or Visual Studio with C++ tools. On Linux, install `build-essential` and `cmake`.

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

### Build & Install

#### Standard Installation
```bash
pip install .
```

#### Development Installation
```bash
pip install -e .
```

### Verify Installation

```python
import motion_correction
print("Installation successful!")
```
You can also run `python run_test.py` for a simple test.
