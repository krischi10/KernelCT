# KernelCT

This repository contains the Python implementation of kernel-based reconstruction methods for Computerized Tomography (CT). This code was developed as part of the PhD thesis

> *Kernel-Based Generalized Interpolation and its Application to Computerized Tomography*.
> Kristof Albrecht.
> Dissertation, University of Hamburg, 2024.

Theoretical details and references for the implementation can be found in the thesis, which can
be downloaded on [ediss publication server of UHH](https://ediss.sub.uni-hamburg.de/handle/ediss/11362).

## Project Structure

The repository is organized as follows:

- `KernelCT/`: This directory contains the core Python package for the kernel-based reconstruction methods.
- `demo/`: This directory contains Jupyter notebooks demonstrating the usage of the `KernelCT` package.
- `requirements.txt`: A list of Python packages required to run the code.

## Usage

To get started with this project, it is recommended to explore the Jupyter notebooks in the `demo/` directory. They provide small examples of how to use the different reconstruction methods implemented in the `KernelCT` package.

Before running the notebooks, ensure you have installed the required dependencies, e.g. via

```bash
pip install -r requirements.txt
```

## References

The implementation of the greedy algorithms was inspired by the [VKOGA (Vectorial Kernel Orthogonal Greedy Algorithm) repository](https://github.com/GabrieleSantin/VKOGA).

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
