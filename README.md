# Numerical Computing
**Aalborg University - Semester 8**

## Overview

This repository contains coursework and projects for the Numerical Computing course at Aalborg University.

## Environment Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate numenv
```

## Dependencies

The `numenv` environment includes:
- NumPy - Numerical computing
- SciPy - Scientific computing
- Matplotlib - Plotting and visualization
- [Additional packages as needed]

## Usage

Always activate the environment before working:

```bash
conda activate numenv
```

To deactivate when done:

```bash
conda deactivate
```

## Updating the Environment

If dependencies change, update your environment:

```bash
conda env update -f environment.yml
```

## Troubleshooting

### Environment creation fails
```bash
# Remove old environment and try again
conda env remove -n numenv
conda env create -f environment.yml
```

## Course Information

**Institution:** Aalborg University  
**Semester:** 8  
**Subject:** Numerical Computing

