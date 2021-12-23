# Quantified Truth Inference

This repository is the official implementation of Quantified Truth Inference for Dynamic-label Malware Annotation (DMA).

![architecture](../architecture.png)

*DMA Architecture*

## Installation
- Getting the code
```shell
git clone https://github.com/jyker/dma.git
```
- Install [poetry](https://python-poetry.org/docs/#installation) (Poetry is a tool for dependency management and packaging in Python.)

- Install requirements
```shell
cd dma/qvote
poetry install
```
## Evaluation
```
poetry shell
cd example
python qvote_malgenome.py
```