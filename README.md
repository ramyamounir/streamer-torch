# STREAMER
  
[![PyPI](https://img.shields.io/pypi/v/streamer-torch)](https://pypi.org/project/streamer-torch/)
[![Publish to PyPI](https://github.com/ramyamounir/streamer-torch/actions/workflows/pypi_publish.yaml/badge.svg)](https://github.com/ramyamounir/streamer-torch/actions/workflows/pypi_publish.yaml)

The official PyTorch implementation of our NeurIPS'23 paper [STREAMER: Streaming Representation Learning and Event Segmentation in a Hierarchical Manner](https://ramymounir.com/publications/streamer/)

![Overview of STREAMER](https://github.com/ramyamounir/streamer-torch/blob/main/assets/overview.jpg)

---

## Overview

### Documentation

Checkout the [documentation](https://ramymounir.com/docs/streamer/) of STREAMER modules to learn more details about how to use our codebase.

### Installation

```bash
pip install streamer-torch # with pip from PyPI
pip install git+'https://github.com/ramyamounir/streamer-torch' # with GitHub
```

### Inference

> Note: Pretrained weights are coming soon..


```python
from streamer.models.inference_model import InferenceModel

model = InferenceModel(checkpoint='to/checkpoint/path/')
result = model(filename='to/video/file/path')
```


### Training

In order to perform training with streamer:

1. Use the [Dataset README.md](https://github.com/ramyamounir/streamer-torch/blob/main/streamer/preprocessing/README.md) to preprocess datasets for streaming loading and evaluation.
2. Use the provided [training script](https://github.com/ramyamounir/streamer-torch/blob/main/train.py) to train on multiple gpus (i.e., or multi-node).
3. The script [compare.py](https://github.com/ramyamounir/streamer-torch/blob/main/streamer/experiments/) can be used to evaluate the model's prediction using Hierarchical Level Reduction.

> Bash scripts with CLI arguments are provided in [helper_scripts](https://github.com/ramyamounir/streamer-torch/tree/main/helper_scripts)

---

Citing STREAMER
-------
If you find our approaches useful in your research, please consider citing:
```
@inproceedings{mounir2023streamer,
  title={STREAMER: Streaming Representation Learning and Event Segmentation in a Hierarchical Manner},
  author={Mounir, Ramy and Vijayaraghavan, Sujal and Sarkar, Sudeep},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


