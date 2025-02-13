### Layout-Anchored Prioritizing Continual Learning for Continuous Building Footprint Extraction From High-Resolution Remote Sensing Imagery

The repo is based on [avalanche](https://github.com/ContinualAI/avalanche).

### Introduction
Continuous building footprint extraction requires learning new building patterns from remote sensing imagery without forgetting old knowledge. It is inherently challenging due to the spatial layout heterogeneity, which leads to the problem of knowledge forgetting in two aspects: complex background could have distinct patterns (background diversity), and buildings could have similar patterns to the background (foreground-background similarity). To solve the issues, we propose a domain-incremental continual learning algorithm named layout-anchored prioritizing learning network (LAPNet), including a latent layout anchoring module and layout-aware prioritizing learning module. The latent layout anchoring aggregates background information into latent layout features and employs a herding strategy to select representative layout anchors iteratively. This module maintains a memory buffer to narrow the background differences by dynamically discarding unrepresentative experiences and storing layout-anchored experiences. Furthermore, layout-aware prioritizing learning uses these experiences to identify and emphasize the most valuable knowledge for maximizing interclass distance. This module leverages layout variance metric to measure interclass discrepancies and employs prioritizing learning to reweight the optimization function based on this layout prior. We established a Global-CL dataset to validate the proposed LAPNet framework, containing six study areas across four continents with different remote sensing sensors. Experiments showed that LAPNet achieves state-of-the-art performance in continuous building footprint extraction by effectively correlating knowledge across various domains.

## Installation
* Python (3.9.20)
* Pytorch (1.10.1+cu11.3)

Please refer to INSTALL.md in [avalanche](https://github.com/ContinualAI/avalanche).

## Getting Started
1. Train a model
```
bash scripts/train_segm.sh
```

2. Test a dataset
```
bash scripts/test_segm.sh
```

## Citation

```BibTeX
@article{chen2023semi,
  title={Semi-supervised knowledge distillation framework for global-scale urban man-made object remote sensing mapping},
  author={Chen, Dingyuan and Ma, Ailong and Zhong, Yanfei},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={122},
  pages={103439},
  year={2023},
  publisher={Elsevier}
}
