# LabelPropagation_MetaPath

# Dynamic Heterogeneous Graph Representation for Social Networks

## Overview
We present a **dynamic heterogeneous graph** representation for social networks that captures:
- **Multiplexity** (multiple types of nodes and edges)
- **Time-awareness** (timestamps on every node and edge)

Our approach enables:
- Complex **time-dependent queries**
- Improved **classification accuracy** in dynamic networks

We introduce **Meta-paths + LPA**, a generalization of the Label Propagation Algorithm for temporal heterogeneous graphs, and show that it outperforms four state-of-the-art algorithms by **at least 13.79% accuracy**.

---

## Key Features
- 📅 **Temporal Graph Structure** — Time is integrated into all components.  
- 🔗 **Heterogeneous Support** — Multiple node and edge types.  
- 🧭 **Temporal Meta-paths** — Extends meta-path concepts to dynamic settings.  
- 🤖 **Meta-paths + LPA Algorithm** — Enhanced label propagation for classification.  
- 📊 **Benchmark Proven** — Validated on *Steemit* + three benchmark datasets.

---

## Repository Structure
```
├── data/                  # Datasets (Steemit data)
├── code/                 # Graph construction + algorithms
└── README.md
```

---

## Results
**Meta-paths + LPA** consistently outperforms competitors:  
- Accuracy improvement: **+13.79%** minimum across benchmarks.

---

## Citation
```bibtex
@article{YourCitationKey,
  title={Social Network Prediction Problems: Using Meta-Paths and Dynamic Heterogeneous Graph Representation for Label Propagation},
  author={Maleki, N., Padmanabhan, B., & Dutta, K.},
  journal={INFORMS Journal on Computing},
  year={2025}
}
```

