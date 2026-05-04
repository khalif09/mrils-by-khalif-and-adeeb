# MRILS & Enhanced MRILS

**Multi‑Restart Iterated Local Search for Influence Maximization**

_©️ 2026 Sheikh Khalif and Mohammed Adeeb. All rights reserved._

## Authors
- **Sheikh Khalif** – National Institute of Technology Srinagar, India  
  Email: khalif.123.sk@gmail.com
- **Mohammed Adeeb** – National Institute of Technology Srinagar, India

## Description
This repository contains the Python implementation of the MRILS and MRILS‑Enhanced algorithms for the influence maximization problem under the Independent Cascade model. The methods combine Reverse Influence Sampling (RIS) with multi‑restart greedy selection, RR‑set‑guided local search, and Monte Carlo evaluation.

The algorithms are described in detail in the accompanying paper:

> Sheikh Khalif and Mohammed Adeeb, **“MRILS and Enhanced MRILS: Multi‑Restart Iterated Local Search for Influence Maximization”**, 2026.

## Files
- `mrils.py` – Main algorithm implementations and helpers.
- `test_mrils.py` – Full test suite (pytest) for correctness and comparison.
- `LICENSE` – MIT License.

## Requirements
- Python 3.7+
- [NetworkX](https://networkx.org/)
- [pytest](https://pytest.org/)

Install dependencies:
```bash
pip install networkx pytest
