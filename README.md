# TUAP: Temporally Unified Adversarial Perturbations for Time Series Forecasting

Official code repository for the paper **"Temporally Unified Adversarial Perturbations for Time Series Forecasting"** (arXiv 2026).

## Overview

This work addresses the **temporally inconsistent perturbations** problem in adversarial attacks on time series forecasting. Existing methods generate divergent perturbation values for the same timestamp across overlapping samples, making such attacks impractical for real-world data manipulation.

We propose **Temporally Unified Adversarial Perturbations (TUAPs)**, which enforce identical perturbations for each timestamp across all overlapping samples. We also introduce **Timestamp-wise Gradient Accumulation Method (TGAM)** to effectively generate TUAPs by aggregating gradient information from overlapping samples. **MI-TGAM** integrates TGAM with momentum-based attack algorithms for enhanced effectiveness and transferability.

## Key Contributions

- **TUAP**: A novel adversarial perturbation concept that ensures temporal consistency across overlapping samples
- **TGAM**: A modular and efficient method for generating TUAPs via timestamp-wise gradient accumulation
- **MI-TGAM**: Momentum-integrated TGAM for stronger white-box and black-box transfer attacks

## Citation

```bibtex
@article{su2026tuap,
  title={Temporally Unified Adversarial Perturbations for Time Series Forecasting}, 
  author={Ruixian Su and Yukun Bao and Xinze Zhang},
  year={2026},
  eprint={2602.11940},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.11940}, 
}
```
