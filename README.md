# Fifth Force Measurements

This repository contains the code used for the fifth force measurements presented in [arXiv:2502.12843](https://arxiv.org/abs/2502.12843).  

The test is based on combining measurements of  

- the growth rate of structure, $\hat{f} = f\sigma_8$  
- the evolution of the Weyl potential, $\hat{J}(z)$ (see [arXiv:2209.08987](https://arxiv.org/abs/2209.08987) and [arXiv:2312.06434](https://arxiv.org/abs/2312.06434)).  

---

## Repository contents

- **`interpolate_fhat_Jhat.py`**  
  Functions for reconstructing $\hat{f}(z)$ or $\hat{J}(z)$ over some redshift range based on spline interpolation, given measurements at individual redshifts.  

- **`calculate_Gamma.py`**  
  Functions to compute $\Gamma(z)$, the strength of the fifth force, from $\hat{f}(z)$ and $\hat{J}(z)$ (when both are available at matching redshifts).  

- **`Jhat_fhat_current.ipynb`**  
  Notebook with calculations and plots based on current data:  
  - $\hat{f}(z)$ from various galaxy surveys  
  - $\hat{J}(z)$ from DES Y3 data (see [arXiv:2312.06434](https://arxiv.org/abs/2312.06434))  

- **`Jhat_fhat_DESI_LSST.ipynb`**  
  Notebook with forecasts using:  
  - $\hat{f}(z)$ from the full DESI survey  
  - $\hat{J}(z)$ from LSST  


