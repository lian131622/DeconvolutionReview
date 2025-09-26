# Deconvolution Algorithms review for UT and THz NDE applications

This repository provides a comprehensive review and implementation of deconvolution algorithms for Ultrasonic Testing (UT) and Terahertz (THz) Non-Destructive Evaluation (NDE) applications. The codebase includes scripts for generating synthetic signals, adding noise, and benchmarking various deconvolution methods such as OMP, MUSIC, L1, and AR algorithms.

### Features

- **Signal Generation**: Create synthetic reference and test signals with configurable parameters.
- **Noise Addition**: Simulate different Signal-to-Noise Ratios (SNR) for robust benchmarking.
- **Deconvolution Algorithms**: Implement and compare multiple state-of-the-art algorithms.
- **Performance Metrics**: Evaluate algorithms using metrics like Time of Flight (ToF) error, amplitude error, and delay estimation.
- **Visualization**: Plot and analyze results for deeper insights.

### Demo

The `.ipynb` file demonstrates and compares deconvolution algorithms on UT and THz NDE signals with visualization.

### Getting Started

1. **Requirements**:

   - Python 3.x
   - `numpy`, `scipy`, `matplotlib`, `tqdm`
2. **Data**:
   Place your `.mat` data files in the `Data/` directory as referenced in the scripts.
3. **Running Examples**:

   - To run a benchmark or demo, execute the relevant script in the `Scripts/` folder, e.g.:
     ```
     python Scripts/OpenMaps.py
     ```
   - Modify parameters in the scripts to suit your experimental setup.

### Folder/Files Structure

- `Data/` - Example data files (not included).
- `Deconvolution.py` - Algorithm implementations.
- `README.md` - Project overview and instructions.

### Cite the review paper

This git is the code for the paper: **Stratigraphic reconstruction from terahertz and ultrasonic signals by deconvolution: A review**

Cite as:
@article{ZITO2026103524,
title = {Stratigraphic reconstruction from terahertz and ultrasonic signals by deconvolution: A review},
journal = {NDT & E International},
volume = {158},
pages = {103524},
year = {2026},
issn = {0963-8695},
doi = {https://doi.org/10.1016/j.ndteint.2025.103524},
url = {https://www.sciencedirect.com/science/article/pii/S0963869525002051},
author = {Rocco Zito and Haolian Shi and Marco Ricci and Stefano Laureti and D.S. Citrin and Alexandre Locquet},
keywords = {Deconvolution, Stratigraphy, Terahertz, Ultrasound, Matching pursuit, Sparse deconvolution, Autoregressive spectral extrapolation, MUltiple SIgnal Classification},
abstract = {Nondestructive evaluation techniques, such as terahertz and ultrasonic testing, use short pulses to probe layered materials and reconstruct their stratigraphy by analyzing time delays between echoes at internal interfaces. However, when layers are sufficiently thin, successive echoes temporally overlap, making direct identification of their number and timing challenging. In such cases, deconvolution techniques are employed to extract the impulse response or key features such as echo locations and amplitudes, improving resolution of the local stratigraphy. This review examines four widely used deconvolution algorithms for stratigraphic reconstruction under the assumption of a sparse impulse response, where layer boundaries are modeled as discrete, sharp echoes. Two time-domain methods—orthogonal matching pursuit and ℓ1-norm-based sparse deconvolution—and two frequency-domain approaches—multiple signal classification and autoregressive spectral extrapolation—are discussed. Their theoretical foundations, practical implementation, and comparative performance are evaluated using synthetic signals and experimental echograms from terahertz pulsed imaging and ultrasound sonography. These techniques enhance the ability to distinguish closely spaced interfaces and are applicable to defect detection in materials, tissue-layer analysis in medical diagnostics, and preprocessing for 3D imaging.}
}

The data used can be found in the mendely dataset at: [Deconvolution Review - Mendeley Data](https://data.mendeley.com/datasets/6t2yz6kcf9/2)

And can be cited as:

"Zito, Rocco; Shi, Haolian; Ricci, Marco; Locquet, Alexandre; Citrin, David; Laureti, Stefano (2025), “Deconvolution Review”, Mendeley Data, V2, doi: 10.17632/6t2yz6kcf9.2"
