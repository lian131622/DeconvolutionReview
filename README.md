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
