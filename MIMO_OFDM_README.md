# MIMO-OFDM System with Channel Estimation

This project implements a complete MIMO-OFDM (Multiple-Input Multiple-Output Orthogonal Frequency Division Multiplexing) system simulation with advanced channel estimation techniques, based on the IEEE paper "Channel Estimation using LS and MMSE Channel Estimation Techniques for MIMO-OFDM Systems" (2022).

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Implementation Features](#implementation-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Analysis](#results-and-analysis)
- [File Structure](#file-structure)
- [Technical Details](#technical-details)
- [Performance Metrics](#performance-metrics)

## 🎯 Overview

The system simulates a complete MIMO-OFDM transmission chain including:
- **Transmitter**: Bit generation, modulation, IFFT, cyclic prefix insertion
- **Channel**: Multipath fading with AWGN
- **Receiver**: Channel estimation, equalization, demodulation
- **Analysis**: Comprehensive performance evaluation

### Key Features
- **Multiple Modulation Schemes**: BPSK, QPSK, 16-QAM, 32-QAM, 64-QAM
- **Channel Estimation Methods**: LS, MMSE, LMSE, Perfect CSI, No Estimation
- **MIMO Configuration**: 2×2 (2 transmitters, 2 receivers)
- **Image Transmission**: Visual demonstration of system performance
- **Comprehensive Analysis**: BER, SER, MSE, PSNR, SSIM metrics

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TRANSMITTER   │    │     CHANNEL     │    │    RECEIVER     │
│                 │    │                 │    │                 │
│ • Bit Gen       │    │ • Multipath     │    │ • Channel Est   │
│ • Modulation    │───▶│ • Rayleigh      │───▶│ • Equalization  │
│ • IFFT          │    │ • AWGN          │    │ • Demodulation  │
│ • Cyclic Prefix │    │                 │    │ • Bit Recovery  │
│ • Pilot Insert  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Implementation Features

### System Parameters
- **FFT Size**: 64 subcarriers
- **Cyclic Prefix**: 16 samples
- **MIMO Configuration**: 2×2
- **SNR Range**: 10-30 dB
- **Channel Taps**: 4 (multipath)
- **Pilot Type**: Block-type (full OFDM symbol)

### Modulation Schemes
1. **BPSK**: 1 bit/symbol
2. **QPSK**: 2 bits/symbol
3. **16-QAM**: 4 bits/symbol
4. **32-QAM**: 5 bits/symbol (cross constellation)
5. **64-QAM**: 6 bits/symbol

### Channel Estimation Methods
1. **Perfect CSI**: Ideal channel knowledge
2. **LS (Least Squares)**: H_LS = Y/X
3. **MMSE (Minimum Mean Square Error)**: H_MMSE = R_HH(R_HH + σ²I)⁻¹H_LS
4. **LMSE (Linear MMSE)**: Simplified linear approximation
5. **No Estimation**: No channel compensation

## 📦 Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Packages
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- opencv-python >= 4.5.0
- Pillow >= 8.3.0
- scikit-learn >= 1.0.0
- pandas

## 🎮 Usage

### Basic Simulation
```bash
python mimo_ofdm_system.py
```

### Advanced Analysis
```bash
python analysis_script.py
```

### Custom Image
```python
from mimo_ofdm_system import MIMOOFDMSystem

# Initialize system
system = MIMOOFDMSystem()

# Run simulation with custom image
results = system.run_simulation('your_image.png')

# Generate plots
system.plot_results()
```

## 📊 Results and Analysis

### Generated Outputs

#### 1. Performance Plots
- **BER vs SNR**: Bit Error Rate performance
- **SER vs SNR**: Symbol Error Rate performance
- **MSE vs SNR**: Channel estimation error
- **Constellation Diagrams**: Signal space representation

#### 2. Image Analysis
- **Original vs Reconstructed**: Visual comparison
- **PSNR/SSIM Metrics**: Image quality assessment
- **Individual Images**: Saved for each modulation/estimation method

#### 3. Advanced Analysis
- **Performance Heatmaps**: Comparative analysis
- **Channel Error Analysis**: Estimation accuracy
- **SNR Threshold Analysis**: Required SNR for target BER
- **Modulation Comparison**: Spectral efficiency vs performance

### Key Findings
1. **MMSE** generally outperforms LS and LMSE
2. **Higher-order modulations** require higher SNR
3. **Channel estimation** significantly improves performance
4. **Perfect CSI** provides upper bound performance
5. **Image quality** correlates with BER performance

## 📁 File Structure

```
├── mimo_ofdm_system.py      # Main simulation system
├── analysis_script.py       # Advanced analysis tools
├── requirements.txt         # Python dependencies
├── MIMO_OFDM_README.md     # This documentation
├── test_image.png          # Generated test image
├── simulation_results.pkl   # Saved simulation data
├── plots/                  # Basic performance plots
│   ├── ber_vs_snr.png
│   ├── ser_vs_snr.png
│   ├── mse_vs_snr.png
│   ├── constellations.png
│   ├── channel_estimation_comparison.png
│   └── image_comparison_*.png
├── images/                 # Reconstructed images
│   ├── original_image.png
│   └── *_reconstructed.png
└── advanced_plots/         # Advanced analysis plots
    ├── performance_heatmap.png
    ├── channel_error_analysis.png
    ├── image_quality_metrics.png
    ├── snr_threshold_analysis.png
    └── modulation_comparison.png
```

## 🔧 Technical Details

### Channel Model
The system uses a realistic multipath fading channel:
```python
h(t) = Σ αᵢ δ(t - τᵢ)
```
where αᵢ are complex channel coefficients and τᵢ are delays.

### Channel Estimation Algorithms

#### 1. Least Squares (LS)
```python
H_LS = Y / X
```
Simple but noise-sensitive.

#### 2. Minimum Mean Square Error (MMSE)
```python
H_MMSE = R_HH(R_HH + σ²(XᴴX)⁻¹)⁻¹H_LS
```
Optimal but requires channel statistics.

#### 3. Linear MMSE (LMSE)
```python
H_LMSE = α × H_LS
```
where α = SNR/(SNR + 1) is the Wiener filter coefficient.

### Equalization
Zero-forcing equalization is used:
```python
X_est = H⁻¹ × Y
```

## 📈 Performance Metrics

### Communication Metrics
- **BER (Bit Error Rate)**: Bit-level error probability
- **SER (Symbol Error Rate)**: Symbol-level error probability
- **MSE (Mean Square Error)**: Channel estimation accuracy

### Image Quality Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Image reconstruction quality
- **SSIM (Structural Similarity Index)**: Perceptual image quality
- **NCC (Normalized Cross-Correlation)**: Image similarity

### Analysis Metrics
- **SNR Thresholds**: Required SNR for target performance
- **Spectral Efficiency**: Bits per symbol vs performance trade-off
- **Estimation Accuracy**: Channel estimation error analysis

## 🎯 Use Cases

1. **Wireless Communication Research**: MIMO-OFDM system design
2. **Channel Estimation Studies**: Comparative analysis of estimation methods
3. **Image Transmission**: Visual demonstration of system performance
4. **Educational Purposes**: Understanding OFDM and MIMO concepts
5. **Performance Benchmarking**: Baseline for advanced techniques

## 🔮 Future Enhancements

- **Deep Learning Channel Estimation**: Neural network-based methods
- **Adaptive Modulation**: Dynamic modulation selection
- **MIMO Detection**: Advanced detection algorithms
- **Realistic Channel Models**: 3GPP channel models
- **Coding**: Error correction coding integration

## 📚 References

1. IEEE Paper: "Channel Estimation using LS and MMSE Channel Estimation Techniques for MIMO-OFDM Systems" (2022)
2. Goldsmith, A. "Wireless Communications" Cambridge University Press
3. Tse, D. and Viswanath, P. "Fundamentals of Wireless Communication"
4. 3GPP Technical Specifications for MIMO-OFDM

## 👥 Contributing

Feel free to contribute by:
- Adding new channel estimation methods
- Implementing advanced MIMO detection
- Enhancing visualization tools
- Adding more realistic channel models

## 📝 License

This project is for educational and research purposes. Please cite the original IEEE paper when using this code.

---

**Note**: This implementation is designed for educational purposes and may require modifications for production use. The system provides a solid foundation for understanding MIMO-OFDM systems and channel estimation techniques.