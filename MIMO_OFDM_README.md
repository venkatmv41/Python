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
=======
## Overview

This MATLAB implementation provides a complete simulation of a MIMO-OFDM (Multiple-Input Multiple-Output Orthogonal Frequency Division Multiplexing) system with advanced channel estimation techniques. The implementation is based on the IEEE paper "Channel Estimation using LS and MMSE Channel Estimation Techniques for MIMO-OFDM Systems" (2022).

## System Features

### 🎯 Core Functionality
- **MIMO Configuration**: 2x2 MIMO system (2 transmit, 2 receive antennas)
- **OFDM Parameters**: 64-point FFT with 16-sample cyclic prefix
- **Modulation Schemes**: BPSK, QPSK, 16-QAM, 32-QAM, 64-QAM
- **Channel Estimation**: LS, MMSE, LMSE, and Perfect CSI
- **Image Transmission**: Complete image transmission and reconstruction

### 📊 Performance Metrics
- Bit Error Rate (BER) vs SNR
- Symbol Error Rate (SER) vs SNR
- Mean Square Error (MSE) for channel estimation
- Peak Signal-to-Noise Ratio (PSNR) for image quality
- Spectral Efficiency analysis

## System Parameters

```matlab
FFT Size (N_fft):           64
Cyclic Prefix (N_cp):       16
Number of Tx Antennas:      2
Number of Rx Antennas:      2
Pilot Subcarriers:          16
Data Subcarriers:           48
SNR Range:                  10-30 dB (5 dB steps)
Pilot Spacing:              4 (comb-type pilots)
```

## File Structure

```
├── main_mimo_ofdm_system.m     # Main simulation script
├── transmitter_processing.m     # Transmitter chain implementation
├── channel_and_noise.m         # Channel modeling and noise addition
├── receiver_processing.m       # Receiver chain with channel estimation
├── reconstruct_image.m         # Image reconstruction from bits
├── generate_all_plots.m        # Comprehensive plotting functions
├── create_test_image.m         # Test image generation
└── MIMO_OFDM_README.md         # This documentation file
```

## Usage Instructions

### 1. Basic Simulation
```matlab
% Run the complete simulation
main_mimo_ofdm_system
```

### 2. Create Test Image (Optional)
```matlab
% Create a test image if none is available
create_test_image
```

### 3. Load and Analyze Results
```matlab
% Load saved results
load('mimo_ofdm_results.mat');

% Access specific results
ber_bpsk_mmse = results.BPSK.BER_MMSE;
mse_qpsk_ls = results.QPSK.MSE_LS;
```

## Channel Estimation Techniques

### 1. Least Squares (LS) Estimation
```
H_LS = Y / X
```
- Simple and computationally efficient
- No prior channel knowledge required
- Sensitive to noise

### 2. Minimum Mean Square Error (MMSE) Estimation
```
H_MMSE = R_HH * (R_HH + σ²(X^H*X)^(-1))^(-1) * H_LS
```
- Optimal in MMSE sense
- Requires noise variance knowledge
- Better performance than LS

### 3. Linear MMSE (LMSE) Estimation
```
H_LMSE = W * H_LS
```
- Simplified version of MMSE
- Linear filtering approach
- Good compromise between complexity and performance

## Modulation Schemes

| Modulation | Bits/Symbol | Constellation Points | Normalization |
|------------|-------------|---------------------|---------------|
| BPSK       | 1           | 2                   | ±1            |
| QPSK       | 2           | 4                   | /√2           |
| 16-QAM     | 4           | 16                  | /√10          |
| 32-QAM     | 5           | 32                  | /√20          |
| 64-QAM     | 6           | 64                  | /√42          |

## Channel Model

### Multipath Fading Channel
- **Number of Paths**: 4
- **Path Delays**: [0, 1, 2, 3] samples
- **Path Gains**: [1, 0.5, 0.3, 0.1]
- **Fading Type**: Rayleigh fading
- **Noise Type**: Additive White Gaussian Noise (AWGN)

## Generated Outputs

### 1. Performance Plots
- `BER_vs_SNR_Analysis.png`: BER performance comparison
- `SER_vs_SNR_Analysis.png`: SER performance comparison
- `MSE_vs_SNR_Analysis.png`: Channel estimation MSE
- `Constellation_Diagrams.png`: Constellation plots
- `Performance_Summary_Table.png`: Performance summary

### 2. Image Transmission Results
- `BPSK_Image_Transmission.png`: BPSK image results
- `QPSK_Image_Transmission.png`: QPSK image results
- `16QAM_Image_Transmission.png`: 16-QAM image results
- `32QAM_Image_Transmission.png`: 32-QAM image results
- `64QAM_Image_Transmission.png`: 64-QAM image results

### 3. Data Files
- `mimo_ofdm_results.mat`: Complete simulation results
- `test_image.png`: Test image used for transmission

## Key Results and Observations

### 1. BER Performance
- **Perfect CSI**: Provides lower bound performance
- **MMSE**: Best among practical estimators
- **LMSE**: Good compromise between complexity and performance
- **LS**: Simplest but worst performance
- **No Estimation**: Shows importance of channel estimation

### 2. Channel Estimation MSE
- MSE decreases with increasing SNR
- MMSE provides lowest MSE
- LS estimation MSE is highest

### 3. Modulation Comparison
- Higher-order modulations (64-QAM) achieve higher data rates
- Lower-order modulations (BPSK, QPSK) are more robust
- Trade-off between spectral efficiency and error performance

### 4. Image Quality
- PSNR values indicate image reconstruction quality
- Higher SNR and better channel estimation improve image quality
- Visual degradation patterns show effect of channel estimation errors

## Technical Implementation Details

### Transmitter Chain
1. **Bit Generation**: Convert image to binary data
2. **Serial-to-Parallel**: Group bits for parallel transmission
3. **Modulation**: Map bits to constellation symbols
4. **Pilot Insertion**: Add known pilots for channel estimation
5. **IFFT**: Convert to time domain
6. **Cyclic Prefix**: Add guard interval

### Receiver Chain
1. **Remove Cyclic Prefix**: Extract OFDM symbol
2. **FFT**: Convert to frequency domain
3. **Channel Estimation**: Estimate channel using pilots
4. **Equalization**: Compensate for channel effects
5. **Demodulation**: Convert symbols back to bits
6. **Parallel-to-Serial**: Reconstruct bit stream

### Channel Estimation Process
1. **Pilot Extraction**: Extract known pilot symbols
2. **LS Estimation**: Initial channel estimate
3. **MMSE/LMSE Filtering**: Improve estimate using statistics
4. **Interpolation**: Estimate channel for data subcarriers

## Performance Optimization

### Computational Complexity
- **LS**: O(N) - Linear complexity
- **MMSE**: O(N³) - Matrix inversion required
- **LMSE**: O(N²) - Reduced complexity

### Memory Requirements
- Channel matrices: N_rx × N_tx × N_fft
- Signal buffers: N_symbols × (N_fft + N_cp)
- Results storage: Multiple SNR points and modulations

## Extensions and Future Work

### Possible Enhancements
1. **Advanced Channel Models**: Vehicular channels, frequency-selective fading
2. **Adaptive Algorithms**: Adaptive modulation and coding
3. **MIMO Techniques**: Spatial multiplexing, diversity combining
4. **Channel Coding**: Error correction codes
5. **Synchronization**: Timing and frequency offset compensation

### Research Directions
1. **Machine Learning**: Deep learning for channel estimation
2. **Massive MIMO**: Large antenna arrays
3. **mmWave Communications**: High-frequency applications
4. **5G/6G Integration**: Next-generation wireless systems

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce image size or SNR points
2. **Convergence Problems**: Check channel conditioning
3. **Plot Errors**: Ensure all results are properly stored
4. **Image Quality**: Verify bit-to-pixel conversion

### Performance Tips
1. Use parallel processing for SNR loops
2. Optimize matrix operations
3. Reduce simulation parameters for faster execution
4. Use appropriate data types (single vs double precision)

## References

1. IEEE Paper: "Channel Estimation using LS and MMSE Channel Estimation Techniques for MIMO-OFDM Systems" (2022)
2. OFDM Theory: "OFDM for Wireless Communications Systems" by Ramjee Prasad
3. MIMO Systems: "Introduction to Space-Time Wireless Communications" by Paulraj et al.
4. Channel Estimation: "Wireless Communications" by Andrea Goldsmith

## License and Citation

This implementation is provided for educational and research purposes. Please cite the original IEEE paper when using this code for academic work.

---

**Author**: Advanced AI Implementation  
**Date**: 2024  
**Version**: 1.0  
**MATLAB Version**: R2020b or later recommended

