# MIMO-OFDM System with Channel Estimation

## Overview

This MATLAB implementation provides a complete simulation of a MIMO-OFDM (Multiple-Input Multiple-Output Orthogonal Frequency Division Multiplexing) system with advanced channel estimation techniques. The implementation is based on the IEEE paper "Channel Estimation using LS and MMSE Channel Estimation Techniques for MIMO-OFDM Systems" (2022).

## System Features

### 🎯 Core Functionality
- **MIMO Configuration**: 2x2 MIMO system (2 transmit, 2 receive antennas)
- **OFDM Parameters**: 64-point FFT with 16-sample cyclic prefix
- **Modulation Schemes**: BPSK, QPSK, QAM16, QAM32, QAM64
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
| QAM16      | 4           | 16                  | /√10          |
| QAM32      | 5           | 32                  | /√20          |
| QAM64      | 6           | 64                  | /√42          |

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
- `QAM16_Image_Transmission.png`: QAM16 image results
- `QAM32_Image_Transmission.png`: QAM32 image results
- `QAM64_Image_Transmission.png`: QAM64 image results

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
- Higher-order modulations (QAM64) achieve higher data rates
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