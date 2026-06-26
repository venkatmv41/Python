# Channel Estimation for OFDM MIMO Systems using LS, MMSE, and LMS

## Project Overview

This MATLAB project implements and compares three channel estimation techniques (Least Squares, Minimum Mean Square Error, and Least Mean Squares) for MIMO-OFDM systems with image transmission. The project evaluates the performance of these techniques across different modulation schemes including BPSK, QPSK, 16-QAM, and 32-QAM.

## Features

### Core Functionality
- **Image Transmission**: Load and process grayscale images for transmission over MIMO-OFDM systems
- **Multiple Modulation Schemes**: BPSK, QPSK, 16-QAM, and 32-QAM modulation
- **Channel Estimation Methods**: 
  - Least Squares (LS)
  - Minimum Mean Square Error (MMSE)
  - Least Mean Squares (LMS) adaptive filtering
- **OFDM Implementation**: Complete OFDM modulation with comb-type pilot structure
- **MIMO Channel Simulation**: Realistic MIMO channel using `comm.MIMOChannel`

### Performance Metrics
- **Bit Error Rate (BER)**: Error rate at bit level
- **Symbol Error Rate (SER)**: Error rate at symbol level  
- **Mean Squared Error (MSE)**: Channel estimation accuracy
- **Peak Signal-to-Noise Ratio (PSNR)**: Image reconstruction quality

### Visualization
- BER vs SNR plots (log scale)
- SER vs SNR plots
- MSE vs SNR plots
- PSNR vs SNR plots (image quality metric)
- Constellation diagrams (before and after channel estimation)
- Performance comparison charts

## System Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Subcarriers | 64 | Number of OFDM subcarriers |
| Cyclic Prefix | 16 | Cyclic prefix length |
| TX Antennas | 2 | Number of transmit antennas |
| RX Antennas | 2 | Number of receive antennas |
| Pilot Spacing | 4 | Spacing for comb-type pilots |
| SNR Range | 0:5:30 dB | Signal-to-noise ratio range |
| OFDM Symbols | 100 | Number of OFDM symbols |
| Channel Taps | 4 | Number of multipath channel taps |

## Prerequisites

### MATLAB Toolboxes Required
- **Communications Toolbox**: For modulation/demodulation functions
- **Signal Processing Toolbox**: For FFT/IFFT operations
- **Image Processing Toolbox**: For image handling functions

### MATLAB Version
- MATLAB R2019b or later recommended

## Installation and Usage

### Quick Start
1. Clone or download the project files
2. Ensure all required toolboxes are installed
3. Open MATLAB and navigate to the project directory
4. Run the main script:
```matlab
channel_estimation_ofdm_mimo
```

### File Structure
```
project/
├── channel_estimation_ofdm_mimo.m    # Main script with all functions
├── README.md                         # This documentation file
└── (generated plots and results)
```

## Technical Implementation

### Channel Estimation Algorithms

#### 1. Least Squares (LS)
- Direct estimation: `H = Y / X` at pilot positions
- Linear interpolation for data subcarriers
- Simple but noise-sensitive

#### 2. Minimum Mean Square Error (MMSE) 
- Starts with LS estimation
- Applies Wiener filtering for noise reduction
- Better performance than LS in noisy conditions

#### 3. Least Mean Squares (LMS)
- Adaptive algorithm with step size μ = 0.01
- Updates channel estimates symbol-by-symbol
- Good for time-varying channels

### Modulation Schemes

| Scheme | M | Bits/Symbol | Constellation |
|--------|---|-------------|---------------|
| BPSK   | 2 | 1           | PSK           |
| QPSK   | 4 | 2           | PSK           |
| 16-QAM | 16| 4           | QAM           |
| 32-QAM | 32| 5           | QAM           |

### OFDM Structure
- **Comb-type pilots**: Every 4th subcarrier carries pilot symbols
- **Cyclic prefix**: 25% of symbol duration (16 samples)
- **FFT size**: 64 points
- **Pilot symbols**: BPSK modulated for simplicity

## Results and Analysis

The simulation generates comprehensive performance analysis including:

### Performance Plots
1. **BER vs SNR**: Shows bit error rate performance across SNR range
2. **SER vs SNR**: Symbol error rate for different modulation schemes
3. **MSE vs SNR**: Channel estimation accuracy
4. **PSNR vs SNR**: Image reconstruction quality
5. **Constellation Diagrams**: Visual representation of received symbols
6. **Performance Comparison**: Bar charts comparing all methods

### Expected Results
- **MMSE** typically outperforms LS and LMS in high noise
- **Higher-order modulation** (32-QAM) more sensitive to channel estimation errors
- **LMS** shows good performance for time-varying channels
- **PSNR** improves with better channel estimation

## Code Organization

The project is implemented as a single MATLAB file with modular functions:

### Main Functions
- `load_and_process_image()`: Image loading and binary conversion
- `generate_modulated_data()`: Modulation for different schemes
- `create_ofdm_signal()`: OFDM signal generation with pilots
- `mimo_channel_simulation()`: Channel modeling and noise addition
- `channel_estimation()`: Dispatcher for estimation methods
- `calculate_performance_metrics()`: BER, SER, MSE, PSNR calculation
- `generate_all_plots()`: Comprehensive visualization

### Channel Estimation Functions
- `ls_channel_estimation()`: Least Squares implementation
- `mmse_channel_estimation()`: MMSE implementation  
- `lms_channel_estimation()`: LMS adaptive implementation

## Customization Options

### Modifying Parameters
Edit the parameters section in the main script:
```matlab
params.N_subcarriers = 64;     % Change number of subcarriers
params.SNR_range = 0:5:30;     % Modify SNR range
params.pilot_spacing = 4;      % Adjust pilot density
```

### Adding New Modulation Schemes
Add to the modulation_schemes cell array:
```matlab
modulation_schemes = {'BPSK', 'QPSK', '16QAM', '32QAM', '64QAM'};
```

### Custom Images
Replace the image loading section to use your own images:
```matlab
img = imread('your_image.png');
```

## Troubleshooting

### Common Issues
1. **Missing Toolbox Error**: Install required MATLAB toolboxes
2. **Memory Issues**: Reduce number of symbols or SNR points
3. **Slow Execution**: Decrease simulation parameters

### Performance Optimization
- Use parallel computing for multiple SNR points
- Vectorize operations where possible
- Reduce number of Monte Carlo runs

## Research Applications

This code is suitable for:
- **Academic Research**: Channel estimation algorithm comparison
- **System Design**: OFDM system performance evaluation
- **Algorithm Development**: Baseline for new estimation techniques
- **Education**: Understanding MIMO-OFDM principles

## References and Further Reading

1. "OFDM for Wireless Multimedia Communications" by Richard van Nee
2. "MIMO-OFDM Wireless Communications with MATLAB" by Yong Soo Cho
3. IEEE papers on channel estimation techniques
4. 3GPP standards for OFDM implementation

## License

This project is for educational and research purposes. Please cite appropriately if used in publications.

## Author

MATLAB Channel Estimation Project - 2024

## Support

For questions or issues:
1. Check MATLAB documentation for toolbox functions
2. Verify all prerequisites are met
3. Review parameter settings for your specific use case

---

**Note**: This implementation prioritizes clarity and educational value. For production systems, additional optimizations and error handling may be required.
