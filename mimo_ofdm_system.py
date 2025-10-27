import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.linalg import inv, pinv
import cv2
from PIL import Image
import os
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class MIMOOFDMSystem:
    def __init__(self):
        # System parameters
        self.N_fft = 64          # FFT size
        self.N_cp = 16           # Cyclic prefix length
        self.N_tx = 2            # Number of transmitters
        self.N_rx = 2            # Number of receivers
        self.pilot_spacing = 4   # Pilot spacing for comb-type
        self.pilot_power = 1     # Pilot power
        
        # Modulation parameters
        self.modulation_schemes = ['BPSK', 'QPSK', '16-QAM', '32-QAM', '64-QAM']
        self.constellation_maps = self._initialize_constellations()
        
        # Channel parameters
        self.channel_taps = 4    # Number of multipath taps
        self.channel_delay_spread = 3
        
        # Simulation parameters
        self.snr_range = np.arange(10, 31, 2)  # SNR range in dB
        self.num_symbols = 100   # Number of OFDM symbols per frame
        
        # Results storage
        self.results = {}
        
    def _initialize_constellations(self):
        """Initialize constellation mappings for different modulation schemes"""
        constellations = {}
        
        # BPSK
        constellations['BPSK'] = np.array([-1, 1])
        
        # QPSK
        constellations['QPSK'] = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        
        # 16-QAM
        points = [-3, -1, 1, 3]
        constellations['16-QAM'] = np.array([i+1j*q for i in points for q in points]) / np.sqrt(10)
        
        # 32-QAM (Cross constellation)
        constellations['32-QAM'] = self._generate_32qam()
        
        # 64-QAM
        points = [-7, -5, -3, -1, 1, 3, 5, 7]
        constellations['64-QAM'] = np.array([i+1j*q for i in points for q in points]) / np.sqrt(42)
        
        return constellations
    
    def _generate_32qam(self):
        """Generate 32-QAM constellation (cross-shaped)"""
        # Standard 32-QAM cross constellation
        points = []
        for i in range(-3, 4):
            for q in range(-3, 4):
                if abs(i) + abs(q) <= 3:
                    points.append(i + 1j*q)
        
        # Select first 32 points and normalize
        points = np.array(points[:32])
        return points / np.sqrt(np.mean(np.abs(points)**2))
    
    def bits_to_symbols(self, bits, modulation):
        """Convert bits to constellation symbols"""
        constellation = self.constellation_maps[modulation]
        bits_per_symbol = int(np.log2(len(constellation)))
        
        # Reshape bits into groups
        num_symbols = len(bits) // bits_per_symbol
        bit_groups = bits[:num_symbols * bits_per_symbol].reshape(-1, bits_per_symbol)
        
        # Convert to decimal indices
        indices = np.sum(bit_groups * (2**np.arange(bits_per_symbol-1, -1, -1)), axis=1)
        
        return constellation[indices]
    
    def symbols_to_bits(self, symbols, modulation):
        """Convert symbols back to bits using minimum distance detection"""
        constellation = self.constellation_maps[modulation]
        bits_per_symbol = int(np.log2(len(constellation)))
        
        # Find closest constellation point for each symbol
        distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        
        # Convert indices to bits
        bits = []
        for idx in indices:
            bit_array = np.array([int(b) for b in format(idx, f'0{bits_per_symbol}b')])
            bits.extend(bit_array)
        
        return np.array(bits)
    
    def generate_pilots(self, pilot_type='block'):
        """Generate pilot symbols"""
        if pilot_type == 'block':
            # Block-type pilots (entire OFDM symbol)
            pilots = np.ones(self.N_fft, dtype=complex)
        else:
            # Comb-type pilots
            pilots = np.zeros(self.N_fft, dtype=complex)
            pilot_indices = np.arange(0, self.N_fft, self.pilot_spacing)
            pilots[pilot_indices] = 1
        
        return pilots
    
    def add_cyclic_prefix(self, ofdm_symbol):
        """Add cyclic prefix to OFDM symbol"""
        return np.concatenate([ofdm_symbol[-self.N_cp:], ofdm_symbol])
    
    def remove_cyclic_prefix(self, received_symbol):
        """Remove cyclic prefix from received symbol"""
        return received_symbol[self.N_cp:]
    
    def generate_channel(self):
        """Generate multipath fading channel"""
        # Generate complex channel coefficients for each tap
        h = np.random.randn(self.N_tx, self.N_rx, self.channel_taps) + \
            1j * np.random.randn(self.N_tx, self.N_rx, self.channel_taps)
        
        # Apply power delay profile (exponential decay)
        power_delay_profile = np.exp(-np.arange(self.channel_taps) / 2)
        h = h * np.sqrt(power_delay_profile / 2)
        
        return h
    
    def apply_channel(self, tx_signal, channel_coeffs, snr_db):
        """Apply multipath channel and AWGN"""
        # Convert SNR from dB to linear
        snr_linear = 10**(snr_db / 10)
        
        # Initialize received signal
        rx_signal = np.zeros((self.N_rx, tx_signal.shape[1]), dtype=complex)
        
        # Apply channel for each TX-RX pair
        for rx_idx in range(self.N_rx):
            for tx_idx in range(self.N_tx):
                # Convolve with channel impulse response
                channel_response = channel_coeffs[tx_idx, rx_idx, :]
                convolved = convolve(tx_signal[tx_idx], channel_response, mode='same')
                rx_signal[rx_idx] += convolved
        
        # Add AWGN
        signal_power = np.mean(np.abs(rx_signal)**2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*rx_signal.shape) + 
                                           1j * np.random.randn(*rx_signal.shape))
        
        return rx_signal + noise
    
    def channel_frequency_response(self, channel_coeffs):
        """Compute frequency response of channel"""
        H = np.zeros((self.N_tx, self.N_rx, self.N_fft), dtype=complex)
        
        for tx_idx in range(self.N_tx):
            for rx_idx in range(self.N_rx):
                # Zero-pad channel coefficients
                h_padded = np.zeros(self.N_fft)
                h_padded[:len(channel_coeffs[tx_idx, rx_idx])] = channel_coeffs[tx_idx, rx_idx]
                
                # Compute FFT
                H[tx_idx, rx_idx] = np.fft.fft(h_padded)
        
        return H
    
    def ls_channel_estimation(self, Y_pilot, X_pilot):
        """Least Squares channel estimation"""
        # Y_pilot: received pilot symbols (N_rx x N_pilot)
        # X_pilot: transmitted pilot symbols (N_tx x N_pilot)
        
        H_ls = np.zeros((self.N_tx, self.N_rx, self.N_fft), dtype=complex)
        
        for rx_idx in range(self.N_rx):
            for tx_idx in range(self.N_tx):
                # LS estimation: H = Y / X
                H_ls[tx_idx, rx_idx] = Y_pilot[rx_idx] / X_pilot[tx_idx]
        
        return H_ls
    
    def mmse_channel_estimation(self, H_ls, snr_db):
        """MMSE channel estimation"""
        snr_linear = 10**(snr_db / 10)
        
        # Simplified MMSE (assuming uncorrelated channels)
        # H_mmse = R_HH * (R_HH + sigma^2 * I)^(-1) * H_ls
        
        # Estimate channel autocorrelation (simplified)
        R_HH = np.eye(self.N_fft) * 0.5  # Assumed channel correlation
        noise_var = 1 / snr_linear
        
        H_mmse = np.zeros_like(H_ls)
        
        for tx_idx in range(self.N_tx):
            for rx_idx in range(self.N_rx):
                # MMSE filter
                W_mmse = R_HH @ inv(R_HH + noise_var * np.eye(self.N_fft))
                H_mmse[tx_idx, rx_idx] = W_mmse @ H_ls[tx_idx, rx_idx]
        
        return H_mmse
    
    def lmse_channel_estimation(self, H_ls, snr_db):
        """Linear MMSE channel estimation"""
        snr_linear = 10**(snr_db / 10)
        
        # LMSE with simplified assumptions
        # Similar to MMSE but with linear approximation
        
        alpha = snr_linear / (snr_linear + 1)  # Wiener filter coefficient
        H_lmse = alpha * H_ls
        
        return H_lmse
    
    def zero_forcing_equalization(self, Y, H_est):
        """Zero-forcing equalization"""
        X_est = np.zeros((self.N_tx, self.N_fft), dtype=complex)
        
        for k in range(self.N_fft):
            # Extract channel matrix for subcarrier k
            H_k = H_est[:, :, k]  # N_rx x N_tx
            
            # Pseudo-inverse for ZF equalization
            H_inv = pinv(H_k)
            
            # Equalize
            X_est[:, k] = H_inv @ Y[:, k]
        
        return X_est
    
    def calculate_ber(self, tx_bits, rx_bits):
        """Calculate Bit Error Rate"""
        errors = np.sum(tx_bits != rx_bits)
        return errors / len(tx_bits)
    
    def calculate_ser(self, tx_symbols, rx_symbols):
        """Calculate Symbol Error Rate"""
        errors = np.sum(tx_symbols != rx_symbols)
        return errors / len(tx_symbols)
    
    def calculate_mse(self, H_true, H_est):
        """Calculate Mean Square Error for channel estimation"""
        return np.mean(np.abs(H_true - H_est)**2)
    
    def image_to_bits(self, image_path, target_size=(64, 64)):
        """Convert image to binary data"""
        # Load and resize image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Create a test image if file doesn't exist
            img = np.random.randint(0, 256, target_size, dtype=np.uint8)
        else:
            img = cv2.resize(img, target_size)
        
        # Convert to binary
        bits = np.unpackbits(img.flatten())
        return bits, img
    
    def bits_to_image(self, bits, original_shape):
        """Convert binary data back to image"""
        # Ensure we have the right number of bits
        required_bits = np.prod(original_shape) * 8
        if len(bits) < required_bits:
            bits = np.pad(bits, (0, required_bits - len(bits)), 'constant')
        else:
            bits = bits[:required_bits]
        
        # Convert to bytes and reshape
        bytes_array = np.packbits(bits.astype(int))
        img = bytes_array.reshape(original_shape)
        return img
    
    def simulate_transmission(self, bits, modulation, snr_db, estimation_method='perfect'):
        """Simulate complete transmission chain"""
        # Generate channel
        channel_coeffs = self.generate_channel()
        H_true = self.channel_frequency_response(channel_coeffs)
        
        # Modulate bits
        symbols = self.bits_to_symbols(bits, modulation)
        
        # Pad symbols to fit OFDM frames
        symbols_per_frame = self.N_fft * self.N_tx
        num_frames = int(np.ceil(len(symbols) / symbols_per_frame))
        padded_symbols = np.pad(symbols, (0, num_frames * symbols_per_frame - len(symbols)), 'constant')
        
        # Reshape for MIMO-OFDM
        tx_symbols = padded_symbols.reshape(num_frames, self.N_tx, self.N_fft)
        
        # Generate pilots
        pilots = self.generate_pilots('block')
        
        # Initialize arrays
        rx_symbols_est = []
        channel_estimates = []
        
        for frame_idx in range(num_frames):
            # Add pilot symbol at beginning of each frame
            tx_frame = np.vstack([np.tile(pilots, (self.N_tx, 1)), tx_symbols[frame_idx]])
            
            # IFFT
            tx_time = np.fft.ifft(tx_frame, axis=1)
            
            # Add cyclic prefix
            tx_cp = np.array([self.add_cyclic_prefix(tx_time[tx_idx]) for tx_idx in range(self.N_tx)])
            
            # Transmit through channel
            rx_cp = self.apply_channel(tx_cp, channel_coeffs, snr_db)
            
            # Remove cyclic prefix
            rx_time = np.array([self.remove_cyclic_prefix(rx_cp[rx_idx]) for rx_idx in range(self.N_rx)])
            
            # FFT
            rx_freq = np.fft.fft(rx_time, axis=1)
            
            # Separate pilots and data
            rx_pilots = rx_freq[:, :self.N_fft]  # First symbol is pilot
            rx_data = rx_freq[:, self.N_fft:]    # Rest is data
            
            # Channel estimation
            if estimation_method == 'perfect':
                H_est = H_true
            elif estimation_method == 'ls':
                H_est = self.ls_channel_estimation(rx_pilots, np.tile(pilots, (self.N_tx, 1)))
            elif estimation_method == 'mmse':
                H_ls = self.ls_channel_estimation(rx_pilots, np.tile(pilots, (self.N_tx, 1)))
                H_est = self.mmse_channel_estimation(H_ls, snr_db)
            elif estimation_method == 'lmse':
                H_ls = self.ls_channel_estimation(rx_pilots, np.tile(pilots, (self.N_tx, 1)))
                H_est = self.lmse_channel_estimation(H_ls, snr_db)
            else:  # no estimation
                H_est = np.ones_like(H_true)
            
            # Equalization
            rx_symbols_eq = self.zero_forcing_equalization(rx_data, H_est)
            
            rx_symbols_est.append(rx_symbols_eq.flatten())
            channel_estimates.append(H_est)
        
        # Concatenate results
        rx_symbols_est = np.concatenate(rx_symbols_est)
        
        # Demodulate
        rx_bits = self.symbols_to_bits(rx_symbols_est[:len(symbols)], modulation)
        
        return rx_bits[:len(bits)], channel_estimates, H_true
    
    def run_simulation(self, image_path='test_image.png'):
        """Run complete simulation"""
        print("Starting MIMO-OFDM simulation...")
        
        # Load image
        bits, original_image = self.image_to_bits(image_path)
        print(f"Image loaded: {original_image.shape}, Total bits: {len(bits)}")
        
        # Initialize results
        results = {
            'ber': {},
            'ser': {},
            'mse': {},
            'images': {},
            'channel_estimates': {}
        }
        
        estimation_methods = ['perfect', 'ls', 'mmse', 'lmse', 'none']
        
        for modulation in self.modulation_schemes:
            print(f"\nProcessing {modulation}...")
            
            results['ber'][modulation] = {}
            results['ser'][modulation] = {}
            results['mse'][modulation] = {}
            results['images'][modulation] = {}
            results['channel_estimates'][modulation] = {}
            
            for method in estimation_methods:
                print(f"  Channel estimation: {method}")
                
                ber_values = []
                ser_values = []
                mse_values = []
                
                for snr_db in self.snr_range:
                    # Simulate transmission
                    rx_bits, channel_est, H_true = self.simulate_transmission(
                        bits, modulation, snr_db, method
                    )
                    
                    # Calculate metrics
                    ber = self.calculate_ber(bits, rx_bits)
                    
                    # Convert to symbols for SER calculation
                    tx_symbols = self.bits_to_symbols(bits, modulation)
                    rx_symbols = self.bits_to_symbols(rx_bits, modulation)
                    ser = self.calculate_ser(tx_symbols, rx_symbols)
                    
                    # Calculate MSE for channel estimation
                    if method != 'perfect':
                        mse = self.calculate_mse(H_true, channel_est[0])
                    else:
                        mse = 0
                    
                    ber_values.append(ber)
                    ser_values.append(ser)
                    mse_values.append(mse)
                
                results['ber'][modulation][method] = ber_values
                results['ser'][modulation][method] = ser_values
                results['mse'][modulation][method] = mse_values
                
                # Store reconstructed image at highest SNR
                rx_bits_best, _, _ = self.simulate_transmission(
                    bits, modulation, self.snr_range[-1], method
                )
                reconstructed_image = self.bits_to_image(rx_bits_best, original_image.shape)
                results['images'][modulation][method] = reconstructed_image
                
                # Store channel estimates
                _, channel_est, H_true = self.simulate_transmission(
                    bits, modulation, 20, method  # Use 20 dB for channel estimation storage
                )
                results['channel_estimates'][modulation][method] = {
                    'estimated': channel_est[0],
                    'true': H_true
                }
        
        # Store original image
        results['original_image'] = original_image
        
        # Save results
        self.results = results
        with open('simulation_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print("\nSimulation completed!")
        return results
    
    def plot_results(self):
        """Generate all plots"""
        if not self.results:
            print("No results to plot. Run simulation first.")
            return
        
        # Create output directory
        os.makedirs('plots', exist_ok=True)
        os.makedirs('images', exist_ok=True)
        
        # Plot BER vs SNR
        self._plot_ber_vs_snr()
        
        # Plot SER vs SNR
        self._plot_ser_vs_snr()
        
        # Plot MSE vs SNR
        self._plot_mse_vs_snr()
        
        # Plot constellation diagrams
        self._plot_constellations()
        
        # Save transmitted/received images
        self._save_images()
        
        # Plot channel estimation comparison
        self._plot_channel_estimation()
    
    def _plot_ber_vs_snr(self):
        """Plot BER vs SNR for all modulation schemes"""
        plt.figure(figsize=(15, 10))
        
        estimation_methods = ['perfect', 'ls', 'mmse', 'lmse', 'none']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, modulation in enumerate(self.modulation_schemes):
            plt.subplot(2, 3, i+1)
            
            for method, color in zip(estimation_methods, colors):
                ber_values = self.results['ber'][modulation][method]
                plt.semilogy(self.snr_range, ber_values, 
                           marker='o', label=f'{method.upper()}', color=color)
            
            plt.xlabel('SNR (dB)')
            plt.ylabel('BER')
            plt.title(f'BER vs SNR - {modulation}')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/ber_vs_snr.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ser_vs_snr(self):
        """Plot SER vs SNR for all modulation schemes"""
        plt.figure(figsize=(15, 10))
        
        estimation_methods = ['perfect', 'ls', 'mmse', 'lmse', 'none']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, modulation in enumerate(self.modulation_schemes):
            plt.subplot(2, 3, i+1)
            
            for method, color in zip(estimation_methods, colors):
                ser_values = self.results['ser'][modulation][method]
                plt.semilogy(self.snr_range, ser_values, 
                           marker='s', label=f'{method.upper()}', color=color)
            
            plt.xlabel('SNR (dB)')
            plt.ylabel('SER')
            plt.title(f'SER vs SNR - {modulation}')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/ser_vs_snr.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_mse_vs_snr(self):
        """Plot MSE vs SNR for channel estimation methods"""
        plt.figure(figsize=(15, 10))
        
        estimation_methods = ['ls', 'mmse', 'lmse']
        colors = ['red', 'green', 'orange']
        
        for i, modulation in enumerate(self.modulation_schemes):
            plt.subplot(2, 3, i+1)
            
            for method, color in zip(estimation_methods, colors):
                mse_values = self.results['mse'][modulation][method]
                plt.semilogy(self.snr_range, mse_values, 
                           marker='^', label=f'{method.upper()}', color=color)
            
            plt.xlabel('SNR (dB)')
            plt.ylabel('MSE')
            plt.title(f'Channel Estimation MSE - {modulation}')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/mse_vs_snr.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_constellations(self):
        """Plot constellation diagrams"""
        plt.figure(figsize=(15, 10))
        
        for i, modulation in enumerate(self.modulation_schemes):
            plt.subplot(2, 3, i+1)
            
            constellation = self.constellation_maps[modulation]
            plt.scatter(constellation.real, constellation.imag, 
                       s=100, alpha=0.7, edgecolors='black')
            
            plt.xlabel('In-phase')
            plt.ylabel('Quadrature')
            plt.title(f'{modulation} Constellation')
            plt.grid(True)
            plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('plots/constellations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_images(self):
        """Save transmitted and received images"""
        original_image = self.results['original_image']
        
        # Save original image
        cv2.imwrite('images/original_image.png', original_image)
        
        estimation_methods = ['perfect', 'ls', 'mmse', 'lmse', 'none']
        
        for modulation in self.modulation_schemes:
            for method in estimation_methods:
                reconstructed = self.results['images'][modulation][method]
                filename = f'images/{modulation}_{method}_reconstructed.png'
                cv2.imwrite(filename, reconstructed)
        
        # Create comparison plots
        self._create_image_comparison_plots()
    
    def _create_image_comparison_plots(self):
        """Create comparison plots showing original vs reconstructed images"""
        original_image = self.results['original_image']
        estimation_methods = ['perfect', 'ls', 'mmse', 'lmse', 'none']
        
        for modulation in self.modulation_schemes:
            plt.figure(figsize=(18, 6))
            
            # Original image
            plt.subplot(2, 3, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            # Reconstructed images
            for i, method in enumerate(estimation_methods):
                plt.subplot(2, 3, i+2)
                reconstructed = self.results['images'][modulation][method]
                plt.imshow(reconstructed, cmap='gray')
                plt.title(f'{method.upper()} Estimation')
                plt.axis('off')
            
            plt.suptitle(f'Image Transmission - {modulation}')
            plt.tight_layout()
            plt.savefig(f'plots/image_comparison_{modulation}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_channel_estimation(self):
        """Plot channel estimation comparison"""
        plt.figure(figsize=(15, 10))
        
        modulation = 'QPSK'  # Use QPSK as example
        estimation_methods = ['ls', 'mmse', 'lmse']
        
        for i, method in enumerate(estimation_methods):
            plt.subplot(2, 2, i+1)
            
            H_est = self.results['channel_estimates'][modulation][method]['estimated']
            H_true = self.results['channel_estimates'][modulation][method]['true']
            
            # Plot magnitude response for first TX-RX pair
            plt.plot(np.abs(H_true[0, 0, :]), 'b-', label='True Channel', linewidth=2)
            plt.plot(np.abs(H_est[0, 0, :]), 'r--', label='Estimated Channel', linewidth=2)
            
            plt.xlabel('Subcarrier Index')
            plt.ylabel('Channel Magnitude')
            plt.title(f'{method.upper()} Channel Estimation')
            plt.legend()
            plt.grid(True)
        
        # Overall comparison
        plt.subplot(2, 2, 4)
        for method in estimation_methods:
            H_est = self.results['channel_estimates'][modulation][method]['estimated']
            H_true = self.results['channel_estimates'][modulation][method]['true']
            
            error = np.abs(H_true[0, 0, :] - H_est[0, 0, :])
            plt.plot(error, label=f'{method.upper()} Error', linewidth=2)
        
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Estimation Error')
        plt.title('Channel Estimation Error Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/channel_estimation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the simulation"""
    # Create test image if it doesn't exist
    if not os.path.exists('test_image.png'):
        # Create a simple test pattern
        test_img = np.zeros((64, 64), dtype=np.uint8)
        test_img[16:48, 16:48] = 255  # White square
        test_img[24:40, 24:40] = 128  # Gray square inside
        cv2.imwrite('test_image.png', test_img)
        print("Created test image: test_image.png")
    
    # Initialize system
    system = MIMOOFDMSystem()
    
    # Run simulation
    results = system.run_simulation('test_image.png')
    
    # Generate plots
    system.plot_results()
    
    # Print summary
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"System Parameters:")
    print(f"  FFT Size: {system.N_fft}")
    print(f"  Cyclic Prefix: {system.N_cp}")
    print(f"  MIMO Configuration: {system.N_tx}x{system.N_rx}")
    print(f"  SNR Range: {system.snr_range[0]} to {system.snr_range[-1]} dB")
    print(f"  Modulation Schemes: {', '.join(system.modulation_schemes)}")
    print(f"  Channel Estimation Methods: LS, MMSE, LMSE")
    
    print(f"\nOutput Files Generated:")
    print(f"  - simulation_results.pkl (raw data)")
    print(f"  - plots/ber_vs_snr.png")
    print(f"  - plots/ser_vs_snr.png") 
    print(f"  - plots/mse_vs_snr.png")
    print(f"  - plots/constellations.png")
    print(f"  - plots/channel_estimation_comparison.png")
    print(f"  - plots/image_comparison_*.png (for each modulation)")
    print(f"  - images/ folder with all reconstructed images")
    
    print(f"\nSimulation completed successfully!")

if __name__ == "__main__":
    main()