import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import stats
import pandas as pd

class MIMOOFDMAnalyzer:
    def __init__(self, results_file='simulation_results.pkl'):
        """Initialize analyzer with simulation results"""
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                self.results = pickle.load(f)
        else:
            print(f"Results file {results_file} not found. Run simulation first.")
            self.results = None
        
        self.modulation_schemes = ['BPSK', 'QPSK', '16-QAM', '32-QAM', '64-QAM']
        self.estimation_methods = ['perfect', 'ls', 'mmse', 'lmse', 'none']
        self.snr_range = np.arange(10, 31, 2)
        
    def calculate_additional_metrics(self):
        """Calculate additional performance metrics"""
        if not self.results:
            return None
        
        additional_metrics = {}
        
        for modulation in self.modulation_schemes:
            additional_metrics[modulation] = {}
            
            for method in self.estimation_methods:
                ber_values = np.array(self.results['ber'][modulation][method])
                ser_values = np.array(self.results['ser'][modulation][method])
                mse_values = np.array(self.results['mse'][modulation][method])
                
                # Calculate metrics
                metrics = {
                    'ber_mean': np.mean(ber_values),
                    'ber_std': np.std(ber_values),
                    'ser_mean': np.mean(ser_values),
                    'ser_std': np.std(ser_values),
                    'mse_mean': np.mean(mse_values),
                    'mse_std': np.std(mse_values),
                    'ber_min': np.min(ber_values),
                    'ber_max': np.max(ber_values),
                    'ser_min': np.min(ser_values),
                    'ser_max': np.max(ser_values),
                    'snr_threshold_ber_01': self._find_snr_threshold(ber_values, 0.1),
                    'snr_threshold_ber_001': self._find_snr_threshold(ber_values, 0.01),
                    'snr_threshold_ser_01': self._find_snr_threshold(ser_values, 0.1),
                    'snr_threshold_ser_001': self._find_snr_threshold(ser_values, 0.01)
                }
                
                additional_metrics[modulation][method] = metrics
        
        return additional_metrics
    
    def _find_snr_threshold(self, error_values, threshold):
        """Find SNR threshold for given error rate"""
        for i, error in enumerate(error_values):
            if error <= threshold:
                return self.snr_range[i]
        return None  # Threshold not reached
    
    def calculate_image_quality_metrics(self):
        """Calculate image quality metrics (PSNR, SSIM-like)"""
        if not self.results:
            return None
        
        original_image = self.results['original_image']
        quality_metrics = {}
        
        for modulation in self.modulation_schemes:
            quality_metrics[modulation] = {}
            
            for method in self.estimation_methods:
                reconstructed = self.results['images'][modulation][method]
                
                # Calculate PSNR
                mse_img = np.mean((original_image.astype(float) - reconstructed.astype(float))**2)
                if mse_img == 0:
                    psnr = float('inf')
                else:
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse_img))
                
                # Calculate normalized cross-correlation
                ncc = np.corrcoef(original_image.flatten(), reconstructed.flatten())[0, 1]
                
                # Calculate structural similarity (simplified)
                ssim = self._calculate_ssim(original_image, reconstructed)
                
                quality_metrics[modulation][method] = {
                    'psnr': psnr,
                    'ncc': ncc,
                    'ssim': ssim,
                    'mse_image': mse_img
                }
        
        return quality_metrics
    
    def _calculate_ssim(self, img1, img2):
        """Calculate simplified SSIM"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def compare_estimation_methods(self):
        """Compare different channel estimation methods"""
        if not self.results:
            return None
        
        comparison = {}
        
        for modulation in self.modulation_schemes:
            comparison[modulation] = {}
            
            # Compare at different SNR levels
            for snr_idx, snr in enumerate(self.snr_range):
                comparison[modulation][f'snr_{snr}'] = {}
                
                # Get values for each method
                for method in ['ls', 'mmse', 'lmse']:
                    ber = self.results['ber'][modulation][method][snr_idx]
                    ser = self.results['ser'][modulation][method][snr_idx]
                    mse = self.results['mse'][modulation][method][snr_idx]
                    
                    comparison[modulation][f'snr_{snr}'][method] = {
                        'ber': ber,
                        'ser': ser,
                        'mse': mse
                    }
                
                # Find best method for each metric
                methods = ['ls', 'mmse', 'lmse']
                ber_values = [comparison[modulation][f'snr_{snr}'][m]['ber'] for m in methods]
                ser_values = [comparison[modulation][f'snr_{snr}'][m]['ser'] for m in methods]
                mse_values = [comparison[modulation][f'snr_{snr}'][m]['mse'] for m in methods]
                
                comparison[modulation][f'snr_{snr}']['best_ber'] = methods[np.argmin(ber_values)]
                comparison[modulation][f'snr_{snr}']['best_ser'] = methods[np.argmin(ser_values)]
                comparison[modulation][f'snr_{snr}']['best_mse'] = methods[np.argmin(mse_values)]
        
        return comparison
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        if not self.results:
            print("No results available for analysis.")
            return
        
        print("="*80)
        print("COMPREHENSIVE MIMO-OFDM SYSTEM ANALYSIS REPORT")
        print("="*80)
        
        # Basic system information
        print("\n1. SYSTEM PARAMETERS")
        print("-" * 40)
        print(f"FFT Size: 64")
        print(f"Cyclic Prefix: 16")
        print(f"MIMO Configuration: 2x2")
        print(f"SNR Range: 10-30 dB")
        print(f"Modulation Schemes: {', '.join(self.modulation_schemes)}")
        print(f"Channel Estimation Methods: LS, MMSE, LMSE")
        
        # Calculate additional metrics
        additional_metrics = self.calculate_additional_metrics()
        
        print("\n2. PERFORMANCE SUMMARY")
        print("-" * 40)
        
        # Create summary table
        summary_data = []
        for modulation in self.modulation_schemes:
            for method in self.estimation_methods:
                metrics = additional_metrics[modulation][method]
                summary_data.append({
                    'Modulation': modulation,
                    'Method': method.upper(),
                    'Avg BER': f"{metrics['ber_mean']:.2e}",
                    'Avg SER': f"{metrics['ser_mean']:.2e}",
                    'Avg MSE': f"{metrics['mse_mean']:.2e}",
                    'Min BER': f"{metrics['ber_min']:.2e}",
                    'Min SER': f"{metrics['ser_min']:.2e}"
                })
        
        # Print summary table
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Image quality analysis
        print("\n3. IMAGE QUALITY ANALYSIS")
        print("-" * 40)
        
        quality_metrics = self.calculate_image_quality_metrics()
        
        quality_data = []
        for modulation in self.modulation_schemes:
            for method in self.estimation_methods:
                metrics = quality_metrics[modulation][method]
                quality_data.append({
                    'Modulation': modulation,
                    'Method': method.upper(),
                    'PSNR (dB)': f"{metrics['psnr']:.2f}",
                    'NCC': f"{metrics['ncc']:.3f}",
                    'SSIM': f"{metrics['ssim']:.3f}",
                    'MSE': f"{metrics['mse_image']:.2f}"
                })
        
        df_quality = pd.DataFrame(quality_data)
        print(df_quality.to_string(index=False))
        
        # Channel estimation comparison
        print("\n4. CHANNEL ESTIMATION COMPARISON")
        print("-" * 40)
        
        comparison = self.compare_estimation_methods()
        
        # Count best method occurrences
        best_counts = {'ls': 0, 'mmse': 0, 'lmse': 0}
        
        for modulation in self.modulation_schemes:
            for snr in self.snr_range:
                best_method = comparison[modulation][f'snr_{snr}']['best_ber']
                best_counts[best_method] += 1
        
        print("Best BER Performance Count (across all modulations and SNR levels):")
        for method, count in best_counts.items():
            percentage = (count / (len(self.modulation_schemes) * len(self.snr_range))) * 100
            print(f"  {method.upper()}: {count} times ({percentage:.1f}%)")
        
        # SNR threshold analysis
        print("\n5. SNR THRESHOLD ANALYSIS")
        print("-" * 40)
        
        print("SNR required for BER ≤ 0.01:")
        for modulation in self.modulation_schemes:
            print(f"\n{modulation}:")
            for method in ['ls', 'mmse', 'lmse']:
                threshold = additional_metrics[modulation][method]['snr_threshold_ber_001']
                if threshold is not None:
                    print(f"  {method.upper()}: {threshold} dB")
                else:
                    print(f"  {method.upper()}: >30 dB")
        
        print("\n6. MODULATION SCHEME COMPARISON")
        print("-" * 40)
        
        # Average BER across all SNR levels for each modulation
        mod_comparison = {}
        for modulation in self.modulation_schemes:
            avg_ber = np.mean([additional_metrics[modulation][method]['ber_mean'] 
                              for method in ['ls', 'mmse', 'lmse']])
            mod_comparison[modulation] = avg_ber
        
        sorted_mods = sorted(mod_comparison.items(), key=lambda x: x[1])
        print("Modulation schemes ranked by average BER performance:")
        for i, (mod, ber) in enumerate(sorted_mods, 1):
            print(f"  {i}. {mod}: {ber:.2e}")
        
        print("\n7. CHANNEL ESTIMATION METHOD RECOMMENDATIONS")
        print("-" * 40)
        
        # Analyze which method performs best overall
        method_scores = {'ls': 0, 'mmse': 0, 'lmse': 0}
        
        for modulation in self.modulation_schemes:
            for method in ['ls', 'mmse', 'lmse']:
                ber_rank = sorted(['ls', 'mmse', 'lmse'], 
                                key=lambda x: additional_metrics[modulation][x]['ber_mean']).index(method)
                method_scores[method] += (3 - ber_rank)  # Higher score for better rank
        
        best_method = max(method_scores, key=method_scores.get)
        print(f"Overall best performing method: {best_method.upper()}")
        print("Method scores (higher is better):")
        for method, score in sorted(method_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method.upper()}: {score}")
        
        print("\n" + "="*80)
        print("REPORT COMPLETED")
        print("="*80)
    
    def create_advanced_plots(self):
        """Create additional advanced analysis plots"""
        if not self.results:
            return
        
        os.makedirs('advanced_plots', exist_ok=True)
        
        # 1. Performance comparison heatmap
        self._plot_performance_heatmap()
        
        # 2. Channel estimation error analysis
        self._plot_channel_error_analysis()
        
        # 3. Image quality metrics
        self._plot_image_quality_metrics()
        
        # 4. SNR threshold analysis
        self._plot_snr_threshold_analysis()
        
        # 5. Modulation scheme comparison
        self._plot_modulation_comparison()
        
        print("Advanced plots saved in 'advanced_plots' directory")
    
    def _plot_performance_heatmap(self):
        """Create performance heatmap"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        estimation_methods = ['ls', 'mmse', 'lmse']
        
        for idx, metric in enumerate(['ber', 'ser', 'mse']):
            data = np.zeros((len(self.modulation_schemes), len(estimation_methods)))
            
            for i, modulation in enumerate(self.modulation_schemes):
                for j, method in enumerate(estimation_methods):
                    # Use average across all SNR values
                    values = self.results[metric][modulation][method]
                    data[i, j] = np.mean(values)
            
            im = axes[idx].imshow(data, cmap='viridis', aspect='auto')
            axes[idx].set_xticks(range(len(estimation_methods)))
            axes[idx].set_xticklabels([m.upper() for m in estimation_methods])
            axes[idx].set_yticks(range(len(self.modulation_schemes)))
            axes[idx].set_yticklabels(self.modulation_schemes)
            axes[idx].set_title(f'Average {metric.upper()}')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx])
            
            # Add text annotations
            for i in range(len(self.modulation_schemes)):
                for j in range(len(estimation_methods)):
                    text = axes[idx].text(j, i, f'{data[i, j]:.2e}',
                                        ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        plt.savefig('advanced_plots/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_channel_error_analysis(self):
        """Plot detailed channel estimation error analysis"""
        plt.figure(figsize=(15, 10))
        
        modulation = 'QPSK'  # Use QPSK as example
        estimation_methods = ['ls', 'mmse', 'lmse']
        
        for i, method in enumerate(estimation_methods):
            plt.subplot(2, 2, i+1)
            
            H_est = self.results['channel_estimates'][modulation][method]['estimated']
            H_true = self.results['channel_estimates'][modulation][method]['true']
            
            # Calculate error statistics
            error = np.abs(H_true - H_est)
            
            # Plot error distribution
            plt.hist(error.flatten(), bins=50, alpha=0.7, density=True)
            plt.xlabel('Estimation Error Magnitude')
            plt.ylabel('Probability Density')
            plt.title(f'{method.upper()} Error Distribution')
            plt.grid(True)
        
        # Overall error comparison
        plt.subplot(2, 2, 4)
        
        for method in estimation_methods:
            H_est = self.results['channel_estimates'][modulation][method]['estimated']
            H_true = self.results['channel_estimates'][modulation][method]['true']
            
            error = np.abs(H_true - H_est)
            error_mean = np.mean(error, axis=(0, 1))  # Average over TX-RX pairs
            
            plt.plot(error_mean, label=f'{method.upper()}', linewidth=2)
        
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Mean Estimation Error')
        plt.title('Channel Estimation Error vs Subcarrier')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('advanced_plots/channel_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_image_quality_metrics(self):
        """Plot image quality metrics"""
        quality_metrics = self.calculate_image_quality_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['psnr', 'ncc', 'ssim', 'mse_image']
        metric_names = ['PSNR (dB)', 'NCC', 'SSIM', 'MSE']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            x_pos = np.arange(len(self.modulation_schemes))
            width = 0.15
            
            for i, method in enumerate(self.estimation_methods):
                values = [quality_metrics[mod][method][metric] for mod in self.modulation_schemes]
                ax.bar(x_pos + i * width, values, width, label=method.upper())
            
            ax.set_xlabel('Modulation Scheme')
            ax.set_ylabel(name)
            ax.set_title(f'Image Quality: {name}')
            ax.set_xticks(x_pos + width * 2)
            ax.set_xticklabels(self.modulation_schemes)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_plots/image_quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_snr_threshold_analysis(self):
        """Plot SNR threshold analysis"""
        additional_metrics = self.calculate_additional_metrics()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        thresholds = ['snr_threshold_ber_01', 'snr_threshold_ber_001']
        threshold_names = ['BER ≤ 0.1', 'BER ≤ 0.01']
        
        for idx, (threshold, name) in enumerate(zip(thresholds, threshold_names)):
            ax = axes[idx]
            
            x_pos = np.arange(len(self.modulation_schemes))
            width = 0.25
            
            for i, method in enumerate(['ls', 'mmse', 'lmse']):
                values = []
                for mod in self.modulation_schemes:
                    val = additional_metrics[mod][method][threshold]
                    values.append(val if val is not None else 35)  # Use 35 if threshold not reached
                
                ax.bar(x_pos + i * width, values, width, label=method.upper())
            
            ax.set_xlabel('Modulation Scheme')
            ax.set_ylabel('SNR Threshold (dB)')
            ax.set_title(f'SNR Threshold for {name}')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(self.modulation_schemes)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 35)
        
        plt.tight_layout()
        plt.savefig('advanced_plots/snr_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_modulation_comparison(self):
        """Plot modulation scheme comparison"""
        additional_metrics = self.calculate_additional_metrics()
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Average BER comparison
        plt.subplot(2, 2, 1)
        methods = ['ls', 'mmse', 'lmse']
        x_pos = np.arange(len(self.modulation_schemes))
        width = 0.25
        
        for i, method in enumerate(methods):
            values = [additional_metrics[mod][method]['ber_mean'] for mod in self.modulation_schemes]
            plt.bar(x_pos + i * width, values, width, label=method.upper())
        
        plt.xlabel('Modulation Scheme')
        plt.ylabel('Average BER')
        plt.title('Average BER by Modulation Scheme')
        plt.xticks(x_pos + width, self.modulation_schemes)
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average SER comparison
        plt.subplot(2, 2, 2)
        for i, method in enumerate(methods):
            values = [additional_metrics[mod][method]['ser_mean'] for mod in self.modulation_schemes]
            plt.bar(x_pos + i * width, values, width, label=method.upper())
        
        plt.xlabel('Modulation Scheme')
        plt.ylabel('Average SER')
        plt.title('Average SER by Modulation Scheme')
        plt.xticks(x_pos + width, self.modulation_schemes)
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Spectral efficiency vs BER
        plt.subplot(2, 2, 3)
        spectral_efficiency = [1, 2, 4, 5, 6]  # bits per symbol
        
        for method in methods:
            ber_values = [additional_metrics[mod][method]['ber_mean'] for mod in self.modulation_schemes]
            plt.plot(spectral_efficiency, ber_values, 'o-', label=method.upper(), linewidth=2)
        
        plt.xlabel('Spectral Efficiency (bits/symbol)')
        plt.ylabel('Average BER')
        plt.title('Spectral Efficiency vs BER Trade-off')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        
        # Plot 4: Channel estimation MSE comparison
        plt.subplot(2, 2, 4)
        for i, method in enumerate(methods):
            values = [additional_metrics[mod][method]['mse_mean'] for mod in self.modulation_schemes]
            plt.bar(x_pos + i * width, values, width, label=method.upper())
        
        plt.xlabel('Modulation Scheme')
        plt.ylabel('Average Channel Estimation MSE')
        plt.title('Channel Estimation MSE by Modulation Scheme')
        plt.xticks(x_pos + width, self.modulation_schemes)
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_plots/modulation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main analysis function"""
    print("Starting MIMO-OFDM Analysis...")
    
    # Initialize analyzer
    analyzer = MIMOOFDMAnalyzer()
    
    if analyzer.results is None:
        print("No simulation results found. Please run the simulation first.")
        return
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report()
    
    # Create advanced plots
    analyzer.create_advanced_plots()
    
    print("\nAnalysis completed successfully!")
    print("Check 'advanced_plots' directory for additional visualizations.")

if __name__ == "__main__":
    main()