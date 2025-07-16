%% MIMO-OFDM System with Channel Estimation
% Implementation based on IEEE paper: "Channel Estimation using LS and MMSE 
% Channel Estimation Techniques for MIMO-OFDM Systems" (2022)
% Author: Advanced AI Implementation
% Date: 2024

clear all; close all; clc;

%% System Parameters
params = struct();
params.N_fft = 64;           % FFT size
params.N_cp = 16;            % Cyclic prefix length
params.N_tx = 2;             % Number of transmit antennas
params.N_rx = 2;             % Number of receive antennas
params.N_pilot = 16;         % Number of pilot subcarriers
params.N_data = params.N_fft - params.N_pilot; % Data subcarriers
params.SNR_dB = 10:5:30;     % SNR range in dB
params.N_symbols = 100;      % Number of OFDM symbols
params.pilot_spacing = 4;    % Pilot spacing for comb-type

% Modulation schemes
modulation_types = {'BPSK', 'QPSK', 'QAM16', 'QAM32', 'QAM64'};
modulation_orders = [2, 4, 16, 32, 64];

%% Load and prepare test image
try
    test_image = imread('test_image.png');
    if size(test_image, 3) == 3
        test_image = rgb2gray(test_image);
    end
catch
    % Create a test pattern if no image is available
    test_image = uint8(255 * checkerboard(8, 8));
end

% Resize image to manageable size
test_image = imresize(test_image, [64, 64]);
image_bits = reshape(de2bi(test_image(:), 8, 'left-msb'), [], 1);

%% Initialize results storage
results = struct();
for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    results.(mod_type) = struct();
    results.(mod_type).BER_perfect = zeros(1, length(params.SNR_dB));
    results.(mod_type).BER_LS = zeros(1, length(params.SNR_dB));
    results.(mod_type).BER_MMSE = zeros(1, length(params.SNR_dB));
    results.(mod_type).BER_LMSE = zeros(1, length(params.SNR_dB));
    results.(mod_type).BER_no_est = zeros(1, length(params.SNR_dB));
    results.(mod_type).SER_perfect = zeros(1, length(params.SNR_dB));
    results.(mod_type).SER_LS = zeros(1, length(params.SNR_dB));
    results.(mod_type).SER_MMSE = zeros(1, length(params.SNR_dB));
    results.(mod_type).SER_LMSE = zeros(1, length(params.SNR_dB));
    results.(mod_type).SER_no_est = zeros(1, length(params.SNR_dB));
    results.(mod_type).MSE_LS = zeros(1, length(params.SNR_dB));
    results.(mod_type).MSE_MMSE = zeros(1, length(params.SNR_dB));
    results.(mod_type).MSE_LMSE = zeros(1, length(params.SNR_dB));
    results.(mod_type).received_images = cell(length(params.SNR_dB), 4); % 4 estimation methods
end

%% Main Simulation Loop
fprintf('Starting MIMO-OFDM Simulation...\n');

for mod_idx = 1:length(modulation_types)
    mod_type = modulation_types{mod_idx};
    mod_order = modulation_orders(mod_idx);
    
    fprintf('Processing %s modulation...\n', mod_type);
    
    % Prepare image data for current modulation
    bits_per_symbol = log2(mod_order);
    required_bits = ceil(length(image_bits) / bits_per_symbol) * bits_per_symbol;
    padded_bits = [image_bits; zeros(required_bits - length(image_bits), 1)];
    
    for snr_idx = 1:length(params.SNR_dB)
        SNR_dB = params.SNR_dB(snr_idx);
        
        fprintf('  SNR = %d dB\n', SNR_dB);
        
        % Initialize error counters
        total_bits = 0;
        total_symbols = 0;
        errors_perfect = 0; errors_LS = 0; errors_MMSE = 0; errors_LMSE = 0; errors_no_est = 0;
        symbol_errors_perfect = 0; symbol_errors_LS = 0; symbol_errors_MMSE = 0; 
        symbol_errors_LMSE = 0; symbol_errors_no_est = 0;
        mse_LS = 0; mse_MMSE = 0; mse_LMSE = 0;
        
        % Process image data in chunks
        chunk_size = params.N_data * bits_per_symbol;
        num_chunks = ceil(length(padded_bits) / chunk_size);
        
        received_bits_perfect = zeros(size(padded_bits));
        received_bits_LS = zeros(size(padded_bits));
        received_bits_MMSE = zeros(size(padded_bits));
        received_bits_LMSE = zeros(size(padded_bits));
        received_bits_no_est = zeros(size(padded_bits));
        
        for chunk_idx = 1:num_chunks
            start_idx = (chunk_idx - 1) * chunk_size + 1;
            end_idx = min(chunk_idx * chunk_size, length(padded_bits));
            chunk_bits = padded_bits(start_idx:end_idx);
            
            % Pad chunk if necessary
            if length(chunk_bits) < chunk_size
                chunk_bits = [chunk_bits; zeros(chunk_size - length(chunk_bits), 1)];
            end
            
            % Transmitter processing
            [tx_signal, pilot_symbols, data_symbols, H_true] = ...
                transmitter_processing(chunk_bits, mod_type, params);
            
            % Channel and noise
            [rx_signal, noise_var] = channel_and_noise(tx_signal, H_true, SNR_dB, params);
            
            % Receiver processing with different estimation methods
            [rx_bits_perfect, rx_syms_perfect] = receiver_processing(...
                rx_signal, pilot_symbols, data_symbols, H_true, mod_type, params, 'perfect');
            [rx_bits_LS, rx_syms_LS, H_est_LS] = receiver_processing(...
                rx_signal, pilot_symbols, data_symbols, H_true, mod_type, params, 'LS');
            [rx_bits_MMSE, rx_syms_MMSE, H_est_MMSE] = receiver_processing(...
                rx_signal, pilot_symbols, data_symbols, H_true, mod_type, params, 'MMSE', noise_var);
            [rx_bits_LMSE, rx_syms_LMSE, H_est_LMSE] = receiver_processing(...
                rx_signal, pilot_symbols, data_symbols, H_true, mod_type, params, 'LMSE', noise_var);
            [rx_bits_no_est, rx_syms_no_est] = receiver_processing(...
                rx_signal, pilot_symbols, data_symbols, ones(size(H_true)), mod_type, params, 'none');
            
            % Store received bits
            received_bits_perfect(start_idx:end_idx) = rx_bits_perfect(1:length(chunk_bits));
            received_bits_LS(start_idx:end_idx) = rx_bits_LS(1:length(chunk_bits));
            received_bits_MMSE(start_idx:end_idx) = rx_bits_MMSE(1:length(chunk_bits));
            received_bits_LMSE(start_idx:end_idx) = rx_bits_LMSE(1:length(chunk_bits));
            received_bits_no_est(start_idx:end_idx) = rx_bits_no_est(1:length(chunk_bits));
            
            % Calculate errors
            bit_errors_perfect = sum(chunk_bits ~= rx_bits_perfect(1:length(chunk_bits)));
            bit_errors_LS = sum(chunk_bits ~= rx_bits_LS(1:length(chunk_bits)));
            bit_errors_MMSE = sum(chunk_bits ~= rx_bits_MMSE(1:length(chunk_bits)));
            bit_errors_LMSE = sum(chunk_bits ~= rx_bits_LMSE(1:length(chunk_bits)));
            bit_errors_no_est = sum(chunk_bits ~= rx_bits_no_est(1:length(chunk_bits)));
            
            errors_perfect = errors_perfect + bit_errors_perfect;
            errors_LS = errors_LS + bit_errors_LS;
            errors_MMSE = errors_MMSE + bit_errors_MMSE;
            errors_LMSE = errors_LMSE + bit_errors_LMSE;
            errors_no_est = errors_no_est + bit_errors_no_est;
            
            % Symbol errors
            symbol_errors_perfect = symbol_errors_perfect + sum(any(data_symbols ~= rx_syms_perfect, 1));
            symbol_errors_LS = symbol_errors_LS + sum(any(data_symbols ~= rx_syms_LS, 1));
            symbol_errors_MMSE = symbol_errors_MMSE + sum(any(data_symbols ~= rx_syms_MMSE, 1));
            symbol_errors_LMSE = symbol_errors_LMSE + sum(any(data_symbols ~= rx_syms_LMSE, 1));
            symbol_errors_no_est = symbol_errors_no_est + sum(any(data_symbols ~= rx_syms_no_est, 1));
            
            % MSE calculation
            if exist('H_est_LS', 'var')
                mse_LS = mse_LS + mean(abs(H_true - H_est_LS).^2, 'all');
            end
            if exist('H_est_MMSE', 'var')
                mse_MMSE = mse_MMSE + mean(abs(H_true - H_est_MMSE).^2, 'all');
            end
            if exist('H_est_LMSE', 'var')
                mse_LMSE = mse_LMSE + mean(abs(H_true - H_est_LMSE).^2, 'all');
            end
            
            total_bits = total_bits + length(chunk_bits);
            total_symbols = total_symbols + size(data_symbols, 2);
        end
        
        % Calculate final metrics
        results.(mod_type).BER_perfect(snr_idx) = errors_perfect / total_bits;
        results.(mod_type).BER_LS(snr_idx) = errors_LS / total_bits;
        results.(mod_type).BER_MMSE(snr_idx) = errors_MMSE / total_bits;
        results.(mod_type).BER_LMSE(snr_idx) = errors_LMSE / total_bits;
        results.(mod_type).BER_no_est(snr_idx) = errors_no_est / total_bits;
        
        results.(mod_type).SER_perfect(snr_idx) = symbol_errors_perfect / total_symbols;
        results.(mod_type).SER_LS(snr_idx) = symbol_errors_LS / total_symbols;
        results.(mod_type).SER_MMSE(snr_idx) = symbol_errors_MMSE / total_symbols;
        results.(mod_type).SER_LMSE(snr_idx) = symbol_errors_LMSE / total_symbols;
        results.(mod_type).SER_no_est(snr_idx) = symbol_errors_no_est / total_symbols;
        
        results.(mod_type).MSE_LS(snr_idx) = mse_LS / num_chunks;
        results.(mod_type).MSE_MMSE(snr_idx) = mse_MMSE / num_chunks;
        results.(mod_type).MSE_LMSE(snr_idx) = mse_LMSE / num_chunks;
        
        % Reconstruct images
        results.(mod_type).received_images{snr_idx, 1} = reconstruct_image(received_bits_perfect, size(test_image));
        results.(mod_type).received_images{snr_idx, 2} = reconstruct_image(received_bits_LS, size(test_image));
        results.(mod_type).received_images{snr_idx, 3} = reconstruct_image(received_bits_MMSE, size(test_image));
        results.(mod_type).received_images{snr_idx, 4} = reconstruct_image(received_bits_LMSE, size(test_image));
    end
end

%% Save results and generate plots
save('mimo_ofdm_results.mat', 'results', 'params', 'test_image', 'modulation_types');
generate_all_plots(results, params, test_image, modulation_types);

fprintf('Simulation completed successfully!\n');
fprintf('Results saved to mimo_ofdm_results.mat\n');
fprintf('Plots generated and saved as PNG files\n');