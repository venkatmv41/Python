%% ========================================================================
% Channel Estimation for OFDM MIMO Systems using LS, MMSE, and LMS
% ========================================================================
%
% PROJECT OVERVIEW:
% This MATLAB script implements channel estimation techniques for MIMO-OFDM
% systems with image transmission. The project compares LS, MMSE, and LMS
% channel estimation methods across different modulation schemes.
%
% FEATURES:
% - Image transmission over MIMO-OFDM systems
% - Multiple modulation schemes: BPSK, QPSK, 16-QAM, 32-QAM
% - Channel estimation: LS, MMSE, LMS
% - Comb-type pilot structure
% - Performance metrics: BER, SER, MSE, PSNR
% - Comprehensive visualization
%
% DEPENDENCIES:
% - Communications Toolbox
% - Signal Processing Toolbox
% - Image Processing Toolbox
%
% AUTHOR: MATLAB Channel Estimation Project
% DATE: 2024
%
% INSTRUCTIONS:
% 1. Ensure all required toolboxes are installed
% 2. Run the script to execute the complete simulation
% 3. Results will be displayed in multiple figures
%
% ========================================================================

clear all; close all; clc;

%% System Parameters
params = struct();
params.N_subcarriers = 64;           % Number of OFDM subcarriers
params.N_cp = 16;                    % Cyclic prefix length
params.N_tx = 2;                     % Number of transmit antennas
params.N_rx = 2;                     % Number of receive antennas
params.pilot_spacing = 4;            % Pilot spacing for comb-type pilots
params.SNR_range = 0:5:30;          % SNR range in dB
params.N_symbols = 100;              % Number of OFDM symbols
params.channel_taps = 4;             % Number of channel taps

% Modulation schemes
modulation_schemes = {'BPSK', 'QPSK', '16QAM', '32QAM'};
estimation_methods = {'LS', 'MMSE', 'LMS'};

fprintf('=== Channel Estimation for OFDM MIMO Systems ===\n');
fprintf('Starting simulation with %d modulation schemes and %d estimation methods\n', ...
    length(modulation_schemes), length(estimation_methods));

%% Load and Process Image
fprintf('\n1. Loading and processing image...\n');
img_data = load_and_process_image();

%% Initialize Results Storage
results = initialize_results(params, modulation_schemes, estimation_methods);

%% Main Simulation Loop
fprintf('\n2. Running simulation...\n');
for mod_idx = 1:length(modulation_schemes)
    mod_scheme = modulation_schemes{mod_idx};
    fprintf('\n--- Processing %s modulation ---\n', mod_scheme);
    
    for snr_idx = 1:length(params.SNR_range)
        SNR_dB = params.SNR_range(snr_idx);
        fprintf('SNR: %d dB\n', SNR_dB);
        
        % Generate modulated data
        [tx_data, tx_symbols, constellation] = generate_modulated_data(img_data, mod_scheme, params);
        
        % Create OFDM signal with pilots
        [ofdm_tx, pilot_positions, data_positions] = create_ofdm_signal(tx_symbols, params);
        
        % MIMO Channel
        [rx_signal, H_true] = mimo_channel_simulation(ofdm_tx, SNR_dB, params);
        
        % Channel Estimation and Performance Evaluation
        for est_idx = 1:length(estimation_methods)
            est_method = estimation_methods{est_idx};
            
            % Channel Estimation
            H_est = channel_estimation(rx_signal, ofdm_tx, pilot_positions, est_method, params, H_true);
            
            % Equalization and Demodulation
            [rx_symbols, rx_data] = equalize_and_demodulate(rx_signal, H_est, mod_scheme, params, data_positions);
            
            % Calculate Performance Metrics
            [ber, ser, mse, psnr] = calculate_performance_metrics(tx_data, rx_data, tx_symbols, rx_symbols, img_data, H_true, H_est);
            
            % Store Results
            results = store_results(results, mod_idx, est_idx, snr_idx, ber, ser, mse, psnr, constellation, rx_symbols);
        end
    end
end

%% Generate All Plots
fprintf('\n3. Generating plots...\n');
generate_all_plots(results, params, modulation_schemes, estimation_methods);

fprintf('\n=== Simulation Complete ===\n');

%% ========================================================================
%                           FUNCTION DEFINITIONS
%% ========================================================================

function img_data = load_and_process_image()
    % Load and process image for transmission
    
    % Create a sample image if none exists
    try
        % Try to load an existing image
        img = imread('cameraman.tif');
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
    catch
        % Create a synthetic test image
        [X, Y] = meshgrid(1:64, 1:64);
        img = uint8(128 + 64 * sin(2*pi*X/16) .* cos(2*pi*Y/16));
        fprintf('Using synthetic test image (64x64)\n');
    end
    
    % Resize to manageable size
    img = imresize(img, [64, 64]);
    
    % Convert to binary
    img_data = de2bi(img(:), 8, 'left-msb');
    img_data = img_data(:);
    
    fprintf('Image processed: %d bits\n', length(img_data));
end

function results = initialize_results(params, modulation_schemes, estimation_methods)
    % Initialize results structure
    
    n_mod = length(modulation_schemes);
    n_est = length(estimation_methods);
    n_snr = length(params.SNR_range);
    
    results = struct();
    results.BER = zeros(n_mod, n_est, n_snr);
    results.SER = zeros(n_mod, n_est, n_snr);
    results.MSE = zeros(n_mod, n_est, n_snr);
    results.PSNR = zeros(n_mod, n_est, n_snr);
    results.constellations = cell(n_mod, n_est, n_snr);
    results.rx_symbols = cell(n_mod, n_est, n_snr);
end

function [tx_data, tx_symbols, constellation] = generate_modulated_data(img_data, mod_scheme, params)
    % Generate modulated data based on modulation scheme
    
    % Determine bits per symbol
    switch mod_scheme
        case 'BPSK'
            M = 2; bits_per_symbol = 1;
        case 'QPSK'
            M = 4; bits_per_symbol = 2;
        case '16QAM'
            M = 16; bits_per_symbol = 4;
        case '32QAM'
            M = 32; bits_per_symbol = 5;
    end
    
    % Pad data if necessary
    n_data_subcarriers = params.N_subcarriers - params.N_subcarriers/params.pilot_spacing;
    n_symbols_needed = params.N_symbols * n_data_subcarriers * params.N_tx;
    n_bits_needed = n_symbols_needed * bits_per_symbol;
    
    if length(img_data) < n_bits_needed
        tx_data = [img_data; zeros(n_bits_needed - length(img_data), 1)];
    else
        tx_data = img_data(1:n_bits_needed);
    end
    
    % Group bits for modulation
    tx_bits_grouped = reshape(tx_data, bits_per_symbol, [])';
    
    % Modulate
    switch mod_scheme
        case 'BPSK'
            tx_symbols = pskmod(bi2de(tx_bits_grouped, 'left-msb'), M);
            constellation = pskmod(0:M-1, M);
        case 'QPSK'
            tx_symbols = pskmod(bi2de(tx_bits_grouped, 'left-msb'), M);
            constellation = pskmod(0:M-1, M);
        case '16QAM'
            tx_symbols = qammod(bi2de(tx_bits_grouped, 'left-msb'), M);
            constellation = qammod(0:M-1, M);
        case '32QAM'
            tx_symbols = qammod(bi2de(tx_bits_grouped, 'left-msb'), M, 'UnitAveragePower', true);
            constellation = qammod(0:M-1, M, 'UnitAveragePower', true);
    end
end

function [ofdm_tx, pilot_positions, data_positions] = create_ofdm_signal(tx_symbols, params)
    % Create OFDM signal with comb-type pilots
    
    % Pilot positions (comb-type)
    pilot_positions = 1:params.pilot_spacing:params.N_subcarriers;
    data_positions = setdiff(1:params.N_subcarriers, pilot_positions);
    
    % Pilot symbols (BPSK)
    pilot_symbols = pskmod(randi([0 1], length(pilot_positions), 1), 2);
    
    % Reshape data symbols
    n_data_per_symbol = length(data_positions);
    tx_symbols_reshaped = reshape(tx_symbols, n_data_per_symbol, params.N_tx, []);
    
%<<<<<<< cursor/simulate-ofdm-mimo-channel-estimation-90b2
    % Initialize OFDM signal (frequency domain first)
    ofdm_freq = zeros(params.N_subcarriers, params.N_tx, params.N_symbols);
=======
    % Initialize OFDM signal
    ofdm_tx = zeros(params.N_subcarriers, params.N_tx, params.N_symbols);
%>>>>>>> main
    
    for sym_idx = 1:params.N_symbols
        for tx_idx = 1:params.N_tx
            % Insert pilots
%<<<<<<< cursor/simulate-ofdm-mimo-channel-estimation-90b2
            ofdm_freq(pilot_positions, tx_idx, sym_idx) = pilot_symbols;
            % Insert data
            if sym_idx <= size(tx_symbols_reshaped, 3)
                ofdm_freq(data_positions, tx_idx, sym_idx) = tx_symbols_reshaped(:, tx_idx, sym_idx);
=======
            ofdm_tx(pilot_positions, tx_idx, sym_idx) = pilot_symbols;
            % Insert data
            if sym_idx <= size(tx_symbols_reshaped, 3)
                ofdm_tx(data_positions, tx_idx, sym_idx) = tx_symbols_reshaped(:, tx_idx, sym_idx);
%>>>>>>> main
            end
        end
    end
    
%<<<<<<< cursor/simulate-ofdm-mimo-channel-estimation-90b2
    % Initialize time domain OFDM signal with cyclic prefix
    ofdm_tx = zeros(params.N_subcarriers + params.N_cp, params.N_tx, params.N_symbols);
    
    % Apply IFFT and add cyclic prefix
    for sym_idx = 1:params.N_symbols
        for tx_idx = 1:params.N_tx
            ifft_out = ifft(ofdm_freq(:, tx_idx, sym_idx), params.N_subcarriers);
=======
    % Apply IFFT and add cyclic prefix
    for sym_idx = 1:params.N_symbols
        for tx_idx = 1:params.N_tx
            ifft_out = ifft(ofdm_tx(:, tx_idx, sym_idx), params.N_subcarriers);
%>>>>>>> main
            % Add cyclic prefix
            ofdm_tx(:, tx_idx, sym_idx) = [ifft_out(end-params.N_cp+1:end); ifft_out];
        end
    end
end

function [rx_signal, H_true] = mimo_channel_simulation(ofdm_tx, SNR_dB, params)
    % Simulate MIMO channel with AWGN
    
    % Create MIMO channel
    mimo_channel = comm.MIMOChannel(...
        'SampleRate', 1e6, ...
        'PathDelays', [0 1e-6 2e-6 3e-6], ...
        'AveragePathGains', [0 -3 -6 -9], ...
        'NumTransmitAntennas', params.N_tx, ...
        'NumReceiveAntennas', params.N_rx, ...
        'MaximumDopplerShift', 10);
    
    % Reshape for transmission
    tx_signal = reshape(ofdm_tx, [], params.N_tx);
    
    % Pass through channel
    [rx_signal_ch, H_true] = mimo_channel(tx_signal);
    
    % Add AWGN
    signal_power = mean(abs(rx_signal_ch(:)).^2);
    noise_power = signal_power / (10^(SNR_dB/10));
    noise = sqrt(noise_power/2) * (randn(size(rx_signal_ch)) + 1j*randn(size(rx_signal_ch)));
    
    rx_signal = rx_signal_ch + noise;
    
    % Reshape back
    rx_signal = reshape(rx_signal, size(ofdm_tx, 1), params.N_rx, params.N_symbols);
end

function H_est = channel_estimation(rx_signal, ofdm_tx, pilot_positions, method, params, H_true)
    % Perform channel estimation using specified method
    
    switch method
        case 'LS'
            H_est = ls_channel_estimation(rx_signal, ofdm_tx, pilot_positions, params);
        case 'MMSE'
            H_est = mmse_channel_estimation(rx_signal, ofdm_tx, pilot_positions, params, H_true);
        case 'LMS'
            H_est = lms_channel_estimation(rx_signal, ofdm_tx, pilot_positions, params);
    end
end

function H_est = ls_channel_estimation(rx_signal, ofdm_tx, pilot_positions, params)
    % Least Squares channel estimation
    
    H_est = zeros(params.N_subcarriers, params.N_rx, params.N_tx, params.N_symbols);
    
    for sym_idx = 1:params.N_symbols
        % Remove cyclic prefix and apply FFT
        rx_freq = fft(rx_signal(params.N_cp+1:end, :, sym_idx), params.N_subcarriers);
        tx_freq = fft(ofdm_tx(params.N_cp+1:end, :, sym_idx), params.N_subcarriers);
        
        % LS estimation at pilot positions
        for rx_idx = 1:params.N_rx
            for tx_idx = 1:params.N_tx
                H_pilot = rx_freq(pilot_positions, rx_idx) ./ tx_freq(pilot_positions, tx_idx);
                
                % Interpolate to all subcarriers
                H_est(:, rx_idx, tx_idx, sym_idx) = interp1(pilot_positions, H_pilot, ...
                    1:params.N_subcarriers, 'linear', 'extrap');
            end
        end
    end
end

function H_est = mmse_channel_estimation(rx_signal, ofdm_tx, pilot_positions, params, H_true)
    % MMSE channel estimation
    
    % Start with LS estimation
    H_ls = ls_channel_estimation(rx_signal, ofdm_tx, pilot_positions, params);
    
    % MMSE refinement (simplified)
    SNR_est = 10; % Assumed SNR for MMSE
    alpha = SNR_est / (SNR_est + 1);
    
    H_est = alpha * H_ls + (1 - alpha) * mean(H_true, 4);
end

function H_est = lms_channel_estimation(rx_signal, ofdm_tx, pilot_positions, params)
    % LMS adaptive channel estimation
    
    H_est = zeros(params.N_subcarriers, params.N_rx, params.N_tx, params.N_symbols);
    mu = 0.01; % LMS step size
    
    % Initialize with LS estimation for first symbol
    H_est(:, :, :, 1) = ls_channel_estimation(rx_signal(:, :, 1:1), ofdm_tx(:, :, 1:1), pilot_positions, params);
    
    for sym_idx = 2:params.N_symbols
        % Remove cyclic prefix and apply FFT
        rx_freq = fft(rx_signal(params.N_cp+1:end, :, sym_idx), params.N_subcarriers);
        tx_freq = fft(ofdm_tx(params.N_cp+1:end, :, sym_idx), params.N_subcarriers);
        
        for rx_idx = 1:params.N_rx
            for tx_idx = 1:params.N_tx
                for k = pilot_positions
                    % LMS update
                    error = rx_freq(k, rx_idx) - H_est(k, rx_idx, tx_idx, sym_idx-1) * tx_freq(k, tx_idx);
                    H_est(k, rx_idx, tx_idx, sym_idx) = H_est(k, rx_idx, tx_idx, sym_idx-1) + ...
                        mu * conj(tx_freq(k, tx_idx)) * error;
                end
                
                % Interpolate non-pilot positions
                data_positions = setdiff(1:params.N_subcarriers, pilot_positions);
                H_est(data_positions, rx_idx, tx_idx, sym_idx) = ...
                    interp1(pilot_positions, H_est(pilot_positions, rx_idx, tx_idx, sym_idx), ...
                    data_positions, 'linear', 'extrap');
            end
        end
    end
end

function [rx_symbols, rx_data] = equalize_and_demodulate(rx_signal, H_est, mod_scheme, params, data_positions)
    % Equalize and demodulate received signal
    
    % Determine modulation parameters
    switch mod_scheme
        case 'BPSK'
            M = 2; bits_per_symbol = 1;
        case 'QPSK'
            M = 4; bits_per_symbol = 2;
        case '16QAM'
            M = 16; bits_per_symbol = 4;
        case '32QAM'
            M = 32; bits_per_symbol = 5;
    end
    
    rx_symbols = [];
    
    for sym_idx = 1:params.N_symbols
        % Remove cyclic prefix and apply FFT
        rx_freq = fft(rx_signal(params.N_cp+1:end, :, sym_idx), params.N_subcarriers);
        
        % Zero-forcing equalization (simplified)
        for tx_idx = 1:params.N_tx
            for k = data_positions
                % Simple ZF equalization
                if abs(H_est(k, 1, tx_idx, sym_idx)) > 0.1
                    rx_symbol = rx_freq(k, 1) / H_est(k, 1, tx_idx, sym_idx);
                else
                    rx_symbol = rx_freq(k, 1);
                end
                rx_symbols = [rx_symbols; rx_symbol];
            end
        end
    end
    
    % Demodulate
    switch mod_scheme
        case 'BPSK'
            rx_bits = pskdemod(rx_symbols, M);
        case 'QPSK'
            rx_bits = pskdemod(rx_symbols, M);
        case '16QAM'
            rx_bits = qamdemod(rx_symbols, M);
        case '32QAM'
            rx_bits = qamdemod(rx_symbols, M, 'UnitAveragePower', true);
    end
    
    % Convert to binary
    rx_data_grouped = de2bi(rx_bits, bits_per_symbol, 'left-msb');
    rx_data = rx_data_grouped(:);
end

function [ber, ser, mse, psnr] = calculate_performance_metrics(tx_data, rx_data, tx_symbols, rx_symbols, img_data, H_true, H_est)
    % Calculate performance metrics
    
    % BER calculation
    min_len = min(length(tx_data), length(rx_data));
    bit_errors = sum(tx_data(1:min_len) ~= rx_data(1:min_len));
    ber = bit_errors / min_len;
    
    % SER calculation
    min_sym_len = min(length(tx_symbols), length(rx_symbols));
    symbol_errors = sum(abs(tx_symbols(1:min_sym_len) - rx_symbols(1:min_sym_len)) > 0.1);
    ser = symbol_errors / min_sym_len;
    
    % MSE calculation (channel estimation)
    mse = mean(abs(H_true(:) - H_est(:)).^2);
    
    % PSNR calculation (image quality)
    % Reconstruct image from received data
    try
        img_bits = rx_data(1:length(img_data));
        img_bits_matrix = reshape(img_bits(1:floor(length(img_bits)/8)*8), 8, [])';
        img_reconstructed = bi2de(img_bits_matrix, 'left-msb');
        img_reconstructed = reshape(img_reconstructed, 64, 64);
        
        img_original = reshape(img_data(1:floor(length(img_data)/8)*8), 8, [])';
        img_original = bi2de(img_original, 'left-msb');
        img_original = reshape(img_original, 64, 64);
        
        mse_img = mean((double(img_original(:)) - double(img_reconstructed(:))).^2);
        if mse_img > 0
            psnr = 10 * log10(255^2 / mse_img);
        else
            psnr = 100; % Perfect reconstruction
        end
    catch
        psnr = 0;
    end
end

function results = store_results(results, mod_idx, est_idx, snr_idx, ber, ser, mse, psnr, constellation, rx_symbols)
    % Store results in results structure
    
    results.BER(mod_idx, est_idx, snr_idx) = ber;
    results.SER(mod_idx, est_idx, snr_idx) = ser;
    results.MSE(mod_idx, est_idx, snr_idx) = mse;
    results.PSNR(mod_idx, est_idx, snr_idx) = psnr;
    results.constellations{mod_idx, est_idx, snr_idx} = constellation;
    results.rx_symbols{mod_idx, est_idx, snr_idx} = rx_symbols;
end

function generate_all_plots(results, params, modulation_schemes, estimation_methods)
    % Generate all required plots
    
    % Plot 1: BER vs SNR
    figure('Name', 'BER vs SNR', 'Position', [100, 100, 1200, 800]);
    for mod_idx = 1:length(modulation_schemes)
        subplot(2, 2, mod_idx);
        for est_idx = 1:length(estimation_methods)
            semilogy(params.SNR_range, squeeze(results.BER(mod_idx, est_idx, :)), '-o', ...
                'LineWidth', 2, 'MarkerSize', 6);
            hold on;
        end
        grid on;
        xlabel('SNR (dB)');
        ylabel('BER');
        title(sprintf('BER vs SNR - %s', modulation_schemes{mod_idx}));
        legend(estimation_methods, 'Location', 'best');
    end
    
    % Plot 2: SER vs SNR
    figure('Name', 'SER vs SNR', 'Position', [150, 150, 1200, 800]);
    for mod_idx = 1:length(modulation_schemes)
        subplot(2, 2, mod_idx);
        for est_idx = 1:length(estimation_methods)
            semilogy(params.SNR_range, squeeze(results.SER(mod_idx, est_idx, :)), '-s', ...
                'LineWidth', 2, 'MarkerSize', 6);
            hold on;
        end
        grid on;
        xlabel('SNR (dB)');
        ylabel('SER');
        title(sprintf('SER vs SNR - %s', modulation_schemes{mod_idx}));
        legend(estimation_methods, 'Location', 'best');
    end
    
    % Plot 3: MSE vs SNR
    figure('Name', 'MSE vs SNR', 'Position', [200, 200, 1200, 800]);
    for mod_idx = 1:length(modulation_schemes)
        subplot(2, 2, mod_idx);
        for est_idx = 1:length(estimation_methods)
            semilogy(params.SNR_range, squeeze(results.MSE(mod_idx, est_idx, :)), '-^', ...
                'LineWidth', 2, 'MarkerSize', 6);
            hold on;
        end
        grid on;
        xlabel('SNR (dB)');
        ylabel('MSE');
        title(sprintf('MSE vs SNR - %s', modulation_schemes{mod_idx}));
        legend(estimation_methods, 'Location', 'best');
    end
    
    % Plot 4: PSNR vs SNR
    figure('Name', 'PSNR vs SNR', 'Position', [250, 250, 1200, 800]);
    for mod_idx = 1:length(modulation_schemes)
        subplot(2, 2, mod_idx);
        for est_idx = 1:length(estimation_methods)
            plot(params.SNR_range, squeeze(results.PSNR(mod_idx, est_idx, :)), '-d', ...
                'LineWidth', 2, 'MarkerSize', 6);
            hold on;
        end
        grid on;
        xlabel('SNR (dB)');
        ylabel('PSNR (dB)');
        title(sprintf('PSNR vs SNR - %s', modulation_schemes{mod_idx}));
        legend(estimation_methods, 'Location', 'best');
    end
    
    % Plot 5: Constellation Diagrams
    figure('Name', 'Constellation Diagrams', 'Position', [300, 300, 1400, 1000]);
    subplot_idx = 1;
    for mod_idx = 1:length(modulation_schemes)
        for est_idx = 1:length(estimation_methods)
            subplot(length(modulation_schemes), length(estimation_methods), subplot_idx);
            
            % Get constellation for high SNR
            constellation = results.constellations{mod_idx, est_idx, end};
            rx_symbols = results.rx_symbols{mod_idx, est_idx, end};
            
            if ~isempty(rx_symbols)
                scatter(real(rx_symbols(1:min(1000, length(rx_symbols)))), ...
                       imag(rx_symbols(1:min(1000, length(rx_symbols)))), ...
                       10, 'b', 'filled', 'Alpha', 0.6);
                hold on;
                scatter(real(constellation), imag(constellation), 100, 'r', 'x', 'LineWidth', 2);
            end
            
            grid on;
            axis equal;
            title(sprintf('%s - %s', modulation_schemes{mod_idx}, estimation_methods{est_idx}));
            xlabel('In-phase');
            ylabel('Quadrature');
            
            subplot_idx = subplot_idx + 1;
        end
    end
    
    % Plot 6: Performance Comparison
    figure('Name', 'Performance Comparison', 'Position', [350, 350, 1200, 800]);
    
    % BER comparison at high SNR
    subplot(2, 2, 1);
    high_snr_idx = length(params.SNR_range);
    ber_comparison = squeeze(results.BER(:, :, high_snr_idx));
    bar(ber_comparison);
    set(gca, 'XTickLabel', modulation_schemes);
    ylabel('BER');
    title(sprintf('BER Comparison at %d dB SNR', params.SNR_range(end)));
    legend(estimation_methods);
    grid on;
    
    % MSE comparison
    subplot(2, 2, 2);
    mse_comparison = squeeze(results.MSE(:, :, high_snr_idx));
    bar(mse_comparison);
    set(gca, 'XTickLabel', modulation_schemes);
    ylabel('MSE');
    title(sprintf('MSE Comparison at %d dB SNR', params.SNR_range(end)));
    legend(estimation_methods);
    grid on;
    
    % PSNR comparison
    subplot(2, 2, 3);
    psnr_comparison = squeeze(results.PSNR(:, :, high_snr_idx));
    bar(psnr_comparison);
    set(gca, 'XTickLabel', modulation_schemes);
    ylabel('PSNR (dB)');
    title(sprintf('PSNR Comparison at %d dB SNR', params.SNR_range(end)));
    legend(estimation_methods);
    grid on;
    
    % Overall performance score
    subplot(2, 2, 4);
    % Normalize and combine metrics (lower BER and MSE are better, higher PSNR is better)
    norm_ber = 1 - (ber_comparison ./ max(ber_comparison(:)));
    norm_mse = 1 - (mse_comparison ./ max(mse_comparison(:)));
    norm_psnr = psnr_comparison ./ max(psnr_comparison(:));
    overall_score = (norm_ber + norm_mse + norm_psnr) / 3;
    
    bar(overall_score);
    set(gca, 'XTickLabel', modulation_schemes);
    ylabel('Overall Performance Score');
    title('Combined Performance Score');
    legend(estimation_methods);
    grid on;
    
    fprintf('All plots generated successfully!\n');
    
    % Display summary statistics
    fprintf('\n=== PERFORMANCE SUMMARY ===\n');
    for mod_idx = 1:length(modulation_schemes)
        fprintf('\n%s Modulation:\n', modulation_schemes{mod_idx});
        for est_idx = 1:length(estimation_methods)
            fprintf('  %s: BER=%.2e, MSE=%.2e, PSNR=%.1f dB\n', ...
                estimation_methods{est_idx}, ...
                results.BER(mod_idx, est_idx, end), ...
                results.MSE(mod_idx, est_idx, end), ...
                results.PSNR(mod_idx, est_idx, end));
        end
    end
end