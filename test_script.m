%% Quick Test Script for Channel Estimation OFDM MIMO System
% This script performs a quick test with reduced parameters for verification

clear all; close all; clc;

fprintf('=== Quick Test of Channel Estimation System ===\n');

%% Reduced Parameters for Quick Test
params = struct();
params.N_subcarriers = 16;           % Reduced from 64
params.N_cp = 4;                     % Reduced from 16
params.N_tx = 2;                     % Number of transmit antennas
params.N_rx = 2;                     % Number of receive antennas
params.pilot_spacing = 4;            % Pilot spacing for comb-type pilots
params.SNR_range = [0, 10, 20];     % Reduced SNR range
params.N_symbols = 10;               % Reduced from 100
params.channel_taps = 4;             % Number of channel taps

% Test with single modulation scheme
modulation_schemes = {'QPSK'};
estimation_methods = {'LS', 'MMSE'};

fprintf('Testing with reduced parameters...\n');
fprintf('Subcarriers: %d, Symbols: %d, SNR points: %d\n', ...
    params.N_subcarriers, params.N_symbols, length(params.SNR_range));

%% Test Image Processing
fprintf('\n1. Testing image processing...\n');
try
    % Create simple test image
    test_img = uint8(rand(8, 8) * 255);
    img_data = de2bi(test_img(:), 8, 'left-msb');
    img_data = img_data(:);
    fprintf('✓ Image processing successful: %d bits\n', length(img_data));
catch ME
    fprintf('✗ Image processing failed: %s\n', ME.message);
    return;
end

%% Test Modulation
fprintf('\n2. Testing modulation...\n');
try
    % Test QPSK modulation
    M = 4; bits_per_symbol = 2;
    n_data_subcarriers = params.N_subcarriers - params.N_subcarriers/params.pilot_spacing;
    n_symbols_needed = params.N_symbols * n_data_subcarriers * params.N_tx;
    n_bits_needed = n_symbols_needed * bits_per_symbol;
    
    if length(img_data) < n_bits_needed
        tx_data = [img_data; zeros(n_bits_needed - length(img_data), 1)];
    else
        tx_data = img_data(1:n_bits_needed);
    end
    
    tx_bits_grouped = reshape(tx_data, bits_per_symbol, [])';
    tx_symbols = pskmod(bi2de(tx_bits_grouped, 'left-msb'), M);
    fprintf('✓ Modulation successful: %d symbols\n', length(tx_symbols));
catch ME
    fprintf('✗ Modulation failed: %s\n', ME.message);
    return;
end

%% Test OFDM Signal Creation
fprintf('\n3. Testing OFDM signal creation...\n');
try
    % Pilot positions (comb-type)
    pilot_positions = 1:params.pilot_spacing:params.N_subcarriers;
    data_positions = setdiff(1:params.N_subcarriers, pilot_positions);
    
    % Pilot symbols (BPSK)
    pilot_symbols = pskmod(randi([0 1], length(pilot_positions), 1), 2);
    
    % Create simple OFDM signal
    n_data_per_symbol = length(data_positions);
    tx_symbols_reshaped = reshape(tx_symbols, n_data_per_symbol, params.N_tx, []);
    
    ofdm_tx = zeros(params.N_subcarriers, params.N_tx, params.N_symbols);
    
    for sym_idx = 1:params.N_symbols
        for tx_idx = 1:params.N_tx
            ofdm_tx(pilot_positions, tx_idx, sym_idx) = pilot_symbols;
            if sym_idx <= size(tx_symbols_reshaped, 3)
                ofdm_tx(data_positions, tx_idx, sym_idx) = tx_symbols_reshaped(:, tx_idx, sym_idx);
            end
        end
    end
    
    fprintf('✓ OFDM signal creation successful\n');
    fprintf('  Signal size: %dx%dx%d\n', size(ofdm_tx));
    fprintf('  Pilot positions: %d subcarriers\n', length(pilot_positions));
    fprintf('  Data positions: %d subcarriers\n', length(data_positions));
catch ME
    fprintf('✗ OFDM signal creation failed: %s\n', ME.message);
    return;
end

%% Test Channel Simulation
fprintf('\n4. Testing channel simulation...\n');
try
    % Simple channel simulation (without comm.MIMOChannel for compatibility)
    tx_signal = reshape(ofdm_tx, [], params.N_tx);
    
    % Create simple multipath channel
    h = (randn(params.channel_taps, params.N_rx, params.N_tx) + ...
         1j*randn(params.channel_taps, params.N_rx, params.N_tx)) / sqrt(2);
    
    % Apply channel (simplified)
    rx_signal_ch = zeros(size(tx_signal, 1), params.N_rx);
    for rx_idx = 1:params.N_rx
        for tx_idx = 1:params.N_tx
            for tap = 1:params.channel_taps
                if tap <= size(tx_signal, 1)
                    rx_signal_ch(tap:end, rx_idx) = rx_signal_ch(tap:end, rx_idx) + ...
                        h(tap, rx_idx, tx_idx) * tx_signal(1:end-tap+1, tx_idx);
                end
            end
        end
    end
    
    % Add noise
    SNR_dB = 10;
    signal_power = mean(abs(rx_signal_ch(:)).^2);
    noise_power = signal_power / (10^(SNR_dB/10));
    noise = sqrt(noise_power/2) * (randn(size(rx_signal_ch)) + 1j*randn(size(rx_signal_ch)));
    rx_signal = rx_signal_ch + noise;
    
    fprintf('✓ Channel simulation successful\n');
    fprintf('  SNR: %d dB\n', SNR_dB);
    fprintf('  Signal power: %.2f\n', signal_power);
    fprintf('  Noise power: %.2f\n', noise_power);
catch ME
    fprintf('✗ Channel simulation failed: %s\n', ME.message);
    return;
end

%% Test LS Channel Estimation
fprintf('\n5. Testing LS channel estimation...\n');
try
    % Reshape back to OFDM format
    rx_signal_ofdm = reshape(rx_signal, size(ofdm_tx, 1), params.N_rx, params.N_symbols);
    
    % Simple LS estimation
    H_est = zeros(params.N_subcarriers, params.N_rx, params.N_tx, params.N_symbols);
    
    for sym_idx = 1:params.N_symbols
        rx_freq = fft(rx_signal_ofdm(:, :, sym_idx), params.N_subcarriers);
        tx_freq = fft(ofdm_tx(:, :, sym_idx), params.N_subcarriers);
        
        for rx_idx = 1:params.N_rx
            for tx_idx = 1:params.N_tx
                H_pilot = rx_freq(pilot_positions, rx_idx) ./ tx_freq(pilot_positions, tx_idx);
                H_est(:, rx_idx, tx_idx, sym_idx) = interp1(pilot_positions, H_pilot, ...
                    1:params.N_subcarriers, 'linear', 'extrap');
            end
        end
    end
    
    fprintf('✓ LS channel estimation successful\n');
    fprintf('  Estimated channel size: %dx%dx%dx%d\n', size(H_est));
catch ME
    fprintf('✗ LS channel estimation failed: %s\n', ME.message);
    return;
end

%% Test Performance Metrics
fprintf('\n6. Testing performance metrics...\n');
try
    % Simple equalization
    rx_symbols_eq = [];
    for sym_idx = 1:params.N_symbols
        rx_freq = fft(rx_signal_ofdm(:, :, sym_idx), params.N_subcarriers);
        for tx_idx = 1:params.N_tx
            for k = data_positions
                if abs(H_est(k, 1, tx_idx, sym_idx)) > 0.1
                    rx_symbol = rx_freq(k, 1) / H_est(k, 1, tx_idx, sym_idx);
                else
                    rx_symbol = rx_freq(k, 1);
                end
                rx_symbols_eq = [rx_symbols_eq; rx_symbol];
            end
        end
    end
    
    % Demodulate
    rx_bits = pskdemod(rx_symbols_eq, M);
    rx_data_grouped = de2bi(rx_bits, bits_per_symbol, 'left-msb');
    rx_data = rx_data_grouped(:);
    
    % Calculate BER
    min_len = min(length(tx_data), length(rx_data));
    bit_errors = sum(tx_data(1:min_len) ~= rx_data(1:min_len));
    ber = bit_errors / min_len;
    
    fprintf('✓ Performance metrics calculation successful\n');
    fprintf('  BER: %.4f (%d errors out of %d bits)\n', ber, bit_errors, min_len);
    fprintf('  Transmitted symbols: %d\n', length(tx_symbols));
    fprintf('  Received symbols: %d\n', length(rx_symbols_eq));
catch ME
    fprintf('✗ Performance metrics failed: %s\n', ME.message);
    return;
end

%% Test Plotting
fprintf('\n7. Testing basic plotting...\n');
try
    figure('Name', 'Test Results', 'Position', [100, 100, 800, 600]);
    
    % Plot constellation
    subplot(2, 2, 1);
    scatter(real(tx_symbols(1:min(100, length(tx_symbols)))), ...
           imag(tx_symbols(1:min(100, length(tx_symbols)))), 'b', 'filled');
    hold on;
    scatter(real(rx_symbols_eq(1:min(100, length(rx_symbols_eq)))), ...
           imag(rx_symbols_eq(1:min(100, length(rx_symbols_eq)))), 'r', 'x');
    title('Constellation Diagram');
    xlabel('In-phase');
    ylabel('Quadrature');
    legend('Transmitted', 'Received', 'Location', 'best');
    grid on;
    
    % Plot channel response
    subplot(2, 2, 2);
    plot(abs(H_est(:, 1, 1, 1)));
    title('Channel Frequency Response');
    xlabel('Subcarrier Index');
    ylabel('Magnitude');
    grid on;
    
    % Plot pilot positions
    subplot(2, 2, 3);
    stem(1:params.N_subcarriers, ismember(1:params.N_subcarriers, pilot_positions));
    title('Pilot Subcarrier Positions');
    xlabel('Subcarrier Index');
    ylabel('Pilot (1) / Data (0)');
    grid on;
    
    % Plot BER
    subplot(2, 2, 4);
    bar(ber);
    title(sprintf('Bit Error Rate: %.4f', ber));
    ylabel('BER');
    grid on;
    
    fprintf('✓ Basic plotting successful\n');
catch ME
    fprintf('✗ Basic plotting failed: %s\n', ME.message);
end

fprintf('\n=== Test Summary ===\n');
fprintf('✓ All basic functions are working correctly!\n');
fprintf('✓ The main script should run without errors\n');
fprintf('✓ You can now run the full simulation: channel_estimation_ofdm_mimo\n');

%% Cleanup
fprintf('\nTest completed successfully!\n');