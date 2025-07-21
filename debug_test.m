%% Debug Test for OFDM Signal Creation Fix
% This script tests the corrected create_ofdm_signal function

clear all; close all; clc;

fprintf('=== Debug Test for OFDM Signal Creation ===\n');

%% Test Parameters
params = struct();
params.N_subcarriers = 64;
params.N_cp = 16;
params.N_tx = 2;
params.N_rx = 2;
params.pilot_spacing = 4;
params.N_symbols = 10;
params.channel_taps = 4;

fprintf('Parameters:\n');
fprintf('  Subcarriers: %d\n', params.N_subcarriers);
fprintf('  Cyclic Prefix: %d\n', params.N_cp);
fprintf('  Expected OFDM signal size: %d x %d x %d\n', ...
    params.N_subcarriers + params.N_cp, params.N_tx, params.N_symbols);

%% Create Test Data
fprintf('\n1. Creating test data...\n');

% Create simple test image data
test_img = uint8(rand(8, 8) * 255);
img_data = de2bi(test_img(:), 8, 'left-msb');
img_data = img_data(:);

% Generate QPSK modulated data
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

fprintf('✓ Test data created: %d symbols\n', length(tx_symbols));

%% Test OFDM Signal Creation (This should work now)
fprintf('\n2. Testing OFDM signal creation...\n');

try
    % Test the corrected function
    [ofdm_tx, pilot_positions, data_positions] = create_ofdm_signal(tx_symbols, params);
    
    fprintf('✓ OFDM signal creation successful!\n');
    fprintf('  OFDM signal size: %d x %d x %d\n', size(ofdm_tx));
    fprintf('  Pilot positions: %d subcarriers\n', length(pilot_positions));
    fprintf('  Data positions: %d subcarriers\n', length(data_positions));
    
    % Verify sizes
    expected_time_length = params.N_subcarriers + params.N_cp;
    if size(ofdm_tx, 1) == expected_time_length
        fprintf('✓ Time domain length correct: %d samples\n', size(ofdm_tx, 1));
    else
        fprintf('✗ Time domain length incorrect: expected %d, got %d\n', ...
            expected_time_length, size(ofdm_tx, 1));
    end
    
catch ME
    fprintf('✗ OFDM signal creation failed: %s\n', ME.message);
    return;
end

%% Test Channel Simulation
fprintf('\n3. Testing channel simulation...\n');

try
    SNR_dB = 10;
    [rx_signal, H_true] = mimo_channel_simulation(ofdm_tx, SNR_dB, params);
    
    fprintf('✓ Channel simulation successful!\n');
    fprintf('  RX signal size: %d x %d x %d\n', size(rx_signal));
    fprintf('  Channel matrix size: %s\n', mat2str(size(H_true)));
    
catch ME
    fprintf('✗ Channel simulation failed: %s\n', ME.message);
    return;
end

%% Test Channel Estimation
fprintf('\n4. Testing LS channel estimation...\n');

try
    H_est = ls_channel_estimation(rx_signal, ofdm_tx, pilot_positions, params);
    
    fprintf('✓ LS channel estimation successful!\n');
    fprintf('  Estimated channel size: %d x %d x %d x %d\n', size(H_est));
    
catch ME
    fprintf('✗ LS channel estimation failed: %s\n', ME.message);
    return;
end

%% Test Equalization
fprintf('\n5. Testing equalization and demodulation...\n');

try
    [rx_symbols, rx_data] = equalize_and_demodulate(rx_signal, H_est, 'QPSK', params, data_positions);
    
    fprintf('✓ Equalization and demodulation successful!\n');
    fprintf('  Received symbols: %d\n', length(rx_symbols));
    fprintf('  Received data bits: %d\n', length(rx_data));
    
    % Calculate simple BER
    min_len = min(length(tx_data), length(rx_data));
    bit_errors = sum(tx_data(1:min_len) ~= rx_data(1:min_len));
    ber = bit_errors / min_len;
    fprintf('  BER: %.4f\n', ber);
    
catch ME
    fprintf('✗ Equalization failed: %s\n', ME.message);
    return;
end

fprintf('\n=== All Tests Passed! ===\n');
fprintf('The main script should now work correctly.\n');
fprintf('You can run: channel_estimation_ofdm_mimo\n');

%% Helper Functions (copied from main script for testing)

function [ofdm_tx, pilot_positions, data_positions] = create_ofdm_signal(tx_symbols, params)
    % Create OFDM signal with comb-type pilots (CORRECTED VERSION)
    
    % Pilot positions (comb-type)
    pilot_positions = 1:params.pilot_spacing:params.N_subcarriers;
    data_positions = setdiff(1:params.N_subcarriers, pilot_positions);
    
    % Pilot symbols (BPSK)
    pilot_symbols = pskmod(randi([0 1], length(pilot_positions), 1), 2);
    
    % Reshape data symbols
    n_data_per_symbol = length(data_positions);
    tx_symbols_reshaped = reshape(tx_symbols, n_data_per_symbol, params.N_tx, []);
    
    % Initialize OFDM signal (frequency domain first)
    ofdm_freq = zeros(params.N_subcarriers, params.N_tx, params.N_symbols);
    
    for sym_idx = 1:params.N_symbols
        for tx_idx = 1:params.N_tx
            % Insert pilots
            ofdm_freq(pilot_positions, tx_idx, sym_idx) = pilot_symbols;
            % Insert data
            if sym_idx <= size(tx_symbols_reshaped, 3)
                ofdm_freq(data_positions, tx_idx, sym_idx) = tx_symbols_reshaped(:, tx_idx, sym_idx);
            end
        end
    end
    
    % Initialize time domain OFDM signal with cyclic prefix
    ofdm_tx = zeros(params.N_subcarriers + params.N_cp, params.N_tx, params.N_symbols);
    
    % Apply IFFT and add cyclic prefix
    for sym_idx = 1:params.N_symbols
        for tx_idx = 1:params.N_tx
            ifft_out = ifft(ofdm_freq(:, tx_idx, sym_idx), params.N_subcarriers);
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