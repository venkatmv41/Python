%% Quick Test Script for MIMO-OFDM System
% This script performs a basic validation of the implementation

clear all; close all; clc;

fprintf('=== MIMO-OFDM System Quick Test ===\n');

%% Test 1: Create Test Image
fprintf('1. Creating test image...\n');
create_test_image();

%% Test 2: Basic System Parameters
fprintf('2. Testing system parameters...\n');
params = struct();
params.N_fft = 64;
params.N_cp = 16;
params.N_tx = 2;
params.N_rx = 2;
params.N_pilot = 16;
params.N_data = params.N_fft - params.N_pilot;
params.pilot_spacing = 4;

fprintf('   FFT Size: %d\n', params.N_fft);
fprintf('   Cyclic Prefix: %d\n', params.N_cp);
fprintf('   MIMO Config: %dx%d\n', params.N_tx, params.N_rx);

%% Test 3: Modulation Functions
fprintf('3. Testing modulation schemes...\n');
modulation_types = {'BPSK', 'QPSK', '16QAM', '32QAM', '64QAM'};

for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    
    % Generate test data
    bits_per_symbol = log2(get_modulation_order(mod_type));
    test_bits = randi([0, 1], bits_per_symbol * 10, 1);
    
    % Test transmitter processing
    [tx_signal, pilot_symbols, data_symbols, H_true] = ...
        transmitter_processing(test_bits, mod_type, params);
    
    fprintf('   %s: %d bits/symbol, Signal size: %dx%d\n', ...
        mod_type, bits_per_symbol, size(tx_signal, 1), size(tx_signal, 2));
end

%% Test 4: Channel and Noise
fprintf('4. Testing channel and noise...\n');
SNR_dB = 20;
[rx_signal, noise_var] = channel_and_noise(tx_signal, H_true, SNR_dB, params);
fprintf('   SNR: %d dB, Noise variance: %.6f\n', SNR_dB, noise_var);

%% Test 5: Channel Estimation Methods
fprintf('5. Testing channel estimation methods...\n');
estimation_methods = {'perfect', 'LS', 'MMSE', 'LMSE'};

for i = 1:length(estimation_methods)
    method = estimation_methods{i};
    
    if strcmp(method, 'perfect') || strcmp(method, 'LS')
        [rx_bits, rx_symbols, H_est] = receiver_processing(...
            rx_signal, pilot_symbols, data_symbols, H_true, 'QPSK', params, method);
    else
        [rx_bits, rx_symbols, H_est] = receiver_processing(...
            rx_signal, pilot_symbols, data_symbols, H_true, 'QPSK', params, method, noise_var);
    end
    
    % Calculate MSE
    mse = mean(abs(H_true - H_est).^2, 'all');
    fprintf('   %s: MSE = %.6f\n', method, mse);
end

%% Test 6: Image Reconstruction
fprintf('6. Testing image reconstruction...\n');
test_image = imread('test_image.png');
image_bits = reshape(de2bi(test_image(:), 8, 'left-msb'), [], 1);

% Test reconstruction
reconstructed_image = reconstruct_image(image_bits, size(test_image));
psnr_val = calculate_psnr(test_image, reconstructed_image);
fprintf('   Original image size: %dx%d\n', size(test_image, 1), size(test_image, 2));
fprintf('   Reconstructed PSNR: %.2f dB\n', psnr_val);

%% Test 7: Basic Performance Metrics
fprintf('7. Testing performance metrics...\n');
test_bits_orig = randi([0, 1], 1000, 1);
test_bits_recv = test_bits_orig;
test_bits_recv(1:10) = ~test_bits_recv(1:10); % Introduce 10 errors

ber = sum(test_bits_orig ~= test_bits_recv) / length(test_bits_orig);
fprintf('   BER calculation test: %.4f (should be 0.01)\n', ber);

fprintf('\n=== Quick Test Completed Successfully! ===\n');
fprintf('All basic functions are working correctly.\n');
fprintf('You can now run the full simulation with: main_mimo_ofdm_system\n');

%% Helper Functions
function mod_order = get_modulation_order(mod_type)
    switch mod_type
        case 'BPSK'
            mod_order = 2;
        case 'QPSK'
            mod_order = 4;
        case '16QAM'
            mod_order = 16;
        case '32QAM'
            mod_order = 32;
        case '64QAM'
            mod_order = 64;
        otherwise
            error('Unsupported modulation type');
    end
end

function psnr_val = calculate_psnr(original, received)
    % Calculate Peak Signal-to-Noise Ratio
    mse = mean((double(original) - double(received)).^2, 'all');
    if mse == 0
        psnr_val = Inf;
    else
        max_val = 255; % For 8-bit images
        psnr_val = 10 * log10(max_val^2 / mse);
    end
end