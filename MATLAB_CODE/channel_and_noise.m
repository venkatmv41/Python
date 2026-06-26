function [rx_signal, noise_var] = channel_and_noise(tx_signal, H_true, SNR_dB, params)
%% Channel and Noise Processing for MIMO-OFDM System
% Inputs:
%   tx_signal - Transmitted signal from all antennas
%   H_true - True channel matrix
%   SNR_dB - Signal-to-noise ratio in dB
%   params - System parameters structure
% Outputs:
%   rx_signal - Received signal after channel and noise
%   noise_var - Noise variance

%% Remove Cyclic Prefix for channel processing
tx_no_cp = tx_signal(:, params.N_cp+1:end);

%% Apply FFT to convert to frequency domain
tx_freq = zeros(params.N_tx, params.N_fft);
for tx_idx = 1:params.N_tx
    tx_freq(tx_idx, :) = fft(tx_no_cp(tx_idx, :), params.N_fft);
end

%% Apply Channel in Frequency Domain
rx_freq = zeros(params.N_rx, params.N_fft);
for k = 1:params.N_fft
    H_k = squeeze(H_true(:, :, k)); % Channel matrix for subcarrier k
    tx_k = tx_freq(:, k); % Transmitted symbols for subcarrier k
    rx_freq(:, k) = H_k * tx_k; % Apply channel
end

%% Convert back to time domain
rx_time = zeros(params.N_rx, params.N_fft);
for rx_idx = 1:params.N_rx
    rx_time(rx_idx, :) = ifft(rx_freq(rx_idx, :), params.N_fft);
end

%% Add Cyclic Prefix back
rx_with_cp = zeros(params.N_rx, params.N_fft + params.N_cp);
for rx_idx = 1:params.N_rx
    cp = rx_time(rx_idx, end-params.N_cp+1:end);
    rx_with_cp(rx_idx, :) = [cp, rx_time(rx_idx, :)];
end

%% Calculate Signal Power
signal_power = mean(abs(rx_with_cp).^2, 'all');

%% Calculate Noise Variance from SNR
SNR_linear = 10^(SNR_dB / 10);
noise_var = signal_power / SNR_linear;

%% Add AWGN Noise
noise = sqrt(noise_var/2) * (randn(size(rx_with_cp)) + 1j * randn(size(rx_with_cp)));
rx_signal = rx_with_cp + noise;

end