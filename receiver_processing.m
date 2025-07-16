function [rx_bits, rx_symbols, H_estimated] = receiver_processing(rx_signal, pilot_symbols, data_symbols, H_true, mod_type, params, estimation_method, noise_var)
%% Receiver Processing for MIMO-OFDM System
% Inputs:
%   rx_signal - Received signal
%   pilot_symbols - Known pilot symbols
%   data_symbols - Original data symbols (for comparison)
%   H_true - True channel matrix
%   mod_type - Modulation type
%   params - System parameters
%   estimation_method - 'perfect', 'LS', 'MMSE', 'LMSE', 'none'
%   noise_var - Noise variance (optional, for MMSE/LMSE)
% Outputs:
%   rx_bits - Received bits
%   rx_symbols - Received symbols
%   H_estimated - Estimated channel matrix

if nargin < 8
    noise_var = 0.01; % Default noise variance
end

%% Remove Cyclic Prefix
rx_no_cp = rx_signal(:, params.N_cp+1:end);

%% FFT Processing
rx_freq = zeros(params.N_rx, params.N_fft);
for rx_idx = 1:params.N_rx
    rx_freq(rx_idx, :) = fft(rx_no_cp(rx_idx, :), params.N_fft);
end

%% Extract Pilot and Data Subcarriers
pilot_indices = 1:params.pilot_spacing:params.N_fft;
data_indices = setdiff(1:params.N_fft, pilot_indices);

% Ensure we don't exceed available indices
pilot_indices = pilot_indices(1:min(length(pilot_indices), params.N_pilot));
available_data_indices = data_indices(1:min(length(data_indices), size(data_symbols, 2)));

%% Channel Estimation
switch estimation_method
    case 'perfect'
        H_estimated = H_true;
        
    case 'LS'
        H_estimated = ls_channel_estimation(rx_freq, pilot_symbols, pilot_indices, params);
        
    case 'MMSE'
        H_LS = ls_channel_estimation(rx_freq, pilot_symbols, pilot_indices, params);
        H_estimated = mmse_channel_estimation(H_LS, pilot_symbols, pilot_indices, noise_var, params);
        
    case 'LMSE'
        H_LS = ls_channel_estimation(rx_freq, pilot_symbols, pilot_indices, params);
        H_estimated = lmse_channel_estimation(H_LS, pilot_symbols, pilot_indices, noise_var, params);
        
    case 'none'
        H_estimated = ones(size(H_true)); % No channel estimation
        
    otherwise
        error('Unknown estimation method');
end

%% Channel Equalization
rx_equalized = zeros(params.N_tx, length(available_data_indices));
for k_idx = 1:length(available_data_indices)
    k = available_data_indices(k_idx);
    
    % Get channel matrix for this subcarrier
    H_k = squeeze(H_estimated(:, :, k));
    
    % Get received symbols for this subcarrier
    y_k = rx_freq(:, k);
    
    % Zero-forcing equalization (can be extended to MMSE equalization)
    if cond(H_k) < 1e12 % Check for well-conditioned matrix
        x_hat_k = pinv(H_k) * y_k; % Pseudo-inverse for equalization
    else
        x_hat_k = H_k \ y_k; % Matrix division
    end
    
    rx_equalized(:, k_idx) = x_hat_k;
end

%% Demodulation
rx_symbols = zeros(params.N_tx, size(rx_equalized, 2));
rx_bits_cell = cell(params.N_tx, 1);

for tx_idx = 1:params.N_tx
    [rx_symbols(tx_idx, :), rx_bits_cell{tx_idx}] = demodulate_symbols(rx_equalized(tx_idx, :), mod_type);
end

%% Parallel to Serial Conversion
rx_bits = [];
for tx_idx = 1:params.N_tx
    rx_bits = [rx_bits; rx_bits_cell{tx_idx}];
end

% Reshape to match input format
bits_per_symbol = log2(get_modulation_order(mod_type));
total_bits = length(rx_bits);
rx_bits = rx_bits(1:min(total_bits, params.N_data * bits_per_symbol));

end

%% Channel Estimation Functions

function H_LS = ls_channel_estimation(rx_freq, pilot_symbols, pilot_indices, params)
    % Least Squares Channel Estimation
    % H_LS = Y / X where Y is received pilots and X is transmitted pilots
    
    H_LS = zeros(params.N_rx, params.N_tx, params.N_fft);
    
    % Estimate channel at pilot locations
    for p_idx = 1:length(pilot_indices)
        k = pilot_indices(p_idx);
        
        % Get received and transmitted pilot symbols
        Y_k = rx_freq(:, k); % Received pilots
        X_k = pilot_symbols(:, p_idx); % Transmitted pilots
        
        % LS estimation: H = Y / X
        if length(X_k) == params.N_tx && all(X_k ~= 0)
            H_LS(:, :, k) = Y_k * X_k' / (X_k * X_k');
        end
    end
    
    % Interpolate for data subcarriers
    H_LS = interpolate_channel(H_LS, pilot_indices, params);
end

function H_MMSE = mmse_channel_estimation(H_LS, pilot_symbols, pilot_indices, noise_var, params)
    % MMSE Channel Estimation
    % H_MMSE = R_HH * (R_HH + σ²(X^H*X)^(-1))^(-1) * H_LS
    
    H_MMSE = zeros(size(H_LS));
    
    % Calculate autocorrelation matrix (simplified)
    R_HH = calculate_channel_autocorrelation(params);
    
    for p_idx = 1:length(pilot_indices)
        k = pilot_indices(p_idx);
        
        % Get pilot symbols for this subcarrier
        X_k = pilot_symbols(:, p_idx);
        
        if all(X_k ~= 0)
            % Calculate X^H * X
            XHX = X_k' * X_k;
            
            % MMSE filter
            if det(XHX) > 1e-10
                MMSE_filter = R_HH / (R_HH + noise_var * inv(XHX));
                H_MMSE(:, :, k) = MMSE_filter * squeeze(H_LS(:, :, k));
            else
                H_MMSE(:, :, k) = H_LS(:, :, k);
            end
        end
    end
    
    % Interpolate for data subcarriers
    H_MMSE = interpolate_channel(H_MMSE, pilot_indices, params);
end

function H_LMSE = lmse_channel_estimation(H_LS, pilot_symbols, pilot_indices, noise_var, params)
    % Linear MMSE Channel Estimation (simplified version)
    % Similar to MMSE but with linear approximation
    
    H_LMSE = zeros(size(H_LS));
    
    % Simplified LMSE: weighted average of LS estimates
    alpha = 0.8; % Weighting factor (can be optimized)
    
    for k = 1:params.N_fft
        if ismember(k, pilot_indices)
            % For pilot subcarriers, use weighted LS estimate
            H_LMSE(:, :, k) = alpha * H_LS(:, :, k);
        else
            % For data subcarriers, interpolate
            H_LMSE(:, :, k) = H_LS(:, :, k);
        end
    end
    
    % Apply smoothing filter
    H_LMSE = apply_smoothing_filter(H_LMSE, params);
end

function R_HH = calculate_channel_autocorrelation(params)
    % Calculate channel autocorrelation matrix (simplified)
    % In practice, this would be estimated from channel statistics
    
    % Assume exponential correlation model
    correlation_factor = 0.9;
    R_HH = correlation_factor * eye(params.N_tx);
end

function H_interp = interpolate_channel(H_pilot, pilot_indices, params)
    % Interpolate channel estimates from pilot to data subcarriers
    
    H_interp = H_pilot;
    
    for rx_idx = 1:params.N_rx
        for tx_idx = 1:params.N_tx
            % Extract pilot channel estimates
            pilot_values = squeeze(H_pilot(rx_idx, tx_idx, pilot_indices));
            
            % Interpolate to all subcarriers
            if length(pilot_values) > 1
                H_interp_vec = interp1(pilot_indices, pilot_values, 1:params.N_fft, 'linear', 'extrap');
                H_interp(rx_idx, tx_idx, :) = H_interp_vec;
            else
                % If only one pilot, use constant interpolation
                H_interp(rx_idx, tx_idx, :) = pilot_values(1);
            end
        end
    end
end

function H_smooth = apply_smoothing_filter(H, params)
    % Apply smoothing filter to channel estimates
    
    H_smooth = H;
    filter_length = 3; % Simple moving average filter
    
    for rx_idx = 1:params.N_rx
        for tx_idx = 1:params.N_tx
            h_vec = squeeze(H(rx_idx, tx_idx, :));
            h_smooth = conv(h_vec, ones(1, filter_length)/filter_length, 'same');
            H_smooth(rx_idx, tx_idx, :) = h_smooth;
        end
    end
end

%% Demodulation Functions

function mod_order = get_modulation_order(mod_type)
    switch mod_type
        case 'BPSK'
            mod_order = 2;
        case 'QPSK'
            mod_order = 4;
        case 'QAM16'
            mod_order = 16;
        case 'QAM32'
            mod_order = 32;
        case 'QAM64'
            mod_order = 64;
        otherwise
            error('Unsupported modulation type');
    end
end

function [symbols, bits] = demodulate_symbols(rx_symbols, mod_type)
    % Demodulate received symbols to bits
    
    symbols = zeros(size(rx_symbols));
    bits_per_symbol = log2(get_modulation_order(mod_type));
    bits = zeros(bits_per_symbol * length(rx_symbols), 1);
    
    for i = 1:length(rx_symbols)
        symbol = rx_symbols(i);
        
        switch mod_type
            case 'BPSK'
                % Hard decision
                if real(symbol) >= 0
                    symbols(i) = 1;
                    bit_val = 1;
                else
                    symbols(i) = -1;
                    bit_val = 0;
                end
                bits(i) = bit_val;
                
            case 'QPSK'
                constellation = [1+1j, -1+1j, -1-1j, 1-1j] / sqrt(2);
                [~, idx] = min(abs(symbol - constellation));
                symbols(i) = constellation(idx);
                bit_pattern = de2bi(idx-1, 2, 'left-msb');
                bits((i-1)*2+1:i*2) = bit_pattern';
                
            case {'QAM16', 'QAM32', 'QAM64'}
                constellation = get_constellation(mod_type);
                [~, idx] = min(abs(symbol - constellation));
                symbols(i) = constellation(idx);
                bit_pattern = de2bi(idx-1, bits_per_symbol, 'left-msb');
                bits((i-1)*bits_per_symbol+1:i*bits_per_symbol) = bit_pattern';
        end
    end
end

function constellation = get_constellation(mod_type)
    switch mod_type
        case 'QAM16'
            real_part = [-3, -1, 1, 3];
            imag_part = [-3, -1, 1, 3];
            [R, I] = meshgrid(real_part, imag_part);
            constellation = (R(:) + 1j * I(:))' / sqrt(10);
            
        case 'QAM32'
            constellation = zeros(1, 32);
            idx = 1;
            for i = -3:2:3
                for j = -5:2:5
                    if abs(i) + abs(j) <= 5
                        constellation(idx) = (i + 1j * j) / sqrt(20);
                        idx = idx + 1;
                    end
                end
            end
            constellation = constellation(1:32);
            
        case 'QAM64'
            real_part = [-7, -5, -3, -1, 1, 3, 5, 7];
            imag_part = [-7, -5, -3, -1, 1, 3, 5, 7];
            [R, I] = meshgrid(real_part, imag_part);
            constellation = (R(:) + 1j * I(:))' / sqrt(42);
    end
end