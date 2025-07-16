function [tx_signal, pilot_symbols, data_symbols, H_true] = transmitter_processing(data_bits, mod_type, params)
%% Transmitter Processing for MIMO-OFDM System
% Inputs:
%   data_bits - Input data bits
%   mod_type - Modulation type ('BPSK', 'QPSK', '16QAM', '32QAM', '64QAM')
%   params - System parameters structure
% Outputs:
%   tx_signal - Transmitted signal after IFFT and CP addition
%   pilot_symbols - Pilot symbols used
%   data_symbols - Data symbols after modulation
%   H_true - True channel matrix

%% Serial to Parallel Conversion
bits_per_symbol = log2(get_modulation_order(mod_type));
num_data_symbols = params.N_data;

% Reshape data bits for parallel processing
if length(data_bits) < num_data_symbols * bits_per_symbol
    data_bits = [data_bits; zeros(num_data_symbols * bits_per_symbol - length(data_bits), 1)];
end

data_matrix = reshape(data_bits(1:num_data_symbols * bits_per_symbol), bits_per_symbol, num_data_symbols);

%% Constellation Mapping
data_symbols = zeros(params.N_tx, num_data_symbols);
for tx_idx = 1:params.N_tx
    % Alternate data between antennas
    tx_data = data_matrix(:, tx_idx:params.N_tx:end);
    if size(tx_data, 2) < num_data_symbols / params.N_tx
        tx_data = [tx_data, zeros(bits_per_symbol, num_data_symbols / params.N_tx - size(tx_data, 2))];
    end
    data_symbols(tx_idx, :) = modulate_symbols(tx_data, mod_type);
end

%% Generate Pilot Symbols
pilot_symbols = generate_pilot_symbols(params);

%% Subcarrier Allocation
% Allocate pilots and data to subcarriers
ofdm_symbols = zeros(params.N_tx, params.N_fft);
pilot_indices = 1:params.pilot_spacing:params.N_fft;
data_indices = setdiff(1:params.N_fft, pilot_indices);

% Ensure we don't exceed available data
available_data_indices = data_indices(1:min(length(data_indices), size(data_symbols, 2)));

for tx_idx = 1:params.N_tx
    ofdm_symbols(tx_idx, pilot_indices) = pilot_symbols(tx_idx, :);
    ofdm_symbols(tx_idx, available_data_indices) = data_symbols(tx_idx, 1:length(available_data_indices));
end

%% IFFT Processing
ifft_output = zeros(params.N_tx, params.N_fft);
for tx_idx = 1:params.N_tx
    ifft_output(tx_idx, :) = ifft(ofdm_symbols(tx_idx, :), params.N_fft);
end

%% Add Cyclic Prefix
tx_signal = zeros(params.N_tx, params.N_fft + params.N_cp);
for tx_idx = 1:params.N_tx
    cp = ifft_output(tx_idx, end-params.N_cp+1:end);
    tx_signal(tx_idx, :) = [cp, ifft_output(tx_idx, :)];
end

%% Generate True Channel Matrix
H_true = generate_channel_matrix(params);

end

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

function symbols = modulate_symbols(data_matrix, mod_type)
    [bits_per_symbol, num_symbols] = size(data_matrix);
    symbols = zeros(1, num_symbols);
    
    for i = 1:num_symbols
        bits = data_matrix(:, i);
        decimal_val = bi2de(bits', 'left-msb');
        
        switch mod_type
            case 'BPSK'
                symbols(i) = 2 * decimal_val - 1; % Maps 0->-1, 1->1
                
            case 'QPSK'
                constellation = [1+1j, -1+1j, -1-1j, 1-1j] / sqrt(2);
                symbols(i) = constellation(decimal_val + 1);
                
            case '16QAM'
                constellation = qam_constellation(16);
                symbols(i) = constellation(decimal_val + 1);
                
            case '32QAM'
                constellation = qam_constellation(32);
                symbols(i) = constellation(decimal_val + 1);
                
            case '64QAM'
                constellation = qam_constellation(64);
                symbols(i) = constellation(decimal_val + 1);
        end
    end
end

function constellation = qam_constellation(M)
    % Generate QAM constellation points
    switch M
        case 16
            % 16-QAM constellation
            real_part = [-3, -1, 1, 3];
            imag_part = [-3, -1, 1, 3];
            [R, I] = meshgrid(real_part, imag_part);
            constellation = (R(:) + 1j * I(:))' / sqrt(10);
            
        case 32
            % 32-QAM constellation (cross constellation)
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
            
        case 64
            % 64-QAM constellation
            real_part = [-7, -5, -3, -1, 1, 3, 5, 7];
            imag_part = [-7, -5, -3, -1, 1, 3, 5, 7];
            [R, I] = meshgrid(real_part, imag_part);
            constellation = (R(:) + 1j * I(:))' / sqrt(42);
    end
end

function pilot_symbols = generate_pilot_symbols(params)
    % Generate known pilot symbols for channel estimation
    pilot_symbols = zeros(params.N_tx, params.N_pilot);
    
    % Use BPSK pilots for simplicity
    for tx_idx = 1:params.N_tx
        pilot_bits = randi([0, 1], 1, params.N_pilot);
        pilot_symbols(tx_idx, :) = 2 * pilot_bits - 1; % BPSK mapping
    end
end

function H = generate_channel_matrix(params)
    % Generate Rayleigh fading channel matrix
    % Each element represents channel gain from tx_i to rx_j
    H = zeros(params.N_rx, params.N_tx, params.N_fft);
    
    % Generate frequency-selective channel
    for rx_idx = 1:params.N_rx
        for tx_idx = 1:params.N_tx
            % Generate channel impulse response (multipath)
            num_paths = 4; % Number of multipath components
            path_delays = [0, 1, 2, 3]; % Path delays in samples
            path_gains = [1, 0.5, 0.3, 0.1]; % Path gains
            
            % Generate complex channel coefficients
            h_time = zeros(1, max(path_delays) + 1);
            for path_idx = 1:num_paths
                delay = path_delays(path_idx) + 1;
                gain = path_gains(path_idx);
                % Rayleigh fading coefficient
                h_time(delay) = gain * (randn + 1j * randn) / sqrt(2);
            end
            
            % Convert to frequency domain
            H_freq = fft(h_time, params.N_fft);
            H(rx_idx, tx_idx, :) = H_freq;
        end
    end
end