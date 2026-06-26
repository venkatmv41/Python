function params = config_parameters(config_type)
%% Configuration Parameters for Channel Estimation OFDM MIMO System
%
% This function provides different parameter configurations for the 
% channel estimation simulation. Users can easily modify parameters
% here without changing the main simulation code.
%
% INPUT:
%   config_type - String specifying configuration type:
%                'default'  - Standard simulation parameters
%                'quick'    - Fast simulation for testing
%                'detailed' - High-resolution simulation
%                'custom'   - User-defined parameters
%
% OUTPUT:
%   params - Structure containing all simulation parameters
%
% USAGE:
%   params = config_parameters('default');
%   params = config_parameters('quick');

if nargin < 1
    config_type = 'default';
end

%% Base Parameters Structure
params = struct();

%% Configuration Selection
switch lower(config_type)
    case 'default'
        %% Standard Configuration
        % OFDM Parameters
        params.N_subcarriers = 64;           % Number of OFDM subcarriers
        params.N_cp = 16;                    % Cyclic prefix length
        params.pilot_spacing = 4;            % Pilot spacing (every 4th subcarrier)
        params.N_symbols = 100;              % Number of OFDM symbols
        
        % MIMO Parameters
        params.N_tx = 2;                     % Number of transmit antennas
        params.N_rx = 2;                     % Number of receive antennas
        
        % Channel Parameters
        params.channel_taps = 4;             % Number of multipath taps
        params.path_delays = [0 1e-6 2e-6 3e-6];     % Path delays (seconds)
        params.path_gains = [0 -3 -6 -9];            % Path gains (dB)
        params.max_doppler = 10;             % Maximum Doppler shift (Hz)
        
        % Simulation Parameters
        params.SNR_range = 0:5:30;          % SNR range in dB
        params.monte_carlo_runs = 1;         % Number of Monte Carlo runs
        
    case 'quick'
        %% Quick Test Configuration (for fast testing)
        % OFDM Parameters
        params.N_subcarriers = 16;           % Reduced subcarriers
        params.N_cp = 4;                     % Reduced CP
        params.pilot_spacing = 4;            % Pilot spacing
        params.N_symbols = 20;               % Reduced symbols
        
        % MIMO Parameters
        params.N_tx = 2;                     % TX antennas
        params.N_rx = 2;                     % RX antennas
        
        % Channel Parameters
        params.channel_taps = 2;             % Reduced taps
        params.path_delays = [0 1e-6];       % Simplified delays
        params.path_gains = [0 -3];          % Simplified gains
        params.max_doppler = 5;              % Reduced Doppler
        
        % Simulation Parameters
        params.SNR_range = [0, 10, 20];     % Limited SNR points
        params.monte_carlo_runs = 1;         % Single run
        
    case 'detailed'
        %% High-Resolution Configuration (for detailed analysis)
        % OFDM Parameters
        params.N_subcarriers = 128;          % More subcarriers
        params.N_cp = 32;                    % Longer CP
        params.pilot_spacing = 4;            % Standard pilot spacing
        params.N_symbols = 200;              % More symbols
        
        % MIMO Parameters
        params.N_tx = 4;                     % More TX antennas
        params.N_rx = 4;                     % More RX antennas
        
        % Channel Parameters
        params.channel_taps = 6;             % More taps
        params.path_delays = [0 0.5e-6 1e-6 1.5e-6 2e-6 2.5e-6];
        params.path_gains = [0 -2 -4 -6 -8 -10];
        params.max_doppler = 20;             % Higher Doppler
        
        % Simulation Parameters
        params.SNR_range = 0:2:30;          % Finer SNR resolution
        params.monte_carlo_runs = 5;         % Multiple runs for averaging
        
    case 'custom'
        %% Custom Configuration Template
        % Users can modify these parameters as needed
        
        % OFDM Parameters
        params.N_subcarriers = 64;           % Modify as needed
        params.N_cp = 16;                    % Modify as needed
        params.pilot_spacing = 4;            % 2, 4, 8 are typical values
        params.N_symbols = 100;              % Adjust based on memory
        
        % MIMO Parameters
        params.N_tx = 2;                     % 1, 2, 4, 8 typical
        params.N_rx = 2;                     % Should be >= N_tx
        
        % Channel Parameters
        params.channel_taps = 4;             % 1-10 typical
        params.path_delays = [0 1e-6 2e-6 3e-6];
        params.path_gains = [0 -3 -6 -9];
        params.max_doppler = 10;             % 0-100 Hz typical
        
        % Simulation Parameters
        params.SNR_range = 0:5:30;          % Adjust range and step
        params.monte_carlo_runs = 1;         % Increase for better statistics
        
        % Additional custom parameters can be added here
        % params.custom_parameter = value;
        
    otherwise
        error('Unknown configuration type: %s. Use ''default'', ''quick'', ''detailed'', or ''custom''.', config_type);
end

%% Common Parameters (applied to all configurations)

% Modulation Schemes
params.modulation_schemes = {'BPSK', 'QPSK', '16QAM', '32QAM'};

% Channel Estimation Methods
params.estimation_methods = {'LS', 'MMSE', 'LMS'};

% LMS Parameters
params.lms_step_size = 0.01;             % LMS adaptation step size
params.lms_forget_factor = 0.99;         % Forgetting factor for LMS

% MMSE Parameters
params.mmse_snr_estimate = 10;           % SNR estimate for MMSE (dB)

% Image Processing Parameters
params.image_size = [64, 64];            % Image dimensions
params.image_type = 'synthetic';         % 'synthetic' or 'file'
params.image_filename = 'cameraman.tif'; % Image file if using file

% Performance Calculation Parameters
params.ber_threshold = 1e-6;             % Minimum BER for calculations
params.ser_threshold = 1e-6;             % Minimum SER for calculations

% Plotting Parameters
params.plot_style = 'publication';       % 'publication' or 'presentation'
params.save_figures = false;             % Save figures to files
params.figure_format = 'png';            % 'png', 'pdf', 'eps'

%% Parameter Validation
params = validate_parameters(params);

%% Display Configuration
fprintf('\n=== Configuration: %s ===\n', upper(config_type));
fprintf('OFDM: %d subcarriers, %d symbols, CP=%d\n', ...
    params.N_subcarriers, params.N_symbols, params.N_cp);
fprintf('MIMO: %dx%d antennas\n', params.N_tx, params.N_rx);
fprintf('Channel: %d taps, Doppler=%.1f Hz\n', ...
    params.channel_taps, params.max_doppler);
fprintf('SNR Range: %.1f to %.1f dB (%d points)\n', ...
    min(params.SNR_range), max(params.SNR_range), length(params.SNR_range));
fprintf('Modulation: %s\n', strjoin(params.modulation_schemes, ', '));
fprintf('Estimation: %s\n', strjoin(params.estimation_methods, ', '));

end

function params = validate_parameters(params)
%% Validate and adjust parameters if necessary

% Ensure pilot spacing divides number of subcarriers
if mod(params.N_subcarriers, params.pilot_spacing) ~= 0
    warning('Pilot spacing does not divide evenly into subcarriers. Adjusting...');
    params.pilot_spacing = 4; % Default fallback
end

% Ensure reasonable number of data subcarriers
n_pilots = floor(params.N_subcarriers / params.pilot_spacing);
n_data = params.N_subcarriers - n_pilots;
if n_data < 1
    error('Too few data subcarriers. Reduce pilot_spacing or increase N_subcarriers.');
end

% Ensure cyclic prefix is reasonable
if params.N_cp >= params.N_subcarriers
    warning('Cyclic prefix too long. Setting to 25%% of symbol length.');
    params.N_cp = floor(params.N_subcarriers / 4);
end

% Ensure MIMO configuration is valid
if params.N_rx < params.N_tx
    warning('Number of RX antennas should be >= TX antennas for good performance.');
end

% Validate SNR range
if min(params.SNR_range) < -10 || max(params.SNR_range) > 50
    warning('SNR range may be unrealistic. Typical range is -5 to 35 dB.');
end

% Validate channel parameters
if length(params.path_delays) ~= length(params.path_gains)
    error('Path delays and gains must have same length.');
end

if length(params.path_delays) ~= params.channel_taps
    warning('Adjusting channel_taps to match path delays/gains.');
    params.channel_taps = length(params.path_delays);
end

fprintf('✓ Parameters validated successfully\n');

end

%% Example Usage Functions

function example_usage()
%% Examples of how to use different configurations

fprintf('\n=== Configuration Examples ===\n');

% Example 1: Quick test
fprintf('\n1. Quick test configuration:\n');
params_quick = config_parameters('quick');

% Example 2: Default simulation
fprintf('\n2. Default configuration:\n');
params_default = config_parameters('default');

% Example 3: High-resolution analysis
fprintf('\n3. Detailed configuration:\n');
params_detailed = config_parameters('detailed');

% Example 4: Custom parameters
fprintf('\n4. Custom configuration template:\n');
params_custom = config_parameters('custom');
% Modify params_custom as needed before using

fprintf('\nTo use in main simulation:\n');
fprintf('params = config_parameters(''default'');\n');
fprintf('Then pass params to your simulation functions.\n');

end