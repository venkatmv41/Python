%% Test Script to Verify Field Name Fix
% This script tests that the modulation type names work as valid MATLAB field names

clear all; close all; clc;

fprintf('Testing MATLAB field name compatibility...\n');

% Test modulation types
modulation_types = {'BPSK', 'QPSK', 'QAM16', 'QAM32', 'QAM64'};

% Test creating structure with these field names
results = struct();

for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    
    try
        % This should work now
        results.(mod_type) = struct();
        results.(mod_type).BER = rand(1, 5); % Test data
        results.(mod_type).SER = rand(1, 5); % Test data
        
        fprintf('✓ %s: Field name valid and structure created successfully\n', mod_type);
    catch ME
        fprintf('✗ %s: Error - %s\n', mod_type, ME.message);
    end
end

% Test accessing the fields
fprintf('\nTesting field access...\n');
for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    
    try
        ber_data = results.(mod_type).BER;
        fprintf('✓ %s: Field access successful, BER data length = %d\n', mod_type, length(ber_data));
    catch ME
        fprintf('✗ %s: Field access failed - %s\n', mod_type, ME.message);
    end
end

fprintf('\n=== Field Name Fix Test Completed ===\n');
fprintf('All modulation types should now work as valid MATLAB field names.\n');
fprintf('You can now run: main_mimo_ofdm_system\n');