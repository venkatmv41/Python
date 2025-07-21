%% Clean Run Script - Channel Estimation OFDM MIMO
% This script clears all cached functions and variables before running

% Clear everything
clear all;
close all;
clc;

% Clear function cache
clear functions;

% Display current directory
fprintf('Current directory: %s\n', pwd);

% List MATLAB files to confirm we have the right files
fprintf('\nMATLAB files in current directory:\n');
files = dir('*.m');
for i = 1:length(files)
    fprintf('  %s\n', files(i).name);
end

% Run the main simulation
fprintf('\n=== Starting Clean Simulation ===\n');
try
    channel_estimation_ofdm_mimo;
    fprintf('\n=== Simulation Completed Successfully! ===\n');
catch ME
    fprintf('\n=== Error Occurred ===\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('File: %s\n', ME.stack(1).file);
    fprintf('Line: %d\n', ME.stack(1).line);
    
    % Display the problematic line if possible
    if exist(ME.stack(1).file, 'file')
        try
            lines = readlines(ME.stack(1).file);
            if length(lines) >= ME.stack(1).line
                fprintf('Problematic line: %s\n', lines(ME.stack(1).line));
            end
        catch
            % Ignore if can't read file
        end
    end
end