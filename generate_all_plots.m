function generate_all_plots(results, params, test_image, modulation_types)
%% Generate All Plots for MIMO-OFDM System Analysis
% Inputs:
%   results - Results structure from simulation
%   params - System parameters
%   test_image - Original test image
%   modulation_types - Cell array of modulation types

%% Set up plot parameters
colors = {'b-o', 'r-s', 'g-^', 'm-d', 'c-v'};
estimation_methods = {'Perfect', 'LS', 'MMSE', 'LMSE', 'No Est'};
estimation_colors = {'k-', 'b--', 'r:', 'g-.', 'm:'};

%% 1. BER vs SNR Plot for All Modulations
figure('Position', [100, 100, 1200, 800]);

for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    
    subplot(2, 3, i);
    semilogy(params.SNR_dB, results.(mod_type).BER_perfect, estimation_colors{1}, 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    semilogy(params.SNR_dB, results.(mod_type).BER_LS, estimation_colors{2}, 'LineWidth', 2, 'MarkerSize', 8);
    semilogy(params.SNR_dB, results.(mod_type).BER_MMSE, estimation_colors{3}, 'LineWidth', 2, 'MarkerSize', 8);
    semilogy(params.SNR_dB, results.(mod_type).BER_LMSE, estimation_colors{4}, 'LineWidth', 2, 'MarkerSize', 8);
    semilogy(params.SNR_dB, results.(mod_type).BER_no_est, estimation_colors{5}, 'LineWidth', 2, 'MarkerSize', 8);
    
    grid on;
    xlabel('SNR (dB)');
    ylabel('Bit Error Rate (BER)');
    title([mod_type ' - BER vs SNR']);
    legend(estimation_methods, 'Location', 'best');
    ylim([1e-4, 1]);
end

% Overall comparison
subplot(2, 3, 6);
for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    semilogy(params.SNR_dB, results.(mod_type).BER_MMSE, colors{i}, 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
end
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER Comparison - MMSE Estimation');
legend(modulation_types, 'Location', 'best');
ylim([1e-4, 1]);

sgtitle('BER vs SNR Analysis for MIMO-OFDM System');
saveas(gcf, 'BER_vs_SNR_Analysis.png');

%% 2. SER vs SNR Plot for All Modulations
figure('Position', [150, 150, 1200, 800]);

for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    
    subplot(2, 3, i);
    semilogy(params.SNR_dB, results.(mod_type).SER_perfect, estimation_colors{1}, 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    semilogy(params.SNR_dB, results.(mod_type).SER_LS, estimation_colors{2}, 'LineWidth', 2, 'MarkerSize', 8);
    semilogy(params.SNR_dB, results.(mod_type).SER_MMSE, estimation_colors{3}, 'LineWidth', 2, 'MarkerSize', 8);
    semilogy(params.SNR_dB, results.(mod_type).SER_LMSE, estimation_colors{4}, 'LineWidth', 2, 'MarkerSize', 8);
    semilogy(params.SNR_dB, results.(mod_type).SER_no_est, estimation_colors{5}, 'LineWidth', 2, 'MarkerSize', 8);
    
    grid on;
    xlabel('SNR (dB)');
    ylabel('Symbol Error Rate (SER)');
    title([mod_type ' - SER vs SNR']);
    legend(estimation_methods, 'Location', 'best');
    ylim([1e-4, 1]);
end

% Overall comparison
subplot(2, 3, 6);
for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    semilogy(params.SNR_dB, results.(mod_type).SER_MMSE, colors{i}, 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
end
grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER Comparison - MMSE Estimation');
legend(modulation_types, 'Location', 'best');
ylim([1e-4, 1]);

sgtitle('SER vs SNR Analysis for MIMO-OFDM System');
saveas(gcf, 'SER_vs_SNR_Analysis.png');

%% 3. MSE vs SNR Plot for Channel Estimation
figure('Position', [200, 200, 1200, 800]);

for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    
    subplot(2, 3, i);
    semilogy(params.SNR_dB, results.(mod_type).MSE_LS, 'b--', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    semilogy(params.SNR_dB, results.(mod_type).MSE_MMSE, 'r:', 'LineWidth', 2, 'MarkerSize', 8);
    semilogy(params.SNR_dB, results.(mod_type).MSE_LMSE, 'g-.', 'LineWidth', 2, 'MarkerSize', 8);
    
    grid on;
    xlabel('SNR (dB)');
    ylabel('Mean Square Error (MSE)');
    title([mod_type ' - Channel Estimation MSE']);
    legend({'LS', 'MMSE', 'LMSE'}, 'Location', 'best');
end

% Overall comparison
subplot(2, 3, 6);
for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    semilogy(params.SNR_dB, results.(mod_type).MSE_MMSE, colors{i}, 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
end
grid on;
xlabel('SNR (dB)');
ylabel('Mean Square Error (MSE)');
title('MSE Comparison - MMSE Estimation');
legend(modulation_types, 'Location', 'best');

sgtitle('Channel Estimation MSE vs SNR Analysis');
saveas(gcf, 'MSE_vs_SNR_Analysis.png');

%% 4. Constellation Diagrams
figure('Position', [250, 250, 1200, 800]);

for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    
    subplot(2, 3, i);
    constellation = get_constellation_points(mod_type);
    
    % Add noise to show received constellation
    snr_for_display = 20; % dB
    noise_var = 10^(-snr_for_display/10);
    noisy_constellation = constellation + sqrt(noise_var/2) * (randn(size(constellation)) + 1j * randn(size(constellation)));
    
    scatter(real(constellation), imag(constellation), 100, 'bo', 'filled');
    hold on;
    scatter(real(noisy_constellation), imag(noisy_constellation), 50, 'r+', 'LineWidth', 2);
    
    grid on;
    xlabel('In-phase');
    ylabel('Quadrature');
    title([mod_type ' Constellation']);
    legend('Ideal', 'Received (20dB)', 'Location', 'best');
    axis equal;
end

sgtitle('Constellation Diagrams');
saveas(gcf, 'Constellation_Diagrams.png');

%% 5. Image Transmission Results
methods = {'Perfect', 'LS', 'MMSE', 'LMSE'};
snr_indices_to_show = [1, 3, 5]; % Show results for different SNR values

for mod_idx = 1:length(modulation_types)
    mod_type = modulation_types{mod_idx};
    
    figure('Position', [300 + mod_idx*50, 300, 1400, 1000]);
    
    subplot_idx = 1;
    
    % Original image
    subplot(length(snr_indices_to_show) + 1, length(methods) + 1, 1);
    imshow(test_image);
    title('Original Image');
    
    for snr_idx = snr_indices_to_show
        SNR_val = params.SNR_dB(snr_idx);
        
        % SNR label
        subplot(length(snr_indices_to_show) + 1, length(methods) + 1, (snr_idx - snr_indices_to_show(1) + 1) * (length(methods) + 1) + 1);
        text(0.5, 0.5, sprintf('SNR = %d dB', SNR_val), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        axis off;
        
        % Show received images for each method
        for method_idx = 1:length(methods)
            subplot(length(snr_indices_to_show) + 1, length(methods) + 1, (snr_idx - snr_indices_to_show(1) + 1) * (length(methods) + 1) + method_idx + 1);
            
            received_img = results.(mod_type).received_images{snr_idx, method_idx};
            imshow(received_img);
            
            if snr_idx == snr_indices_to_show(1)
                title(methods{method_idx});
            end
            
            % Calculate PSNR
            psnr_val = calculate_psnr(test_image, received_img);
            xlabel(sprintf('PSNR: %.2f dB', psnr_val));
        end
    end
    
    sgtitle([mod_type ' - Image Transmission Results']);
    saveas(gcf, [mod_type '_Image_Transmission.png']);
end

%% 6. Performance Summary Table
figure('Position', [400, 400, 1200, 600]);

% Create performance summary for SNR = 20 dB
snr_20_idx = find(params.SNR_dB == 20);
if isempty(snr_20_idx)
    snr_20_idx = 3; % Use middle SNR value
end

summary_data = zeros(length(modulation_types), 12); % 4 methods x 3 metrics
row_labels = modulation_types;
col_labels = {'BER-Perfect', 'BER-LS', 'BER-MMSE', 'BER-LMSE', ...
              'SER-Perfect', 'SER-LS', 'SER-MMSE', 'SER-LMSE', ...
              'MSE-LS', 'MSE-MMSE', 'MSE-LMSE', 'Spectral Eff'};

for i = 1:length(modulation_types)
    mod_type = modulation_types{i};
    
    summary_data(i, 1) = results.(mod_type).BER_perfect(snr_20_idx);
    summary_data(i, 2) = results.(mod_type).BER_LS(snr_20_idx);
    summary_data(i, 3) = results.(mod_type).BER_MMSE(snr_20_idx);
    summary_data(i, 4) = results.(mod_type).BER_LMSE(snr_20_idx);
    
    summary_data(i, 5) = results.(mod_type).SER_perfect(snr_20_idx);
    summary_data(i, 6) = results.(mod_type).SER_LS(snr_20_idx);
    summary_data(i, 7) = results.(mod_type).SER_MMSE(snr_20_idx);
    summary_data(i, 8) = results.(mod_type).SER_LMSE(snr_20_idx);
    
    summary_data(i, 9) = results.(mod_type).MSE_LS(snr_20_idx);
    summary_data(i, 10) = results.(mod_type).MSE_MMSE(snr_20_idx);
    summary_data(i, 11) = results.(mod_type).MSE_LMSE(snr_20_idx);
    
    % Calculate spectral efficiency (bits/s/Hz)
    bits_per_symbol = log2(get_modulation_order(mod_type));
    summary_data(i, 12) = bits_per_symbol * (1 - results.(mod_type).BER_MMSE(snr_20_idx));
end

% Create table
uitable('Data', summary_data, 'ColumnName', col_labels, 'RowName', row_labels, ...
        'Position', [50, 50, 1100, 500], 'ColumnWidth', {80});

sgtitle('Performance Summary at SNR = 20 dB');
saveas(gcf, 'Performance_Summary_Table.png');

fprintf('All plots generated and saved successfully!\n');

end

%% Helper Functions

function constellation = get_constellation_points(mod_type)
    switch mod_type
        case 'BPSK'
            constellation = [-1, 1];
            
        case 'QPSK'
            constellation = [1+1j, -1+1j, -1-1j, 1-1j] / sqrt(2);
            
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