function create_test_image()
%% Create Test Image for MIMO-OFDM Simulation
% This function creates a test image with various patterns for testing

% Create a 64x64 test image with different patterns
test_image = zeros(64, 64);

% Add checkerboard pattern
test_image(1:32, 1:32) = checkerboard(4, 4);

% Add gradient pattern
[X, Y] = meshgrid(1:32, 1:32);
test_image(1:32, 33:64) = X / 32;

% Add circular pattern
[X, Y] = meshgrid(1:32, 1:32);
center = 16;
radius = sqrt((X - center).^2 + (Y - center).^2);
test_image(33:64, 1:32) = (radius <= 12) * 1;

% Add text-like pattern
test_image(33:64, 33:64) = create_text_pattern();

% Normalize to 0-255 range
test_image = uint8(255 * (test_image - min(test_image(:))) / (max(test_image(:)) - min(test_image(:))));

% Save the test image
imwrite(test_image, 'test_image.png');

fprintf('Test image created and saved as test_image.png\n');

end

function text_pattern = create_text_pattern()
    % Create a simple text-like pattern
    text_pattern = zeros(32, 32);
    
    % Create letter-like patterns
    % Letter 'M'
    text_pattern(5:25, 3:5) = 1;
    text_pattern(5:25, 15:17) = 1;
    text_pattern(5:10, 6:14) = 1;
    
    % Letter 'I'
    text_pattern(5:25, 20:22) = 1;
    
    % Letter 'M'
    text_pattern(5:25, 25:27) = 1;
    text_pattern(5:10, 28:30) = 1;
    
    % Letter 'O'
    text_pattern(10:20, 5:7) = 1;
    text_pattern(10:20, 13:15) = 1;
    text_pattern(10:12, 8:12) = 1;
    text_pattern(18:20, 8:12) = 1;
end