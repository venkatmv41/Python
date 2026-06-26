function reconstructed_image = reconstruct_image(received_bits, original_size)
%% Reconstruct Image from Received Bits
% Inputs:
%   received_bits - Received bits from demodulation
%   original_size - Original image size [height, width]
% Outputs:
%   reconstructed_image - Reconstructed image

% Calculate required number of bits for the image
total_pixels = prod(original_size);
bits_per_pixel = 8; % Assuming 8-bit grayscale
required_bits = total_pixels * bits_per_pixel;

% Ensure we have enough bits
if length(received_bits) < required_bits
    % Pad with zeros if not enough bits
    received_bits = [received_bits; zeros(required_bits - length(received_bits), 1)];
else
    % Truncate if too many bits
    received_bits = received_bits(1:required_bits);
end

% Reshape bits to pixel format
bit_matrix = reshape(received_bits, bits_per_pixel, total_pixels);

% Convert bits to pixel values
pixel_values = zeros(1, total_pixels);
for i = 1:total_pixels
    pixel_values(i) = bi2de(bit_matrix(:, i)', 'left-msb');
end

% Reshape to image format
reconstructed_image = uint8(reshape(pixel_values, original_size));

end