%part one
Cameraman_image = imread('cameraman.tif');
Moon_image = imread('5.1.09.tiff');

Cameraman_image_fade = Cameraman_image * 0.2;
Moon_image_fade = Moon_image * 0.8;

Mixte_image = Cameraman_image_fade + Moon_image_fade;


%part two
N1 = 100;
First_part_image_1 = Cameraman_image(1:N1,1:N1);

N_size = size(Cameraman_image);
N2_1 = N_size(1) - 100;
N2_2 = N_size(2) - 100;
last_part_image_2 = Cameraman_image(N2_1+1:N_size(1),N2_2+1:N_size(2));

last_part_Mixte_image = last_part_image_2 * 0.8 + First_part_image_1 * 0.2;


%part three
Cameraman_image_reshape = reshape(Cameraman_image,[65536,1]);
Moon_image_reshape = reshape(Moon_image,[65536,1]);

Both_images = ones(65536,2);
Both_images(:,1) = Cameraman_image_reshape;
Both_images(:,2) = Moon_image_reshape;

Fade_vector = [0.5; 0.5];

mixing_image_vector = Both_images * Fade_vector;
mixing_image_reshape = reshape(mixing_image_vector, [256, 256]);
mixing_image_matrix = uint8(mixing_image_reshape);


%visualization
figure(1)

subplot(3,3,1); imshow(Cameraman_image); title('Cameraman');
subplot(3,3,2); imshow(Moon_image); title('Moon');
subplot(3,3,3); imshow(Mixte_image); title('Mixed');

subplot(3,3,4); imshow(First_part_image_1); title('First part image 1');
subplot(3,3,5); imshow(last_part_image_2); title('last part image 2');
subplot(3,3,6); imshow(last_part_Mixte_image); title('last part Mixte image');

subplot(3,3,8); imshow(mixing_image_matrix); title('Mixed Image by Matrix Multiplication');
