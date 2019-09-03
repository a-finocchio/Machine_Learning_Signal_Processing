function [train_reshape,test_reshape] = loadImagesLab8()
%% load train data
train_directory = 'orl_faces/Train/s%d/%d.pgm';
train_data = zeros(112,92,360);
for i = 1:40
    for j = 1:9
        image_name = sprintf(train_directory, i, j);
        train_data(:,:,(i-1)*9+j) = imread(image_name);
       
%         disp(image_name);
    end
end

image_size = 112*92;
% train_reshape = reshape(train_data, [40,9,image_size]);
train_reshape = reshape(train_data, [image_size,360]);
% train_reshape = train_reshape';




%% load test data
test_directory = 'orl_faces/Test/s%d/%d.pgm';
test_data = zeros(112,92,40);
for i = 1:40
        image_name = sprintf(test_directory, i, 10);
        test_data(:,:,i) = imread(image_name);
end

test_reshape = reshape(test_data, [image_size,40]);
% test_reshape = test_reshape';
end