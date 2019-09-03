%% Driver File for Problem 1: Part 3: Gender Detection
% You will build a gender detection system using the PCA dimensions 
% from images.
% Author Name : Huixiang Li
clear
close all
clc


%% load male train data
directory_train_male  = '../data/lfw_genders/male/train';
file_train_male = fullfile(directory_train_male, '*.jpg');
img_dir_train_male = dir(file_train_male);
%out put image amount
n_img_train_male = length(img_dir_train_male);
image_train_male = zeros(n_img_train_male,250,250);

for i = 1:n_img_train_male
  img_file_name_train_male = img_dir_train_male(i).name;
  img_name_train_male = fullfile(directory_train_male, img_file_name_train_male);
  
  %store img into matrix 
  image_train_male(i,:,:) = double(imread(img_name_train_male));
  
  %display each image for double-check
  %imagesc(imread(img_name_train_male)), axis image, colormap gray, title('show training data');  
  %drawnow; 
  
end


%% load female train data
directory_train_female  = '../data/lfw_genders/female/train';
file_train_female = fullfile(directory_train_female, '*.jpg');
img_dir_train_female = dir(file_train_female);
%out put image amount
n_img_train_female = length(img_dir_train_female);
image_train_female = zeros(n_img_train_female,250,250);

for i = 1:n_img_train_female
  img_file_name_train_female = img_dir_train_female(i).name;
  img_name_train_female = fullfile(directory_train_female, img_file_name_train_female);
  
  %store img into matrix 
  image_train_female(i,:,:) = double(imread(img_name_train_female));
  
  %display each image for double-check
%   imagesc(imread(img_name_train_female)), axis image, colormap gray, title('show training data');  
%   drawnow; 
  
end


%% load male test data
directory_test_male  = '../data/lfw_genders/male/test';
file_test_male = fullfile(directory_test_male, '*.jpg');
img_dir_test_male = dir(file_test_male);
%out put image amount
n_img_test_male = length(img_dir_test_male);
image_test_male = zeros(n_img_test_male,250,250);

for i = 1:n_img_test_male
  img_file_name_test_male = img_dir_test_male(i).name;
  img_name_test_male = fullfile(directory_test_male, img_file_name_test_male);
  
  %store img into matrix 
  image_test_male(i,:,:) = double(imread(img_name_test_male));
  
  %display each image for double-check
%   imagesc(imread(img_name_test_male)), axis image, colormap gray, title('show training data');  
%   drawnow; 
  
end


%% load female test data
directory_test_female  = '../data/lfw_genders/female/test';
file_test_female = fullfile(directory_test_female, '*.jpg');
img_dir_test_female = dir(file_test_female);
%out put image amount
n_img_test_female = length(img_dir_test_female);
image_test_female = zeros(n_img_test_female,250,250);

for i = 1:n_img_test_female
  img_file_name_test_female = img_dir_test_female(i).name;
  img_name_test_female = fullfile(directory_test_female, img_file_name_test_female);
  
  %store img into matrix 
  image_test_female(i,:,:) = double(imread(img_name_test_female));
  
  %display each image for double-check
%   imagesc(imread(img_name_test_female)), axis image, colormap gray, title('show training data');  
%   drawnow; 
  
end


%% reshape images
image_size_GD = 250*250;
reshape_train_male = reshape(image_train_male, [n_img_train_male, image_size_GD]);
reshape_train_female = reshape(image_train_female, [n_img_train_female, image_size_GD]);
reshape_test_male = reshape(image_test_male, [n_img_test_male, image_size_GD]);
reshape_test_female = reshape(image_test_female, [n_img_test_female, image_size_GD]);


%% normalize & center the data
mean_male = mean(reshape_train_male, 1);
mean_female = mean(reshape_train_female, 1);
mean_all = (mean_male + mean_female)/2;

mean_male_repmat = repmat(mean_male, size(reshape_train_male,1),1);
mean_female_repmat = repmat(mean_female, size(reshape_train_female,1),1);

center_male_train = reshape_train_male - mean_all; %- mean_male_repmat;
center_female_train = reshape_train_female - mean_all; %- mean_female_repmat;
center_male_test = reshape_test_male - mean_all; %- mean_male;
center_female_test = reshape_test_female - mean_all; %- mean_female;

center_train = [center_male_train; center_female_train];
center_test = [center_male_test; center_female_test];


%% plot mean faces
figure(2);
subplot(1,3,1); imagesc(reshape(mean_male, [250,250])), title('mean male face'), colormap gray, axis image;
subplot(1,3,2); imagesc(reshape(mean_female, [250,250])), title('mean female face'), colormap gray, axis image;


%% Eigen vectors & values
corr_matrix_total_train = center_train * center_train';
[e_vector_train_total, e_values_train_total] = eig(corr_matrix_total_train);
e_vector_train_total = real([center_male_train; center_female_train]' * e_vector_train_total);
e_values_train_total = abs(e_values_train_total);


%% plot eigen values
figure(3)
plot(e_values_train_total), title('e-values');


%% project mean male & female face ã€question6ã€?

K_value = [50,100,200,300];

error_test_male = [];
error_test_female = [];
accuracy_test_male = [];
accuracy_test_female = [];

time=[];

for i=1:4
    
    K = K_value(i);
    
    tic;
    
    B_train = e_vector_train_total(:,end-K+1:end);

    W_train_male = pinv(B_train) * double(mean_male');
    W_train_female = pinv(B_train) * double(mean_female');

    mean_male_approx = B_train * W_train_male;
    mean_female_approx = B_train * W_train_female;


%% plot projection
    figure(4)
    subplot(1,4,i)
    imagesc(reshape(mean_male_approx, [250,250])), title('Male projection'), colormap gray, axis image;
    figure(5)
    subplot(1,4,i)
    imagesc(reshape(mean_female_approx, [250,250])), title('Female projection'), colormap gray, axis image;


%% project all test data
    W_test = pinv(B_train) * double(center_test');
    test_approx = B_train * W_test;
    test_approx = test_approx';


%% plot project test data
% check_image = 1211;
% 
% figure(6)
% subplot(1,2,1)
% imagesc(reshape(test_approx(check_image,:) , [250,250])), title('test projection'), colormap gray, axis image; %+ mean_total
% subplot(1,2,2)
% imagesc(reshape(center_test(check_image,:) , [250,250])), title('test original'), colormap gray, axis image; %+ mean_total


%% accuracy for test data

    diff_test_male = W_test - W_train_male;
    diff_test_female = W_test - W_train_female;

    dist_test_male = sqrt(sum(diff_test_male.^2,1));
    dist_test_female = sqrt(sum(diff_test_female.^2,1));

    error_test = dist_test_male>dist_test_female;

    error_test_male = [error_test_male, sum(error_test(1,1:1000) == 1)];
    error_test_female = [error_test_female, sum(error_test(1,1001:end) == 0)];

    accuracy_test_male = [accuracy_test_male, 1 - error_test_male(end) / 1000];
    accuracy_test_female = [accuracy_test_female, 1 - error_test_female(end) / 1000];
    
    
    
    end_time = toc;
    time = [time, end_time];
    
end

accuracy_test_total = (accuracy_test_male + accuracy_test_female)/2;

% plot time accuracy
figure(7)
subplot(1,3,1)
plot(K_value,accuracy_test_total);
title('all accuracy')
subplot(1,3,2)
plot(K_value,accuracy_test_male);
title('male accuracy')
subplot(1,3,3)
plot(K_value,accuracy_test_female);
title('female accuracy')

figure(8)
subplot(1,2,1)
plot(K_value,time);
title('K-value vs time')
subplot(1,2,2)
plot(time,accuracy_test_total);
title('time vs accuracy_total')


%% train projection 
% %% project all train data
% W_train = pinv(B_train) * double(center_train');
% train_approx = B_train * W_train;
% train_approx = train_approx';
% 
% 
% %% plot project test data
% % check_image2 = 1966;
% % 
% % figure(6)
% % subplot(1,2,1)
% % imagesc(reshape(train_approx(check_image2,:), [250,250])), title('train projection'), colormap gray, axis image; %+ mean_all
% % subplot(1,2,2)
% % imagesc(reshape(center_train(check_image2,:), [250,250])), title('train original'), colormap gray, axis image; % + mean_all
% 
% %% accuracy of train data
% diff_train_male = W_test - W_train_male;
% diff_train_female = W_test - W_train_female;
% 
% dist_train_male = sqrt(sum(diff_train_male.^2,1));
% dist_train_female = sqrt(sum(diff_train_female.^2,1));
% 
% error_train = dist_train_male>dist_train_female;
% 
% error_train_male = sum(error_train(1,1:1934) == 1);
% error_train_female = sum(error_train(1,1935:end) == 0);
% 
% accuracy_train_male = 1 - error_train_male / 1934;
% accuracy_train_female = 1 - error_train_female / 1934;


K_value_2 = [50,100,200];

error_train_male = [];
error_train_female = [];
accuracy_train_male = [];
accuracy_train_female = [];

time_2=[];

for i=1:3
    
    K_2 = K_value_2(i);
    
    tic;
    
    B_train = e_vector_train_total(:,end-K_2+1:end);

    W_train_male = pinv(B_train) * double(mean_male');
    W_train_female = pinv(B_train) * double(mean_female');

    mean_male_approx = B_train * W_train_male;
    mean_female_approx = B_train * W_train_female;


%% plot projection
%     figure(4)
%     subplot(1,4,i)
%     imagesc(reshape(mean_male_approx, [250,250])), title('Male projection'), colormap gray, axis image;
%     figure(5)
%     subplot(1,4,i)
%     imagesc(reshape(mean_female_approx, [250,250])), title('Female projection'), colormap gray, axis image;


%% project all test data
    W_test = pinv(B_train) * double(center_test');
    test_approx = B_train * W_test;
    test_approx = test_approx';


%% project all train data
    W_train_male = pinv(B_train) * double(center_male_train');
    train_male_approx = B_train * W_train_male;
    train_male_approx = train_male_approx';
    
    W_train_female = pinv(B_train) * double(center_female_train');
    train_female_approx = B_train * W_train_female;
    train_female_approx = train_female_approx';
    
    
%% average male and female train
    W_test_train_male = zeros(1934,2000);
    W_test_train_female = zeros(1934,2000);

    for j = 1:2000
        W_test_train_male(:,j) = (sqrt(sum((W_train_male - W_test(:,j)).^2,1)))'; 
        W_test_train_female(:,j) = (sqrt(sum((W_train_female - W_test(:,j)).^2,1)))';
    end
    
    mean_W_test_train_male = mean(W_test_train_male,1);
    mean_W_test_train_female = mean(W_test_train_female,1);
    
    error_train = mean_W_test_train_male > mean_W_test_train_female;
    
    error_train_male = [error_train_male, sum(error_train(1,1:1000) == 1)];
    error_train_female = [error_train_female, sum(error_train(1,1001:end) == 0)];

    accuracy_train_male = [accuracy_train_male, 1 - error_train_male(end) / 1000];
    accuracy_train_female = [accuracy_train_female, 1 - error_train_female(end) / 1000];
         
    end_time = toc;
    time_2 = [time_2, end_time];
    
end

accuracy_train_total = (accuracy_train_male + accuracy_train_female)/2;

figure(9)
subplot(1,3,1)
plot(K_value_2,accuracy_train_total);
title('all accuracy')
subplot(1,3,2)
plot(K_value_2,accuracy_train_male);
title('male accuracy')
subplot(1,3,3)
plot(K_value_2,accuracy_train_female);
title('female accuracy')

figure(10)
subplot(1,2,1)
plot(K_value_2,time_2);
title('K-value vs time')
subplot(1,2,2)
plot(time_2,accuracy_train_total);
title('time vs accuracy_total')

% end of code


