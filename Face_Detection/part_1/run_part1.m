%% Driver File for Problem 1: Part 1: Face Detection
% Implement a simple face detector that can detect faces in group 
% photos of people.
% Author Name : Huixiang Li


%% Your Driver Script Starts Here
% Follow the steps in methodology here
clear
close all
clc

%% Read in training data
train_directory  = '../data/lfw_1000/';
train_file = fullfile(train_directory, '*.pgm');
train_img_dir = dir(train_file);
%out put image amount
n_train_img = length(train_img_dir)
train_image = zeros(n_train_img,64,64);

for i = 1:n_train_img
  train_img_file_name = train_img_dir(i).name;
  train_img_name = fullfile(train_directory, train_img_file_name);
  
  %store img into matrix 
  train_image(i,:,:) = double(imread(train_img_name));
  
  %display each image for double-check
%   imagesc(imread(train_img_name)), axis image, colormap gray, title('show training data');  
%   drawnow; 
  
end


%% Building image matrix
image_size = 64*64;
train_reshape = reshape(train_image, [1071, image_size]);


%% normalize data
% Your code goes here
typical_face = mean(train_reshape, 1);
typical_face_reshape = reshape(typical_face, 64,64);
typical_face_repmat = repmat(typical_face,1071,1);
train_reshape = train_reshape - typical_face_repmat;


%% Computing Eigen faces
corr_matrix = train_reshape' * train_reshape;
[e_vector, e_values] = eig(corr_matrix);

e_vector = real(e_vector);
e_values = abs(e_values);
e_vector_most = reshape(e_vector(:,end), 64,64);
e_vector_most = e_vector_most + typical_face_reshape;


%% plot to check Eigen value & faces
%show e-values
figure(1)
plot(diag(e_values)), title('e-values');

%show e-faces
figure(2)
imagesc(e_vector_most), title('the most important eigen-face'),axis image, colormap gray; 


%% Scanning an Image or Pattern
scale_value = [1, 0.5, 0.75, 1.5, 2.0]; 


%% Beatles
%test grey image first
test_img_directory1  = '../data/group_photos/Beatles.jpg';
test_img1 = double(imread(test_img_directory1));
%change rgb to grey 
if size(test_img1,3)==3
    test_img1 = squeeze(mean(test_img1,3));
end
%filter it
test_img1 = imresize(test_img1,0.85);
test_img1 = imgaussfilt(test_img1,1.5,'FilterSize',[5,5]);

%scan the score
score1 = score_record_fun(test_img1,e_vector_most,1,scale_value);

%merge and find peaks center
gray_scale_img1 = mat2gray(score1);
final_score1=imbinarize(gray_scale_img1, 0.79);
center_of_peak = regionprops(final_score1,'centroid');
center_of_peak = cell2mat(struct2cell(center_of_peak));  
[a,b] = size(center_of_peak); 
center1 = reshape(center_of_peak,2,b/2);
center1 = round(center1);

figure
imshow(final_score1), title('final score1');

%center in image
center_in_image1 = zeros(size(center1));
for i = 1:size(center1,2)
     center_in_image1(1,i) = center1(1,i)+63;
     center_in_image1(2,i) = center1(2,i)+63; 
end
figure
imagesc(uint8(test_img1));
hold on
for i = 1:size(center1,2)   
   rectangle('Position',[center_in_image1(1,i),center_in_image1(2,i),70, 70],'Edgecolor', 'r');
end
caxis ([0 3000])
colorbar



%% Solvay
%test grey image first
test_img_directory2  = '../data/group_photos/Solvay.jpg';
test_img2 = double(imread(test_img_directory2));
%change rgb to grey 
if size(test_img2,3)==3
    test_img2 = squeeze(mean(test_img2,3));
end
%filter it
test_img2 = imresize(test_img2,1.5);
test_img2 = imgaussfilt(test_img2,1.5,'FilterSize',[5,5]);

%scan the score
score2 = score_record_fun(test_img2,e_vector_most,5,scale_value);

%merge and find peaks center
gray_scale_img2 = mat2gray(score2);
final_score2=imbinarize(gray_scale_img2, 0.4);
center_of_peak2 = regionprops(final_score2,'centroid');
center_of_peak2 = cell2mat(struct2cell(center_of_peak2));  
[a,b] = size(center_of_peak2); 
center2 = reshape(center_of_peak2,2,b/2);
center2 = round(center2);

%center in image
center_in_image2 = zeros(size(center2));
for i = 1:size(center2,2)
     center_in_image2(1,i) = center2(1,i)+63;
     center_in_image2(2,i) = center2(2,i)+63; 
end

% figure
% imagesc(uint8(test_img2));
% hold on
% for i = 1:size(center2,2)   
%    rectangle('Position',[center_in_image2(1,i),center_in_image2(2,i),70, 70],'Edgecolor', 'r');
% end
% caxis ([0 3000])
% colorbar



%% G8
%test grey image first
test_img_directory3  = '../data/group_photos/G8.jpg';
test_img3 = double(imread(test_img_directory3));
%change rgb to grey 
if size(test_img3,3)==3
    test_img3 = squeeze(mean(test_img3,3));
end
%filter it
test_img3 = imresize(test_img3,0.2);
test_img3 = imgaussfilt(test_img3,1.5,'FilterSize',[3,3]);

%scan the score
score3 = score_record_fun(test_img3,e_vector_most,4,scale_value);

%merge and find peaks center
gray_scale_img3 = mat2gray(score3);
final_score3 = imbinarize(gray_scale_img3, 0.65);
center_of_peak3 = regionprops(final_score3,'centroid');
center_of_peak3 = cell2mat(struct2cell(center_of_peak3));  
[a,b] = size(center_of_peak3); 
center3 = reshape(center_of_peak3,2,b/2);
center3 = round(center3);

%center in image
center_in_image2 = zeros(size(center3));
for i = 1:size(center3,2)
     center_in_image2(1,i) = center3(1,i)+63;
     center_in_image2(2,i) = center3(2,i)+63; 
end

% figure
% imagesc(uint8(test_img3));
% hold on
% for i = 1:size(center3,2)   
%    rectangle('Position',[center_in_image2(1,i),center_in_image2(2,i),70, 70],'Edgecolor', 'r');
% end
% caxis ([0 3000])
% colorbar


%% Big_3
%test grey image first
test_img_directory4  = '../data/group_photos/Big_3.jpg';
test_img4 = double(imread(test_img_directory4));
%change rgb to grey 
if size(test_img4,3)==3
    test_img4 = squeeze(mean(test_img4,3));
end
%filter it
test_img4 = imresize(test_img4,1.2);
test_img4 = imgaussfilt(test_img4,1.5,'FilterSize',[5,5]);

%scan the score
score4 = score_record_fun(test_img4,e_vector_most,4,scale_value);

%merge and find peaks center
gray_scale_img4 = mat2gray(score4);
final_score4 = imbinarize(gray_scale_img4, 0.6);
center_of_peak4 = regionprops(final_score4,'centroid');
center_of_peak4 = cell2mat(struct2cell(center_of_peak4));  
[a,b] = size(center_of_peak4); 
center4 = reshape(center_of_peak4,2,b/2);
center4 = round(center4);

%center in image
center_in_image2 = zeros(size(center4));
for i = 1:size(center4,2)
     center_in_image2(1,i) = center4(1,i)+63;
     center_in_image2(2,i) = center4(2,i)+63; 
end

% figure
% imagesc(uint8(test_img4));
% hold on
% for i = 1:size(center4,2)   
%    rectangle('Position',[center_in_image2(1,i),center_in_image2(2,i),70, 70],'Edgecolor', 'r');
% end
% caxis ([0 3000])
% colorbar




%% score function
function score_record_all = score_record_fun(test_img,e_vector_most,step,scale_value)

    [row, col]=size(test_img);
    score_record = zeros(row-126,col-126,step);
    
  % try different scaler for the face detect
  
  for k = 1:step
    %resize
    test_img = imresize(test_img,scale_value(k));
    I = test_img;
    E = e_vector_most;
    
    %patch size
    N = 64;
    M = 64;
%     [P,Q] = size(test_img);
    E = E/norm(E(:));
    
    %compute score
    tmpim = conv2(I, fliplr(flipud(E)),'valid');
    convolvedimage = tmpim(N:end, M:end);
    % normalize it
    % sumE = sum(E(:));
    % patchscore = convolvedimage - sumE*patchmeanofA;    

    % record score
    convolvedimage = imresize(convolvedimage, [row-126, col-126]); 
    score_record(:,:,step)= convolvedimage;
  end
    
    score_record_temp =  mean(score_record,3);
    score_record_all = reshape(score_record_temp,[row-126, col-126]);
    
end    
