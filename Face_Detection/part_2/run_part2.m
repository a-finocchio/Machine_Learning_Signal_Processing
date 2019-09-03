%% Driver File for Problem 1: Part 2: Face Detection
% You will implement an Adaboost Classifier to classify between face images
% and non-face images.
% Author Name : Huixiang Li

% clear;
clc;
close all;
clear;


%% read in training data fron lfw
train_face_directory  = '../data/lfw_1000/';
train_face_file = fullfile(train_face_directory, '*.pgm');
train_face_img_dir = dir(train_face_file);
%out put image amount
n_train_face_img = length(train_face_img_dir)
train_face_image = zeros(n_train_face_img,19,19);

for i = 1:n_train_face_img
  train_img_face_file_name = train_face_img_dir(i).name;
  train_img_face_name = fullfile(train_face_directory, train_img_face_file_name);
  
  %store img into matrix 
  train_face_image_temp = double(imread(train_img_face_name));
  train_face_image(i,:,:) = imresize(train_face_image_temp,[19,19]);
  %display each image for double-check
%   imagesc(imread(train_img_face_name)), axis image, colormap gray, title('show training data');  
%   drawnow; 
%   
end

%% Building image matrix
image_size = 19*19;
train_face_reshape2 = reshape(train_face_image, [n_train_face_img, image_size]);



%% read in training data from adaboost
train_face_directory1  = '../data/boosting_data/train/face/';
train_face_file1 = fullfile(train_face_directory1, '*.pgm');
train_face_img_dir1 = dir(train_face_file1);
%out put image amount
n_train_face_img1 = length(train_face_img_dir1)
train_face_image1 = zeros(n_train_face_img1,19,19);

for i = 1:n_train_face_img1
  train_img_face_file_name1 = train_face_img_dir1(i).name;
  train_img_face_name1 = fullfile(train_face_directory1, train_img_face_file_name1);
  
  %store img into matrix 
  
  train_face_image1(i,:,:) = double(imread(train_img_face_name1));
  %display each image for double-check
%   imagesc(imread(train_img_face_name1)), axis image, colormap gray, title('show training data');  
%   drawnow; 
  
end

%% Building image matrix
image_size1 = 19*19;
train_face_reshape1 = reshape(train_face_image1, [n_train_face_img1, image_size1]);


%% normalize data
% Your code goes here
typical_face = mean(train_face_reshape2, 1);
typical_face_reshape = reshape(typical_face, 19,19);
typical_face_repmat = repmat(typical_face,n_train_face_img,1);
train_face_reshape = train_face_reshape2 - typical_face_repmat;

% unit variance
% stdd = std(train_face_reshape2,0,2);
% train_face_reshape = train_face_reshape./stdd;

% since the result when using this method is only 70%, to achieve better,
% I mute the unit variance command

%% Computing Eigen faces
K=[10];
corr_matrix = train_face_reshape' * train_face_reshape;
[e_face_vector, e_face_values] = eigs(corr_matrix,K(1));

e_face_vector = real(e_face_vector);
e_face_values = abs(e_face_values);
e_vector_most = reshape(e_face_vector(:,end), 19,19);
e_vector_most = e_vector_most + typical_face_reshape;


%% plot to check Eigen value & faces
%show e-values
figure(1)
plot(diag(e_face_values)), title('e-values');

%show e-faces
figure(2)
imagesc(e_vector_most), title('the most important eigen-face'),axis image, colormap gray; 



%% face express
weight_face = pinv(e_face_vector) * train_face_reshape1';


%% test expression
first_face =  e_face_vector * weight_face(:,1);
figure
imagesc(reshape(first_face,19,19)+typical_face_reshape),axis image, colormap gray;


%% %% read in training data from adaboost
train_non_directory  = '../data/boosting_data/train/non-face/';
train_non_file = fullfile(train_non_directory, '*.pgm');
train_non_img_dir = dir(train_non_file);
%out put image amount
n_train_non_img = length(train_non_img_dir)
train_non_image = zeros(n_train_non_img,19,19);

for i = 1:n_train_non_img
  train_img_non_file_name = train_non_img_dir(i).name;
  train_img_non_name = fullfile(train_non_directory, train_img_non_file_name);
  
  %store img into matrix 
  train_non_image(i,:,:) = double(imread(train_img_non_name));
  
  %display each image for double-check
%   imagesc(imread(train_img_non_name)), axis image, colormap gray, title('show training data');  
%   drawnow; 
  
end

%% Building image matrix
image_size = 19*19;
train_non_reshape = reshape(train_non_image, [n_train_non_img, image_size]);
typical_non_repmat = repmat(typical_face,n_train_non_img,1);

% %% normalize data
% Your code goes here
train_non_reshape = train_non_reshape - typical_non_repmat;



%% non face express
weight_non = pinv(e_face_vector) * train_non_reshape';


%% load test
%% read in testing data
test_face_directory  = '../data/boosting_data/test/face/';
test_face_file = fullfile(test_face_directory, '*.pgm');
test_face_img_dir = dir(test_face_file);
%out put image amount
n_test_face_img1 = length(test_face_img_dir)
test_face_image1 = zeros(n_test_face_img1,19,19);

for i = 1:n_test_face_img1
  test_img_face_file_name1 = test_face_img_dir(i).name;
  test_img_face_name1 = fullfile(test_face_directory, test_img_face_file_name1);
  
  %store img into matrix 
  
  test_face_image1(i,:,:) = double(imread(test_img_face_name1));
  %display each image for double-check
%   imagesc(imread(test_img_face_name1)), axis image, colormap gray, title('show training data');  
%   drawnow; 
  
end

%% Building image matrix
image_size1 = 19*19;
test_face_reshape1 = reshape(test_face_image1, [n_test_face_img1, image_size1]);


%% read in testing data
test_non_directory  = '../data/boosting_data/test/non-face/';
test_non_file = fullfile(test_non_directory, '*.pgm');
test_non_img_dir = dir(test_non_file);
%out put image amount
n_test_non_img1 = length(test_non_img_dir)
test_non_image1 = zeros(n_test_non_img1,19,19);

for i = 1:n_test_non_img1
  test_img_non_file_name1 = test_non_img_dir(i).name;
  test_img_non_name1 = fullfile(test_non_directory, test_img_non_file_name1);
  
  %store img into matrix 
  
  test_non_image1(i,:,:) = double(imread(test_img_non_name1));
  %display each image for double-check
%   imagesc(imread(test_img_non_name1)), axis image, colormap gray, title('show training data');  
%   drawnow; 
  
end

%% Building image matrix
image_size1 = 19*19;
test_non_reshape1 = reshape(test_non_image1, [n_test_non_img1, image_size1]);



%% project test 
weight_non_test = pinv(e_face_vector) * test_non_reshape1';
weight_face_test = pinv(e_face_vector) * test_face_reshape1';



%% adaboost

%initialize
%n_train_face_img1 =1;
label_true = [ones(1, n_train_face_img1), -1* ones(1, n_train_non_img)];
Dt = ones(1,n_train_face_img1+n_train_non_img)/(n_train_face_img1+n_train_non_img);
weight_all = [weight_face, weight_non];

et=zeros(10,1);
threshold_record = zeros(10,1);
sign_record = zeros(10,1);
alpha_record = zeros(10,1);


% buid a classifer structure, level number, and step unmber
step_num = 150;
base_classifer_num = 10; 


    %for each weight define a classifer threshold
for t = 1:base_classifer_num
    
%bascially we are supposed to used 10 bases, but after serval tests, build the structure
%layers by layers would not have good accuracy 
    
    
%lets build a new structure no just layer by layer, we do 3E1, then add
%E2345678
     
      if t<3
        weight_t = weight_all(1,:);
      else  
        weight_t = weight_all(t-2,:);
      end  
      % weight_t = weight_all(t,:);
      
      %define step_size
      max_weight = max(weight_t);
      min_weight = min(weight_t);
      range_weight = max_weight - min_weight;
      step_size = range_weight/step_num;
        
      %make matrix to record error
      error_t = zeros(1,step_num);
      label_record = [];
      %error_matrix = [];
      sign_t = []; 
        
      %find a good threshold for classification
      for step = 1: step_num
          threshold_temp = min_weight + step * step_size;
          %test and record label
          threshold_temp = repmat(threshold_temp,1,length(weight_t));
          label_temp = weight_t > threshold_temp;
          label_temp = double(label_temp);
          label_temp(label_temp == 0 ) = -1;
          label_record = [label_record; label_temp];
          %calculate error
          error_count = label_temp ~= label_true;
          error_count = double(error_count);
          %error_matrix = [error_matrix; erro_count];
          error_temp_total = error_count * Dt';
          %check sign
          if error_temp_total > 0.5
              label_temp = -1 * label_temp;
              sign_t = [sign_t, -1];
              error_temp_total = 1 - error_temp_total; 
          else    
              sign_t = [sign_t, 1];
          end
          error_t(1,step)= error_temp_total;    
      end
        
      [et(t,1),idx] = min(error_t);

      %alpha  
      alpha = 0.5 * log((1 - et(t,1)) / et(t,1));
      
      %update weight
      for j = 1 : n_train_face_img1+n_train_non_img
         if label_record(idx, j) ~= label_true(1, j)
               Dt(1, j) = Dt(1, j) * exp(alpha);
         else
               Dt(1, j) = Dt(1, j) * exp(-1 * alpha);
         end
      end  
      Dt = Dt/sum(Dt);
      
      %record constant          
      alpha_record(t,1) = alpha;      
      threshold_record(t,1) = min_weight + idx * step_size;
      sign_record(t,1) = sign_t(idx);       
end
 
n_test_img = n_test_non_img1 + n_test_face_img1;

classified_label = zeros(1,n_test_img);


% classificastion
weight_all_test = [weight_face_test, weight_non_test];


for m = 1:n_test_img
    total_score = 0;
    weight_current_classfier = weight_all_test(:,m);
    for n = 1:10
        threshold_temp_2 = threshold_record(n,1);
        if weight_current_classfier(n,:)>threshold_temp_2
            current_score = 1*sign_record(n,:) * alpha_record(n,:);
        else
            current_score = -1* sign_record(n,:) * alpha_record(n,:);
        end
        total_score = total_score + current_score;
    end   
    classified_label(1,m) = sign(total_score);    
end    
    
label_true2 = [ones(1,n_test_face_img1), -1*ones(1,n_test_non_img1)];

logic_classified = classified_label ~= label_true2;
error_classified = double(logic_classified);
error_rate = sum(error_classified)/n_test_img;
accuracy = 1 - error_rate;



