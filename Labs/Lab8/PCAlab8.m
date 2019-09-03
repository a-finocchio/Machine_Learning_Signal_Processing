function [TrainingPCA,TestingPCA] = PCAlab8(trainingdata,testingdata,PCAdimension)

%% traning data
typical_face = mean(trainingdata, 2);

% train_difference_face  = zeros(40,9,10304);
% for i = 1:40
%     for j = 1:9
%         train_difference_face(i,j,:) = (trainingdata(i,j,:) - typical_face);
%     end    
% end
train_difference_face = (trainingdata - typical_face);
% train_center = double(reshape(train_difference_face, [], 10304));
corr_matrix = train_difference_face * train_difference_face' ;
[e_vector, e_values] = eigs(corr_matrix, PCAdimension);

%e_vector = e_vector/e_values;

% e_vector = real(e_vector);
% e_values = abs(e_values);

% TrainingPCA = pinv(e_vector) * (train_center)';

TrainingPCA = pinv(e_vector) * train_difference_face ; 

%% testing data
% test_difference_face  = zeros(40,1,10304);
% for i = 1:40
%       test_difference_face(i,1,:) = (testingdata(i,1,:) - typical_face);
%         
% end
% test_center = double(reshape(test_difference_face, [], 10304));

test_difference_face  = testingdata - typical_face;
TestingPCA = pinv(e_vector) * test_difference_face ;
