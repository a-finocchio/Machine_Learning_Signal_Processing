%% lab 9 DTW MLSP
%author: Hixiang Li

%% initialization
clc
clear 
close all

%% read in data
train_directory  = './Final_digits/';
train_file_name = fullfile(train_directory, '*.mat');
train_file_dir = dir(train_file_name);
train_n_file = length(train_file_dir);
for i = 1:train_n_file
  train_file_name = train_file_dir(i).name;
  train_data_name = fullfile(train_directory, train_file_name);  
  temp_struct = load(train_data_name); 
  temp = temp_struct.feat;
  [row,pitch,yaw] = size(temp);
  data_train{i}=double(reshape(temp,pitch,yaw));
end

%% computer optimal distance
record = zeros(300,300);
for i = 1:300
    i
    for j = 1:300
        T = data_train{i};
        R = data_train{j};
        %1.dissimilarity matrix
        S = similiar(T,R);
        %2.compute DTW
        [D,idx] = DTW(S);
        % back tracking
        dist = backtrack(D,idx);
        record(i,j) = dist;
    end
end    

%% sort nearest
sort_matrix = zeros(300,29);
for round = 1:300
    [min_value, min_idx] = mink(record(round,:),30);
    sort_matrix(round,:) = min_idx(2:end);
end   


%% transfer to num
num_matrix = zeros(300,29);
for s_i = 1:300
    for s_j = 1:29
       temp = sort_matrix(s_i,s_j);
       num_matrix(s_i,s_j) = ceil(temp/30)-1;
    end
end


%% classify
class_matrix = zeros(300,1);
for c = 1:300
    class_matrix(c,:) = mode(num_matrix(c,:));
end    

%% build true matrix
truth = zeros(300,1);
for t = 1:300
    truth(t,1) = ceil(t/30)-1;
end    


%% accuracy for over_all
error = (class_matrix == truth);
accuracy = 1 - sum(error)/300;

%% accuracy for each digit
acc_m = zeros(10,1);
for m = 1:10
    down = 30*(m-1)+1;
    up = 30*m;
    temp_truth = truth(down:up,1);
    temp_dig = class_matrix(down:up,:);
    error_temp = (temp_truth == temp_dig);
    acc_m(m) = 1-sum(error_temp)/30;
end    
        
%% confuison
Confusion = confusionmat(truth,class_matrix);    
imagesc(Confusion)
        
        
