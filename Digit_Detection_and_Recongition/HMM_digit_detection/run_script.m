
% author : Huixiang Li

%% Initialization
clear all
clc

%% 1 Load data
% load in train and test data
[train_data,train_lbl] = loadUnipenData('./Data/pendigits-orig.tra');
[test_data,test_lbl] = loadUnipenData('./Data/pendigits-orig.tes');


%% 2 Plot some examples from the data
check_num = 4;
figure(1);
plotUnipen(train_data{check_num});
plotUnipen(test_data{check_num});


%% 3 Normalize data
% Reccomended code. You can add inputs/outputs to the functions. You can
% use additional code:
[train_DataNorm] = normUnipen(train_data);
[test_DataNorm] = normUnipen(test_data);


%% plot normalized data
for i = 1:3
    figure(2)
%     pic = randi([1 3000],1,1);
    pic = i+250;
    subplot(2,3,i)
    plotUnipen(train_data{pic});
    subplot(2,3,i+3)
    plot(train_DataNorm{pic}(1,:),train_DataNorm{pic}(2,:));
end
fprintf('\n normalize the data can let the handwriting become more suitable for the image ')
fprintf('\n also, the normalization can let all the handwriting in a similiar size')
fprintf('\n which is good for the trainning and classification ')
fprintf('\n therefore, the normailization is necessary')


%%  kmeans
% Reccomended code. You can add inputs/outputs to the functions. You can
% use additional code:
[traiDataK, testDataK] = adaptPenKmeans(train_DataNorm, test_DataNorm);


%% HMM training and testing

% train
state1 =5;
ESTTR={};
ESTEMIT={};

for i = 1:10
    index = find(train_lbl == i-1);
    seq = traiDataK(index);

    TRGUESS = rand(state1,state1);
    TRGUESS =  TRGUESS ./ sum(TRGUESS,2);
    
    EMITGUESS = rand(state1,256);
    EMITGUESS = EMITGUESS ./ sum(EMITGUESS,2);
 
    [ESTTR{i},ESTEMIT{i}] = hmmtrain(seq,TRGUESS,EMITGUESS,'MAXITERATIONS',200,'Tolerance',1e-4);  
end


% test
n_test = size(test_data,2);
predict = zeros(n_test,10);

for i = 1:n_test
    seq = testDataK{i};
    for j = 1:10
        [~, predict(i,j)] = hmmdecode(seq,ESTTR{j},ESTEMIT{j});
    end
end


%% accuracy
[~,result] =  max(predict,[],2);
result = result -1;
error = length(result(result~=test_lbl'));
accuracy = 1- error/n_test;


%% Change state
iter = [200,100,500];
tol = [1e-4, 1e-5, 1e-6];
state1 = [5,8,15];

%% tol
predict_t={};
accuracy_t={};
for i=2:length(state1)
    [predict_t{i-1},accuracy_t{i-1}] = HMM_train_and_predict(traiDataK,testDataK, train_lbl, test_lbl, iter(1),tol(i), state1(1));
end   

%% iter
predict_i={};
accuracy_i={};
for i=2:length(state1)
    [predict_i{i-1},accuracy_i{i-1}] = HMM_train_and_predict(traiDataK,testDataK, train_lbl, test_lbl, iter(i),tol(1), state1(1));
end    

%% state
predict_s={};
accuracy_s={};
for i=2:length(state1)
    [predict_s{i-1},accuracy_s{i-1}] = HMM_train_and_predict(traiDataK,testDataK, train_lbl, test_lbl, iter(1),tol(1), state1(i));
end    

%% 
fprintf('\n ')
fprintf('\n the best accuracy i got is 88.16 percent')
fprintf('\n the parameter for it is, 8 states, tol = 1e-4, iter = 200 ')
fprintf('\n the accuracy will increase as the state increases, but the calculation will then become longer ')











