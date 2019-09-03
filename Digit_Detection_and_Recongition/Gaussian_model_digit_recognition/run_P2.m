
% author : Huixiang Li

%% INITIALIZE
clc
clear
close all
tic;


%% LOAD DATA
train_img = loadMNISTImages('train-images-idx3-ubyte');
train_lbl = loadMNISTLabels('train-labels-idx1-ubyte');
test_img = loadMNISTImages('t10k-images-idx3-ubyte');
test_lbl = loadMNISTLabels('t10k-labels-idx1-ubyte');
length = size(test_lbl,1);


%% PCA DATA
% normalize
train_mean = mean(train_img, 2);
train_center = train_img - train_mean;
test_center = test_img - train_mean;

% PCA
dimension = 45;
train_corr = train_center * train_center';
[e_vec_train, e_val_train] = eigs(train_corr,dimension);
e_vec_train = real(e_vec_train);
e_val_train = abs(e_val_train);
B_train = e_vec_train;
W_train = pinv(B_train) * double(train_center);
B_test = e_vec_train;
W_test = pinv(B_test) * double(test_center);


%% ORIGINAL DATA ERROR
[predict_f_ori, predict_d_ori] = GMM_CLASSIFY(train_img, train_lbl, test_img, test_lbl,'ORI');
[accuracy_f_ori, accuracy_d_ori, acc_f_ori, acc_d_ori] = ACCURACY(predict_f_ori, predict_d_ori, test_lbl);
toc;


%% PCA DATA ERROR
[predict_f_pca, predict_d_pca] = GMM_CLASSIFY(W_train, train_lbl, W_test, test_lbl,'PCA');
 [accuracy_f_pca, accuracy_d_pca, acc_f_pca, acc_d_pca] = ACCURACY(predict_f_pca, predict_d_pca, test_lbl);
toc;

%% SAVE DATA
% mkdir results

% digit mean image
for i = 1:10
    figure();
    temp_lbl = i-1;
    train_i = train_img(:, train_lbl == temp_lbl);
    digit_mean = mean(train_i ,2);   
    imshow(reshape(digit_mean,28,28));
    saveas(gcf,sprintf('./results/digit_mean_%i.png',temp_lbl));
end

% result txt
RowName = {'overall';'class 0';'class 1';'class 2';'class 3';'class 4';'class 5';'class 6';'class 7';'class 8';'class 9';};
ORI_full_accuracay = [accuracy_f_ori; acc_f_ori];
ORI_diag_accuracay = [accuracy_d_ori; acc_d_ori];
PCA_full_accuracay = [accuracy_f_pca; acc_f_pca];
PCA_diag_accuracay = [accuracy_d_pca; acc_d_pca];
ORI_full_acc = num2str(ORI_full_accuracay,'%.4f');
ORI_diag_acc = num2str(ORI_diag_accuracay,'%.4f');
PCA_full_acc = num2str(PCA_full_accuracay,'%.4f');
PCA_diag_acc = num2str(PCA_diag_accuracay,'%.4f');
T = table( ORI_full_acc,ORI_diag_acc,PCA_full_acc,PCA_diag_acc, 'RowNames',RowName);
writetable(T, './results/results.txt','Delimiter','\t','WriteRowNames',true)
toc;

%% SUB_FUNCTION
% accuracy
function [accuracy_f_ori, accuracy_d_ori, acc_f_ori, acc_d_ori] = ACCURACY(predict_f_ori, predict_d_ori, test_lbl,flag)
    len = size(test_lbl,1);
    % total accuracy
    logic_f_ori = (predict_f_ori == test_lbl);
    logic_d_ori = (predict_d_ori == test_lbl);
    error_f_ori = 1- sum(logic_f_ori)/len;
    error_d_ori = 1- sum(logic_d_ori)/len;
    accuracy_f_ori = 1 - error_f_ori;
    accuracy_d_ori = 1 - error_d_ori;
    % each accuracy
    acc_f_ori = zeros(10,1);
    acc_d_ori = zeros(10,1);
    
    for i = 1:10
        temp_lbl = i-1;
        temp_f = predict_f_ori(test_lbl==temp_lbl);
        temp_d = predict_d_ori(test_lbl==temp_lbl);
        total_f = size(temp_f,1);
        total_d = size(temp_d,1);
        acc_f_ori(i,:) = length(temp_f(temp_f==temp_lbl))/total_f;
        acc_d_ori(i,:) = length(temp_d(temp_d==temp_lbl))/total_d;
    end

end

% GMM
function [predict_lbl_f, predict_lbl_d] = GMM_CLASSIFY(train_img, train_lbl, test_img, test_lbl,flag)
    dim = size(train_img,1);
    
    % pai miu sigma
    pai = zeros(10,1);
    miu = zeros( dim,10);
    sigmad = zeros( dim, dim, 10);
    sigmaf = zeros(dim, dim, 10);
    for i = 1:10
        temp_lbl = i-1;
        train_i = train_img(:, train_lbl == temp_lbl);
        pai(i) = size(train_i ,1)/60000;
        miu(:,i) = mean(train_i ,2);   
        temp_cor = train_i  - miu(:,i);
        temp_corr  = temp_cor * temp_cor'./size(temp_cor,2);
        sigmaf(:,:,i) =  temp_corr;
        sigmad(:,:,i) = diag(diag(temp_corr)) ; 
    end
    
    % initialize
    rou = rand(1,1)/5;
    pinv_sigma_f = zeros(dim, dim,10);
    pinv_sigma_d = zeros(dim, dim,10);
    lg_det_f = zeros(10, 1);
    lg_det_d = zeros(10, 1);
    if flag == 'PCA'
        for i = 1: 10
            pinv_sigma_f(:,:,i) = pinv(sigmaf(:,:,i));
            pinv_sigma_d(:,:,i) = pinv(sigmad(:,:,i));
            lg_det_f(i,:) =  -0.5*log(det(sigmaf(:,:,i)+rou*eye(dim,dim)));
            lg_det_d(i,:) =  -0.5*log(det(sigmad(:,:,i)+rou*eye(dim,dim)));
        end
    end
    if flag == 'ORI'
        for i = 1: 10
            pinv_sigma_f(:,:,i) = pinv(sigmaf(:,:,i));
            pinv_sigma_d(:,:,i) = pinv(sigmad(:,:,i));
            lg_det_f(i,:) =  -0.5*log(trace(sigmaf(:,:,i)));
            lg_det_d(i,:) =  -0.5*log(trace(sigmad(:,:,i)));
        end
    end
    % classify
    length = size(test_lbl,1);
    predict_lbl_d = zeros(length,1);
    predict_lbl_f = zeros(length,1);
    for j =1: length
        test_i = test_img(:,j);
        Pd = zeros(10,1);
        Pf = zeros(10,1);
        for i = 1:10
            temp1f = -0.5*(test_i - miu(:,i))' * pinv_sigma_f(:,:,i) * (test_i - miu(:,i));
            temp1d =  -0.5*(test_i - miu(:,i))' * pinv_sigma_d(:,:,i) * (test_i - miu(:,i));
            Pf(i,:) = log(pai(i)) + temp1f  + lg_det_f(i,:);
            Pd(i,:) = log(pai(i)) + temp1d + lg_det_d(i,:);
        end
        [~,idxf] = max(Pf);
        [~,idxd] = max(Pd);
        predict_lbl_f(j,:) = idxf-1;
        predict_lbl_d(j,:) = idxd-1;   
    end
end
