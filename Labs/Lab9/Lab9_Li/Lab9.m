%% Initialize
clear
close all

%% Load train data
train_img = loadMNISTImages('train-images-idx3-ubyte');
train_lbl = loadMNISTLabels('train-labels-idx1-ubyte');
% test_img = loadMNISTImages('t10k-images-idx3-ubyte');
% test_lbl = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% Build Xi & centerlize
X = cell(1,10);
X_mean = cell(1,10);
X_center = cell(1,10);
for i = 1:10
    label_temp = i-1;
    X{i} = train_img(:,train_lbl==label_temp);
    X_mean{i} = mean(X{i},2);
    X_center{i} = X{i} - X_mean{i};
end

%% Check Xi
% pic = X{1,1};
% imagesc(reshape(pic(:,1),[28,28]));

%% Randomly initialize W and sigma
K = 100;
% K = 50;

row = 784;
W = randn(row,K);
sigma = abs(randn(1,1));
sigma_square = sigma^2;


%% log iteration
log_likelihood = zeros(10,10);
W_all = {};
for i = 1:10 
    X_temp = X_center{i};
    for iter = 1:10 
        iter; 
        %converge test
        if(iter>5 &&  (log_likelihood(iter,i)-log_likelihood(iter-1,i))/log_likelihood(iter-1,i)<= 0.0001)
             break;
             % after some tests, all likelihood is increasing
        end      
        [D,N] = size(X_temp);
        [Ez,Ez_zt,V] = E_step(W,X_temp,sigma_square,K,N);
        log_likelihood_temp = likelihood(W,X_temp,K,sigma_square,D,N);
        [sigma_new,W] = M_step(X_temp,Ez,Ez_zt,N,sigma_square,V,D); 
        sigma_square = sigma_new;
        log_likelihood(iter,i) = log_likelihood_temp;
    end
    W_all{i} = W;
end  

%% generilized image
for im = 1:10
    figure(im)
    for i = 1:25
%         z_gen = 0.075*randn(K,1); % balance the effect from randomlization
        z_gen = randn(K,1);
        x_gen = W_all{im} * z_gen + X_mean{im};
        subplot(5,5,i);
        imagesc(reshape(x_gen,[28,28])), axis image, colormap gray;  
    end 
end    