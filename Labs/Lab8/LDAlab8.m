function [V,D,TrainingLDA,TestingLDA] = LDAlab8(TrainingPCA,TestingPCA,LDAdimension)

%% reshape

% [r_test,c_test] = size(TestingPCA);
% n_class  = 40;
% out   = permute(reshape(TestingPCA,[r_test,c_test/n_class,n_class]),[1,2,3]);

[row,col] = size(TrainingPCA);
% 
% test_pca_reshape = reshape(TestingPCA,[row,1,40]);
% train_pca_reshape = reshape(TrainingPCA,[row,9,40]);


%% mk and m
mk = zeros(row,40);
for i =1:40
    mk(:,i) = mean(TrainingPCA(:,(i-1)*9+1:i*9),2);
end

m = mean(mk,2);



%% SB
Sb = zeros(row,row);

for i = 1:40
    Sb_temp = 9*(mk(:,i)-m)*(mk(:,i)-m)';
    Sb = Sb + Sb_temp;
end  


%% SW
Sw = zeros(row,row);

for j = 1:360
        Sw_temp = (TrainingPCA(:,j)-mk(:,ceil(j/9)))*(TrainingPCA(:,j)-mk(:,ceil(j/9)))';
%         size(Sw_temp)
        Sw = Sw + Sw_temp;
end   


%% SVD
% invsw = inv(Sw);
% V1 = invsw*Sb;
size(Sb);
size(Sw);

[V, D]=eigs( Sb , Sw , LDAdimension-1);


%% do projection
TrainingLDA = pinv(V)*TrainingPCA;
TestingLDA = pinv(V)*TestingPCA;


