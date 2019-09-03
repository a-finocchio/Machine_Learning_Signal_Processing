function [B,W,obj,k] = ssnmf(V,rank,max_iter,lambda,alpha,beta) 
% SSNMF - Non-negative matrix factorization
% [W,H,OBJ,NUM_ITER] = SSNMF(V,RANK,MAX_ITER,LAMBDA)
% V - Input data.
% RANK - Rank size.
% MAX_ITER - Maximum number of iterations (default 50). 
% LAMBDA - Convergence step size (default 0.0001).
% ALPHA - Sparse coefficient for W.
% BETA - Sparse coefficient for B.
% W - Set of basis images.
% H - Set of basis coefficients.
% OBJ - Objective function output.
% NUM_ITER - Number of iterations run.

% Your code here


[row, col] = size(V);

B = rand(row,rank);
W = rand(rank,col);
%ONE = ones(row,col);
ONE1 = ones(1,col);
ONE2 = ones(row,1);
objective_value = compute_objective_ss(V, B, W, alpha, beta);
V=V+eps;
B=B+eps;
W=W+eps;

W_sum = W;

for i = 1:rank
    W(:,i)=W(:,i)/sum(W_sum(:,1));
end    


%val = [];

for i = 1:max_iter
    B = B .* (((V./(B*W))* W.')./( ONE1* W.'+ beta));

    B_sum_new = B;
    for l = 1:rank
        B(:,l)=B(:,l)/(sum(B_sum_new(:,l))+beta);
    end  
    
    W = W.* ((B.'*(V./(B*W)))./(B.'*ONE2 + alpha));


    W_sum_new = W;
    for j = 1:rank
        W(:,j)=W(:,j)/(sum(W_sum_new(:,1))+beta);
    end    

    objective_value_new = compute_objective_ss(V, B, W, alpha, beta);
    
  %  abs(objective_value_new - objective_value)
    
    if(abs(objective_value_new - objective_value) <= lambda)
        k=i;
        break;
    end
    
    if(i == max_iter)
        k = max_iter;
    end
        
    objective_value = objective_value_new;
    obj = objective_value;
end    

end

