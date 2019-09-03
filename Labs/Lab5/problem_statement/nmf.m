function [B,W,obj,k] = nmf(V,rank,max_iter,lambda)
% NMF - Non-negative matrix factorization
% [B,W,OBJ,NUM_ITER] = NMF(V,RANK,MAX_ITER,LAMBDA) 
% V - Input data.
% RANK - Rank size.
% MAX_ITER - Maximum number of iterations (default 50).
% LAMBDA - Convergence step size (default 0.0001). 
% B - Set of basis images.
% W - Set of basis coefficients.
% OBJ - Objective function output.
% NUM_ITER - Number of iterations run.

% Your code here


[row, col] = size(V);

B = rand(row,rank);
W = rand(rank,col);
%ONE = ones(row,col);
ONE1 = ones(1,col);
ONE2 = ones(row,1);
objective_value = compute_objective(V, B, W);
V=V+eps;
B=B+eps;
W=W+eps;

W_sum = W;

for i = 1:rank
    W(:,i)=W(:,i)/sum(W_sum(:,1));
end    




for i = 1:max_iter
    B = B .* (((V./(B*W))* W.')./( ONE1* W.'));

    W = W.* ((B.'*(V./(B*W)))./(B.'*ONE2));

    W_sum_new = W;
    for j = 1:rank
        W(:,j)=W(:,j)/sum(W_sum_new(:,1));
    end    

    objective_value_new = compute_objective(V, B, W);
      
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


