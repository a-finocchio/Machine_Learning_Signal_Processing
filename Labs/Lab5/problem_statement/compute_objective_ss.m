function [obj] = compute_objective_ss(V,B,W, alpha, beta) 
V=V+eps;
B=B+eps;
W=W+eps;
% Your code here
%obj = sum(sum(V.*log(V./(B*W)))) + sum(sum(B*W)) - sum(sum(V)) + alpha * sum(sum(W)) + beta * sum(sum(B));
%formula in paper

obj = sum(sum(V.*log(V./(B*W)))) - sum(sum(B*W)) + sum(sum(V)) + alpha * sum(sum(W)) + beta * sum(sum(B));
%formula in class ppt

end