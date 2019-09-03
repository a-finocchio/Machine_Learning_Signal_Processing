function [obj] = compute_objective(V, B, W) 
V=V+eps;
B=B+eps;
W=W+eps;
% Your code here

obj = sum(sum(V.*log(V./(B*W)))) - sum(sum(B*W)) + sum(sum(V));
%formula in class ppt

%obj = sum(sum(V.*log(V./(B*W)))) + sum(sum(B*W)) - sum(sum(V));
%formula in paper



end