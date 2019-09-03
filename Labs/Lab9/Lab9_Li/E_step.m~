function [Ez,Ez_zt,V]= E_step(W,X_temp,sigma_square,K,N)
    M = W'*W + sigma_square * eye(K);
    U = chol(M);
    V= inv(U);
    M_inv = V*V';
    Ez = {};
    Ez_zt = {};
        for i=1:N
          Ez{i} = M_inv * W' * X_temp(:,i);
          Ez_zt{i} = sigma_square * M_inv + Ez{i}*Ez{i}';
        end
%     M = W.'* W + sigma_square.*eye(K);
%     S = sum(sum(X_temp.^2,1));
% %     U = chol(M);
% %     V = inv(U);
% %     M_inv = V*V.';
%     T = inv(U.')*(W.'*X_temp);      
end