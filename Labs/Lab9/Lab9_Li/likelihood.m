function log_likelihood_temp = likelihood(W,X_temp,K,sigma_square,D,N)
    M = W.'* W + sigma_square.*eye(K);
    S = sum(sum(X_temp.^2,1));
    U = chol(M);
    V = inv(U);
%     M_inv = V*V.';
    T = inv(U')*(W'*X_temp);  
    Tij_square = abs(T).^2;
    sum_Tij = sum(sum(Tij_square));
    Tr_S_invM = (S - sum_Tij) / (N * sigma_square);
    log_M = 2 * sum(log(diag(U)))+(D - K) * log(sigma_square); 
    D_log_2pi = D * log(2 * pi);
    log_likelihood_temp = (-N/2) * (D_log_2pi + log_M + Tr_S_invM); 
end