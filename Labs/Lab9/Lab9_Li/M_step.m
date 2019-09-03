function [sigma,W] = M_step(X_temp,Ez,Ez_zt,N,sigma_square,V,D)
    sum_xn_Ezn = 0;
    sum_Ezn_Eznt = 0;
    sum_sigma_temp = 0;
    for i = 1:N
        sum_xn_Ezn = sum_xn_Ezn + X_temp(:,i) * Ez{i}';
        sum_Ezn_Eznt = sum_Ezn_Eznt + sigma_square * V * V'+ Ez{i} * Ez{i}';        
    end
    W_new = sum_xn_Ezn * inv(sum_Ezn_Eznt);    
    for i =1:N
        xn2 = norm(X_temp(:,i))^2;
        EWxn = 2*Ez{i}'* W_new' * X_temp(:,i);
        TrEWW = sum(diag(Ez_zt{i} * W_new' * W_new));
        sum_sigma_temp = sum_sigma_temp + (xn2 - EWxn + TrEWW);
    end   
    sigma_sq_new = (1/(N*D))*sum_sigma_temp;   
    W = W_new;
    sigma = sigma_sq_new;
end