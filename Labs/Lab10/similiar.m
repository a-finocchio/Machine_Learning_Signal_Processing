function S = similiar(T,R)
    [T_row,~] = size(T);
    [R_row,~] = size(R);
    S = zeros(T_row,R_row);
    for i = 1:T_row
        for j = 1:R_row
            S(i,j) = sqrt(sum((T(i,:)-R(j,:)).^2));
        end
    end
end