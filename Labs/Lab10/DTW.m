function [D,DTW_idx] = DTW(S)
    [row,col] = size(S);
    D=zeros(row,col);
    DTW_idx = zeros(row,col);
    
    % first col
    sum_temp = 0;
    for i_row = 1:row   
        sum_temp = sum_temp + abs(S(i_row,1));
        D(i_row,1) = sum_temp;
        DTW_idx(i_row,1) = 2;
    end 
    
    %last row
    sum_temp = 0;
    for i_col = 1:col   
        sum_temp = sum_temp + abs(S(1,i_col));
        D(1,i_col) = sum_temp;
        DTW_idx(1,i_col) = 3;
    end
    
    DTW_idx(1,1) = 0; 
    %other row & col
    for i=2:row
        for j=2:col
            [min_value, idx] = min([D(i-1,j-1), D(i-1,j), D(i,j-1)]);
            D(i,j) = S(i,j) + min_value;
            DTW_idx(i,j) = idx;
        end   
    end
            
end
