function  dist = backtrack(D,idx)
    [row,col] = size(D);
    
    count = 0;
    while idx(row,col)~=0
        count = count + 1;
        if idx(row,col) == 1
            row = row - 1;
            col = col - 1;
            
        elseif idx(row,col) == 2
            row = row - 1;
            
        elseif idx(row,col) == 3
            col = col - 1;
    
        end
    end    
    dist = D(row,col) / count;

end