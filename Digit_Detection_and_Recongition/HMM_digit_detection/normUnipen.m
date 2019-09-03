function [DataNorm] = normUnipen( Data )

[~,n] = size(Data);
DataNorm = {};

    for i = 1: n
        temp_data = Data{i};
        temp_data_plot = Data{i};
        temp_data(temp_data == -1) =1;
        
        row1 = temp_data(1,:);
        row2 = temp_data(2,:);     
        row1(row1==1)=[];
        row2(row2==1)=[];
        
        row11 = zscore(row1);
        row22 = zscore(row2);
        
        DataNorm{i} = [row11;row22];

    end
    
end


