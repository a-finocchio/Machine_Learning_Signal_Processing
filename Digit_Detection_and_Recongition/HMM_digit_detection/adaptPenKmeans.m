function [traiDataK, testDataK] = adaptPenKmeans(traiData, testData)
%your code goes here. Take into account that the output variables will not 
%include coordinates anymore but 'emissions'.

[~,n] = size(traiData);

traiData_1 = [];
traiData_record = [];
for i = 1:n
    traiData_1 = [traiData_1, traiData{i}];
    traiData_record = [traiData_record, size(traiData{i},2)];
end

[~,nn] = size(testData);
testData_1 = [];
testData_record = [];
for i = 1:nn
    testData_1 = [testData_1, testData{i}];
    testData_record = [testData_record, size(testData{i},2)];
end

[idx, center] = kmeans(traiData_1',256,'MaxIter',300);

codebook = KDTreeSearcher(center);
idxx = knnsearch(codebook,testData_1');

train_encoded={1,n};
sum = 1;
for i = 1:n  
    temp_1= traiData_record(i);
    train_encoded{1,i} = idx(sum:sum+temp_1-1)';
    sum = sum+temp_1;
end   

test_encoded={1,nn};
sum = 1;
for i = 1:nn
    temp_1= testData_record(i);
    test_encoded{1,i} = idxx(sum:sum+temp_1-1)';
    sum = sum+temp_1;
end   

traiDataK = train_encoded;
testDataK = test_encoded;

end