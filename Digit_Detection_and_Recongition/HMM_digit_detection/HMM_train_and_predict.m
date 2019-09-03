function [predict, accuracy] = HMM_train_and_predict(traiDataK,testDataK, train_lbl, test_lbl, iter,tol, state)
    % train
    ESTTR={};
    ESTEMIT={};
    for i = 1:10
        index = train_lbl == i-1;
        seq = traiDataK(index);

        TRGUESS = rand(state,state);
        TRGUESS =  TRGUESS ./ sum(TRGUESS,2);

        EMITGUESS = rand(state,256);
        EMITGUESS = EMITGUESS ./ sum(EMITGUESS,2);

        [ESTTR{i},ESTEMIT{i}] = hmmtrain(seq,TRGUESS,EMITGUESS,'MAXITERATIONS',iter,'Tolerance',tol);  
    end

    % predict
    n_test = size(testDataK,2);
    predict = zeros(n_test,10);
    for i = 1:n_test
        seq = testDataK{i};
        for j = 1:10
            [~, predict(i,j)] = hmmdecode(seq,ESTTR{j},ESTEMIT{j});
        end
    end
    
    [~,result] =  max(predict,[],2);
    result = result -1;
    error = length(result(result~=test_lbl'));
    accuracy = 1- error/n_test;

end