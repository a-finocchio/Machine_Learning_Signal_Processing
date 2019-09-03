function [LDAresults] = distancesLab8(TrainingLDA,TestingLDA)

error = zeros(40,360);   
for i = 1:40
        difference = TrainingLDA(:,:)-repmat(TestingLDA(:,i),1,360);
        error_temp = sqrt(sum(difference.^2,1));
        error(i,:) = error_temp;
end  

closest = zeros(40,1);

for i = 1:40
    [val,loc] = min(error(i,:)) ;
    cel = ceil(loc/9);
    closest(i,:) = cel;
end   


ground_truth = 1:40;

% for i = 1:40
%     lower_end = (i-1)*9;
%     upper_end = i*9;
%     index_result = closest(i,:);
%     
%     if index_result <= upper_end
%         if index_result > lower_end        
%             results_plot(i) = true;
%         else    
%             results_plot(i) = false;
%         end
%     else
%         results_plot(i) = false;
%     end 
% 
% end    

LDAresults = closest == ground_truth';

figure(10);
imagesc(LDAresults), axis image, colormap gray, title("PredictedImage Accuracy");