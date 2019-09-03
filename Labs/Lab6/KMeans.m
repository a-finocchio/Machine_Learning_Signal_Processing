%  K-Means function to generate K clusters of an input image
%  Author Name:

%% function
function [C, segmented_image] = KMeans(X,K,maxIter)
%% 1. Image vectorization based on RGB components
img(:,:,:) = double(imread(X));
R = reshape(img(:,:,1),[],1);
G = reshape(img(:,:,2),[],1);
B = reshape(img(:,:,3),[],1);
img_vec = [R,G,B];
[row_img, col_img,rgb] = size(img);
pix_num = row_img * col_img;

%% 2. Intial RGB Centroid Calculation 
RGB_centroid = mean(img_vec);


%% 3. Randomly initializing K centroids (select those centroids from the actual points)
k_centroid = [];
for i=1:K
    k_centroid_idx = randi(pix_num);
    k_centroid_new = img_vec(k_centroid_idx,:) + 0.1*RGB_centroid;
    k_centroid = [k_centroid; k_centroid_new];
end    

% k_centroid = [];
% for i=1:K
%     k_centroid_new = -1 + (1+1)*rand(1,3);
%     k_centroid_new = 200*k_centroid_new + RGB_centroid;
%     k_centroid = [k_centroid; k_centroid_new];
% end    


%% 4. Assign data points to K clusters using the following distance - dist = norm(C-X,1)
[row, col] = size(img_vec);
dist_group =[];
for i=1:K
   k_centroid_k_matrix = repmat( k_centroid(i,1:3),row,1);
   difference = img_vec(:,1:3) - k_centroid_k_matrix;    
   dist = sqrt(sum(difference.^2,2));
   dist_group =[dist_group, dist]; 
end
[minimum,index] = min(dist_group,[],2);
kmean_vec = [img_vec(:,1:3),index];

    
%% 5. Re-computing K centroids
k_centroid_new_new = [];
for i = 1:K   
    cluster_K = (kmean_vec(:,4)== i);
    new_centroid = mean(img_vec(cluster_K,1:3));
    if new_centroid == [0,0,0]
        new_centroid = rand(1,3);
    end    
    k_centroid_new_new = [k_centroid_new_new; new_centroid]; 
end   
k_centroid = k_centroid_new_new;
    

%% Reiterate through steps 4 and 5 until maxIter reached. Set maxIter = 100
for itertaion = 1:maxIter-1
   dist_group =[];
   for i=1:K
     k_centroid_k_matrix = repmat( k_centroid(i,1:3),row,1);
     difference = img_vec(:,1:3) - k_centroid_k_matrix;    
     dist = sqrt(sum(difference.^2,2));
     dist_group =[dist_group, dist]; 
   end
   [minimum,index] = min(dist_group,[],2);
   kmean_vec = [img_vec(:,1:3),index];
   k_centroid_new_new = [];
   for l = 1:K   
     cluster_K = (kmean_vec(:,4)== l);
     new_centroid = mean(img_vec(cluster_K,1:3));
     if new_centroid == [0,0,0]
      new_centroid = rand(1,3);
     end   
     k_centroid_new_new = [k_centroid_new_new; new_centroid];    
   end   
   k_centroid = k_centroid_new_new; 
end


%% Return K centroid coordinates and segmented Image
new_img_vec = kmean_vec;
for i = 1:K  
     A = (new_img_vec(:,4)== i);   
     new_img_vec(A,1:3)= repmat(k_centroid_new_new(i,:),sum(A(:)),1);      
end

C = k_centroid_new_new;
segmented_image = uint8(reshape(new_img_vec(:,1:3),row_img,col_img,3));

end

