%% LAB #3
%please run section by section
disp('TO VIEW IMAGE PART BY PART, YOU CAN RUN THE CODE SECTION BY SECTION')

%% exercise 1 part one
clear
figure(1)
disp('===== exercise 1 part one =====');
%a
Lena_noisy = imread('lena_noisy.bmp');
%b
Lena_noisy_median = medfilt2(Lena_noisy,[5,5]);
%c
Lena_noisy_Gaussian1 = imgaussfilt(Lena_noisy,1,'FilterSize',[5,5]);
%d
subplot(2,2,1),imshow(Lena_noisy_median),title('Median');
subplot(2,2,2),imshow(Lena_noisy_Gaussian1),title('Gaussian sigma = 1  [5,5]');
disp('%d. Median filter makes the image smoothier than Gaussian s');
%e
%Lena_noisy_Gaussian2 = imgaussfilt(Lena_noisy,2,'FilterSize',[5,5]); %to test sigma's effect
Lena_noisy_Gaussian2 = imgaussfilt(Lena_noisy,2,'FilterSize',[9,9]);
%f
Lena_noisy_Gaussian3 = imgaussfilt(Lena_noisy,2,'FilterSize',[15,15]);
%g
subplot(2,2,3),imshow(Lena_noisy_Gaussian2),title('Gaussian sigma = 2  [9,9]');
subplot(2,2,4),imshow(Lena_noisy_Gaussian3),title('Gaussian sigma = 2  [15,15]');
%h
disp('%h. the higher the kernel size, the smoothier the result, and higher sigma can also casue smoothier result');


%% exercise 1 part two I
clear
figure(2)
disp('===== exercise 1 part two I =====');
%FFT and Histogram
%i
Lena = imread('lena.bmp');
%ii
Lena_FFT = fft2(Lena);
%iii
Lena_Gaussian = imgaussfilt(Lena,2,'FilterSize',[5,5]);
Lena_Gaussian_FFT = fft2(Lena_Gaussian);
%iv
Lena_Median = medfilt2(Lena,[5,5]);
Lena_Median_FFT = fft2(Lena_Median);
%v
Lena_FFT1  = fftshift(Lena_FFT); % Center FFT
Lena_Gaussian_FFT1 = fftshift(Lena_Gaussian_FFT);
Lena_Median_FFT1 = fftshift(Lena_Median_FFT);
Lena_FFT1 = abs(Lena_FFT1);% Get the magnitude
Lena_Gaussian_FFT1 = abs(Lena_Gaussian_FFT1);
Lena_Median_FFT1 = abs(Lena_Median_FFT1);
Lena_FFT1  = log(Lena_FFT1 +eps); % Use log, for perceptual scaling, and +eps since log(0) is undefined
Lena_Gaussian_FFT1 = log(Lena_Gaussian_FFT1 +eps);
Lena_Median_FFT1 = log(Lena_Median_FFT1 +eps);
Lena_FFT1 = mat2gray(Lena_FFT1); % Use mat2gray to scale the image between 0 and 1
Lena_Gaussian_FFT1 = mat2gray(Lena_Gaussian_FFT1);
Lena_Median_FFT1 = mat2gray(Lena_Median_FFT1);
subplot(3,3,1),imshow(Lena_FFT1,[]),title('Original FFT'); % Display the result
subplot(3,3,4),imshow(Lena_Gaussian_FFT1,[]),title('Gaussian FFT');
subplot(3,3,7),imshow(Lena_Median_FFT1,[]),title('Median FFT');
disp('%v. the results of three looks similiar, because we can see the cross line and the cener point. But the Gaussian_FFT and Median_FFT looks sharper than Original_FFT.');
%vi
subplot(3,3,2),histogram(Lena),title('Original Hist');
subplot(3,3,5),histogram(Lena_Gaussian),title('Gaussian Hist');
subplot(3,3,8),histogram(Lena_Median),title('Median Hist');
subplot(3,3,3),histogram(Lena_FFT1),title('Original FFT Hist');
subplot(3,3,6),histogram(Lena_Gaussian_FFT1),title('Gaussian FFT Hist');
subplot(3,3,9),histogram(Lena_Median_FFT1),title('Median FFT Hist');
disp('%vi. from the histogram, the shape of them are similiar. But when look deeper on the peaks, we find that the peaks in hist(Gaussian_FFT) and hist(Median_FFT) looks more clear than the peaks in (Original_FFT).');
disp('%vi. which means from the histogram we can find out that the Gaussian_FFT and Median_FFT are sharper than Original_FFT');


%% exercise 1 part two II
clear
figure(3)
disp('===== exercise 1 part two II =====');
%FFT Algebra
%i
stp1 = imread('stp1.gif');
stp1 = mat2gray(uint8(stp1));
stp2 = imread('stp2.gif');
stp2 = mat2gray(uint8(stp2));
stp_add = stp1 + stp2;
%ii
stp1_FFT = fft2(stp1);
stp2_FFT = fft2(stp2);
stp1_FFT1 = fftshift(stp1_FFT); % Center FFT
stp2_FFT1 = fftshift(stp2_FFT);
stp1_FFT1 = abs(stp1_FFT1); % Get the magnitude
stp2_FFT1 = abs(stp2_FFT1);
stp1_FFT1 = log(stp1_FFT1+eps); % Use log, for perceptual scaling, and +eps since log(0) is undefined
stp2_FFT1 = log(stp2_FFT1+eps);
stp1_FFT1 = mat2gray(stp1_FFT1); % Use mat2gray to scale the image between 0 and 1
stp2_FFT1 = mat2gray(stp2_FFT1);
subplot(3,2,1),imshow(stp1_FFT1,[]),title('FFT(stp1)'); % Display the result
subplot(3,2,2),imshow(stp2_FFT1,[]),title('FFT(stp2)');
disp('%ii. the two images are diiferent from each other, but each image seems to be symmetric and also very different from their original ones');
%iii
FT = stp1_FFT + stp2_FFT;
%iv
FT_inverse = ifft2(FT);
subplot(3,2,3),imshow(stp_add,[]),title('stp1 + stp2');
subplot(3,2,4),imshow(FT_inverse,[]),title('inverse[FFT(stp1)+FFT(stp2)]');
disp('%iv. the two images are the same, becasue FFT(x+y)=FFT(x)+FFT(y)');
%v
STP12 = stp1 .* stp2;
%vi
STP12_FFT = fft2(STP12);
%vii
STP_FFT_M = stp1_FFT * stp2_FFT;
STPii_FFT_inverse = ifft2(STP_FFT_M); 
subplot(3,2,5),imshow(STPii_FFT_inverse,[]),title('inverse[FFT(stp1)*FFT(stp2)]'); % Display the result
subplot(3,2,6),imshow(STP12,[]),title('original multiple stp1*stp2');
disp('%iv. the two images are different, becasue FFT(x*y) != FFT(x) * FFT(y)');
%Bounus
% BONUS POINTS – Can you find the standard deviation of the Gaussian kernel that “Lena_noisy” has been smoothened with?


%% exercise 2
clear
figure(4)
disp('===== exercise 2 =====');
%i 
Cameraman = imread('cameraman.tif');
subplot(2,5,1),imshow(Cameraman),title('Cameraman Original');
%ii
w = dct2(Cameraman);
%iii
Energy = log(abs(w));
subplot(2,5,2), imagesc(Energy),title('w Energy');
%iv
nPrincipal = 128;
A = zeros(nPrincipal,nPrincipal);
B = w(1:nPrincipal,1:nPrincipal);
w2 = [B,A;A,A];
%v
Energy2 = log(abs(w2));
subplot(2,5,3), imagesc(Energy2),title('w2 Energy');
%vi
w_inverse = idct2(w);
%depends on what function you want
subplot(2,5,4), imshow(w_inverse,[]),title('w inverse (imshow)');
subplot(2,5,9), imagesc(w_inverse),title('w inverse (imagesc)');
%vii
w2_inverse = idct2(w2);
%depends on what function you want
subplot(2,5,5), imshow(w2_inverse,[]),title('w2 inverse (imshow)');
subplot(2,5,10), imagesc(w2_inverse),title('w2 inverse (imagesc)');
disp('%vii. from the result we can find that even w2 is crop from w1 and lost a lot of details, but it can still reconstruct a similiar image with the original one.');
disp('%vii. This is because the important details is stored in the top left corner of the w matrix, which means we can send smaller data to represent a same image');







