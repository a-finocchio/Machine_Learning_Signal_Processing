%% Exercise 1: Image manipulation
Cameraman_image = imread('cameraman.tif');
subplot(1,4,1), imshow(Cameraman_image);

fade_factor=0.5;
Fade_image = Cameraman_image * fade_factor;
subplot(1,4,2), imshow(Fade_image);

N1 = 100;
First_part_image = Cameraman_image(1:N1,1:N1);
subplot(1,4,3), imshow(First_part_image);

N_size = size(Cameraman_image);
N2_1 = N_size(1) - 100;
N2_2 = N_size(2) - 100;
last_part_image = Cameraman_image(N2_1:N_size(1),N2_2:N_size(2));
subplot(1,4,4), imshow(last_part_image);


%% Exercise 2: Quantization
[p, fs] = audioread('filename1.wav');
audiowrite('filename2.wav', p, fs, 'BitsPerSample', 8);
p1 = audioread('filename2.wav');
p2 = [p1 > 0];

disp('click for sounds');
w1 = waitforbuttonpress;
if w1 == 0
    disp('16 bits');
    sound(p, fs);
else
    disp('Key press');
end

w2 = waitforbuttonpress;
if w2 == 0
    clear sound
    disp('8 bits');
    sound(p1, fs);
else
    disp('Key press');
end

w3 = waitforbuttonpress;
if w3 == 0
    clear sound
    disp('1 bit');
    sound(double(p2), fs);    
else
    disp('Key press');
end

w4 = waitforbuttonpress;
if w4 == 0
    clear sound
    disp('sound ends');  
else
    disp('Key press');
end


%% Exercise 3: Aliasing Effect
Fs = 8000;
dt = 1/Fs;
t = 0: dt: 0.05;
Fc = 100;
x = cos(Fc*t);
figure;
subplot(2,2,1);
plot(t,x);
title('100Hz');

Fs = 8000;
dt = 1/Fs;
t = 0: dt: 0.05;
Fc = 600;
x = cos(Fc*t);
subplot(2,2,2);
plot(t,x);
title('600Hz');

Fs = 500;
dt = 1/Fs;
t = 0: dt: 0.05;
Fc = 100;
x = cos(Fc*t);
subplot(2,2,3);
plot(t,x);
title('100Hz @ 500Hz sample rate');

Fs = 500;
dt = 1/Fs;
t = 0: dt: 0.05;
Fc = 600;
x = cos(Fc*t);
subplot(2,2,4);
plot(t,x);
title('600Hz @ 500Hz sample rate');

disp(' ');
disp(' ');
disp(' ');
disp('Different Hertz would have different width(period) of the cosine wave,');
disp('the width(period) for 600Hz is small, but large for 100Hz.');
disp(' ');
disp('For the sample rates, 100Hz @ 500Hz is good, since 500Hz is larger than 100Hz. ');
disp('But for 600Hz @ 500Hz, some details are lost, casuing the aliasing effect. ');
disp('This is because sample frequency is smaller than waves frequency. ');
