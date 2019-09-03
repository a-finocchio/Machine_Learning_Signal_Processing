% SCRIPT TO CREATE A DEMO FOR NEWTON'S METHOD
% FOLLOW THE COMMENTS SECTION CAREFULLY AND IMPLEMENT EACH SECTION AS
% INSTRUCTED IN THE COMMENTS
clear;

% create a function poly.m and write desired equation and return
% independent variable
f = @poly;     

% create a function poly_derivative.m and write desired equation and return
% independent variable
fder = @poly_derivative;

maxIters =  200;
tol = 1e-06;
% experiment with different values of xi
xi = 100.0;

% Initialization of relative errors, rel_errs
rel_errs = zeros(maxIters,1);
xr = xi;

% caluculate function values for each value of xlim_values using for loop
f_values=[];
xlim_values=[-abs(xr):0.1:abs(xr)];
% write from here
for i = xlim_values
    f_values = [f_values,f(i)];
end
   
% plot the xlim_values vs function values and draw x-axis and y-axis
% centered at origin
% write your code here
x = xlim_values;
y = f_values;
plot(x,y);
axis([-abs(xr) abs(xr) -abs(f(xr)) abs(f(xr))]);
ax.YAxisLocation = 'origin';  % setting y axis location to origin
ax.XAxisLocation = 'origin';  % setting x axis location to origin
line([-abs(xr), abs(xr)], [0,0]);
line([0,0], [-abs(f(xr)), abs(f(xr))]);
hold on;

% To prevent figures from displaying before the movie
set(gcf,'Visible', 'off');

% write xr as 'x0' to denote initial point. Use text function to write text on figures
% write from here
label_x0 = 'x0';
plot(xr, f(xr), 'o')
text(xr,f(xr),label_x0);

% plot tangent at xr
% write from here
hold on;
tangent = fder(xr).* (xlim_values - xr) + f(xr);
plot(x, tangent);

% draw line from xr to f(xr). Use functions text and line
[xr] = newtons_update(f,fder, xr);
% write from here
hold on;
line([xr, xr], [0, f(xr)]);

% find Newtons update and write on the same plot
% write from here
plot(xr, f(xr), 'o')
text(xr, f(xr), 'x1')

% M is the variable to hold frames of video. Use getframe function
M=[];
count=1;
M = [M, getframe(gcf)];
% write command here and store in M[count]

count=count+1;
%pause

for iter = 1:maxIters
    xrold=xr;
    % find Newtons update
    [x_r] = newtons_update(f,fder, xrold);
    
    % Relative error from xr and xrold and stopping criteria and break if
    % rel_err<tol. 
    % write from here
    rel_err_value = x_r - xrold;
    rel_err(iter) = abs(rel_err_value);
    if rel_err(iter) < tol
        break;
    end    

    % plot the xlim_values vs function values and draw x-axis and y-axis
    % centered at origin
    % write from here
    hold off;  
    x = xlim_values;
    y = f_values;
    plot(x,y)
    axis([-abs(xi) abs(xi) -abs(f(xi)) abs(f(xi))]);
    ax.YAxisLocation = 'origin';  % setting y axis location to origin
    ax.XAxisLocation = 'origin';  % setting x axis location to origin
    line([-abs(xi), abs(xi)], [0,0]);
    line([0,0], [-abs(f(xi)), abs(f(xi))]);
    hold on; 
    
    % plot tangent at xr
    % write from here
    hold on;
    tangent = fder(xr).* (xlim_values - xr) + f(xr);
    plot(xlim_values, tangent);

    % draw line from xr to f(xr)
    % write from here
    label_x_previous = sprintf('x%d', iter);
    plot(xr, f(xr), 'o');
    text(xr,f(xr),label_x_previous);
    
    % write xr as xiter_no. ex: x1, x2 for first and second iteration
    % find Newtons update and write on the same plot
    % write from here
    [xr] = newtons_update(f,fder, xr);
    hold on;
    line([xr, xr], [0, f(xr)]);
    plot(xr, f(xr), '.')
    x_label_post = sprintf('x%d', iter+1);
    text(xr, f(xr), x_label_post)
      
    hold off
    % save the current frame for the video. Store in M(count)
    % write from here
    M = [M, getframe(gcf)];  
    
    count=count+1;
    %pause
 
end
  root = xr; % root found by your algorithm

% play movie using movie commnad. 
% write from here
close all;
set(gcf,'Visible', 'on');

movie(gcf,M,1,1);

movie2gif(M, 'Solution.gif');


