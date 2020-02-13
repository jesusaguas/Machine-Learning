function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Nx2 matrix.

% Create New Figure
figure;
p = (y==1);
plot(X(p,1), X(p,2), 'r+', X(~p,1), X(~p,2), 'bo');

end
