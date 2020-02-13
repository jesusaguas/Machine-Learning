%% Based on exercise 2 of Machine Learning Online Class by Andrew Ng 

clear ; close all;

%% Load and Plot Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('mchip_data.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
N = length(y);
p = randperm(N); %reordena aleatoriamente los datos
X = X(p,:);
y = y(p);

plotData(X, y);
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')

%% Calcula una Solucion Absurda
X = mapFeature(X(:,1), X(:,2));
lambda = 0;
theta = X \ y; %MAL, hay que hacer regresion logistica

%% Dibujar la Solucion
plotDecisionBoundary(theta, X, y);
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')


