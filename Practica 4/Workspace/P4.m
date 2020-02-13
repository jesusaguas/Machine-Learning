%% Practica 4 
% Based on exercise 3 of Machine Learning Online Class by Andrew Ng 
%

clear ; close all;
addpath(genpath('../minfunc'));

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);

%% Calculamos el modelo multi-clase
options.display = 'final';
options.method = 'lbfgs';
initial_theta = zeros(size(X',1),1);

%Calculamos thetas para modelo 1
y1 = y;
y1(y1 ~= 1) = 0;
lambda = kfold(10, 5, X, y1);
theta1 = minFunc(@CosteLogReg, initial_theta, options, X, y1, 10^lambda);

%Calculamos thetas para modelo 2
y2 = y;
y2(y2 ~= 2) = 0;
y2(y2 == 2) = 1;
lambda = kfold(10, 5, X, y2);
theta2 = minFunc(@CosteLogReg, initial_theta, options, X, y2, 10^lambda);

%Calculamos thetas para modelo 3
y3 = y;
y3(y3 ~= 3) = 0;
y3(y3 == 3) = 1;
lambda = kfold(10, 5, X, y3);
theta3 = minFunc(@CosteLogReg, initial_theta, options, X, y3, 10^lambda);

%Calculamos thetas para modelo 4
y4 = y;
y4(y4 ~= 4) = 0;
y4(y4 == 4) = 1;
lambda = kfold(10, 5, X, y4);
theta4 = minFunc(@CosteLogReg, initial_theta, options, X, y4, 10^lambda);

%Calculamos thetas para modelo 5
y5 = y;
y5(y5 ~= 5) = 0;
y5(y5 == 5) = 1;
lambda = kfold(10, 5, X, y5);
theta5 = minFunc(@CosteLogReg, initial_theta, options, X, y5, 10^lambda);

%Calculamos thetas para modelo 6
y6 = y;
y6(y6 ~= 6) = 0;
y6(y6 == 6) = 1;
lambda = kfold(10, 5, X, y6);
theta6 = minFunc(@CosteLogReg, initial_theta, options, X, y6, 10^lambda);

%Calculamos thetas para modelo 7
y7 = y;
y7(y7 ~= 7) = 0;
y7(y7 == 7) = 1;
lambda = kfold(10, 5, X, y7);
theta7 = minFunc(@CosteLogReg, initial_theta, options, X, y7, 10^lambda);

%Calculamos thetas para modelo 8
y8 = y;
y8(y8 ~= 8) = 0;
y8(y8 == 8) = 1;
lambda = kfold(10, 5, X, y8);
theta8 = minFunc(@CosteLogReg, initial_theta, options, X, y8, 10^lambda);

%Calculamos thetas para modelo 9
y9 = y;
y9(y9 ~= 9) = 0;
y9(y9 == 9) = 1;
lambda = kfold(10, 5, X, y9);
theta9 = minFunc(@CosteLogReg, initial_theta, options, X, y9, 10^lambda);

%Calculamos thetas para modelo 10
y10 = y;
y10(y10 ~= 10) = 0;
y10(y10 == 10) = 1;
lambda = kfold(10, 5, X, y10);
theta10 = minFunc(@CosteLogReg, initial_theta, options, X, y10, 10^lambda);


% Juntamos thetas
theta = [theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9,theta10];

%Matriz de prediccion
h = 1./(1+exp(-(Xtest*theta)));
[A,prediccion] = max(h');

% Comparacion prediccion-ytest
errorTest = mean((prediccion')~=ytest);
fprintf('\nError medio de prediccion(test): %4.2f \n', errorTest);

%% Calculo Matriz de Confusi?n, Precision y Recall

%Calculo matriz de confusion
Confusion = confusionmat(ytest',prediccion);

%Calculo Precision y Recall de cada digito
Sumcolumnas=sum(Confusion);
Sumfilas=sum(Confusion,2);
Precision = diag(Confusion) / Sumcolumnas';
Precision=Precision(:,1)
Recall = diag(Confusion) / Sumfilas;
Recall=Recall(:,1)

% Muestro matriz de confusion con la funcion verConfusiones
verConfusiones(Xtest, ytest', prediccion);