%% Practica 5 
% Based on exercise 3 of Machine Learning Online Class by Andrew Ng 
%

clear ; close all;

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);

%Modelos Gaussianos regularizados
%lambda=-3;
lambda = kfold(10, 5, X, y, 1);
modeloNaive = entrenarGaussianas( X, y, 10, 1, 10^lambda);
yhatNaive = clasificacionBayesiana(modeloNaive, Xtest);

lambda = kfold(10, 5, X, y, 0);
modelo = entrenarGaussianas( X, y, 10, 0, 10^lambda);
yhat = clasificacionBayesiana(modelo, Xtest);


% Comparacion prediccion-ytest
errorTestNaive = mean(yhatNaive'~=ytest);
fprintf('\nError medio de prediccion(test) con Bayes Ingenuo: %4.2f \n', errorTestNaive);

errorTest = mean(yhat'~=ytest);
fprintf('\nError medio de prediccion(test) con Bayes Completo: %4.2f \n', errorTest);

%% Calculo Matriz de Confusion, Precision y Recall Bayes Ingenuo

%Calculo matriz de confusion
Confusion_BayesIngenuo = confusionmat(ytest',yhatNaive);

%Calculo Precision y Recall de cada digito
Sumcolumnas=sum(Confusion_BayesIngenuo);
Sumfilas=sum(Confusion_BayesIngenuo,2);
Precision_BayesIngenuo = diag(Confusion_BayesIngenuo) / Sumcolumnas';
Precision_BayesIngenuo=Precision_BayesIngenuo(:,1)
Recall_BayesIngenuo = diag(Confusion_BayesIngenuo) / Sumfilas;
Recall_BayesIngenuo=Recall_BayesIngenuo(:,1)

% Muestro matriz de confusion con la funcion verConfusiones
verConfusiones(Xtest, ytest', yhatNaive);

%% Calculo Matriz de Confusion, Precision y Recall Bayes Completo

%Calculo matriz de confusion
Confusion_BayesCompleto = confusionmat(ytest',yhat);

%Calculo Precision y Recall de cada digito
Sumcolumnas=sum(Confusion_BayesCompleto);
Sumfilas=sum(Confusion_BayesCompleto,2);
Precision_BayesCompleto = diag(Confusion_BayesCompleto) / Sumcolumnas';
Precision_BayesCompleto=Precision_BayesCompleto(:,2)
Recall_BayesCompleto = diag(Confusion_BayesCompleto) / Sumfilas;
Recall_BayesCompleto=Recall_BayesCompleto(:,1)

% Muestro matriz de confusion con la funcion verConfusiones
verConfusiones(Xtest, ytest', yhat);