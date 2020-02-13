clear ; clc; close all;
addpath(genpath('../minfunc'));

%% --------------------APARTADO 1----------------------
disp('********************APARTADO 1*******************');
disp('-----Regresion logistica basica-----');


%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('exam_data.txt');
y = data(:, 3);
N = length(y);
X = data(:, [1, 2]);
[ Xtest, ytest, Xtr, ytr ] = particion( 1, 5, X, y );

%% Regresion logistica basica
options.display = 'final';
options.method = 'newton';
initial_theta = [-70 0.7 0.3]';
Xtr = [ones(size(Xtr,1),1) Xtr];
theta = minFunc(@CosteLogistico, initial_theta, options, Xtr, ytr);

prediccion = 1./(1+exp(-(Xtr*theta)));
prediccion(prediccion < 0.5) = 0;
prediccion(prediccion >= 0.5) = 1;
errorTr = mean(prediccion~=ytr)

Xtest = [ones(size(Xtest,1),1) Xtest];
prediccion = 1./(1+exp(-(Xtest*theta)));
prediccion(prediccion < 0.5) = 0;
prediccion(prediccion >= 0.5) = 1;
errorTest = mean(prediccion~=ytest)

%% Dibujar la Solucion 
plotDecisionBoundary(theta, Xtr, ytr);
xlabel('Exam 1 score');
ylabel('Exam 2 score');

plotDecisionBoundary(theta, Xtest, ytest);
xlabel('Exam 1 score');
ylabel('Exam 2 score');


%% Probabilidad de ser admitido teniendo un 45 en el primer examen 
X2(1:100,:) = 45;
for i=1:100
    X2(i,2)=i;
end
X2 = [ones(size(X2,1),1) X2];
prediccion = 1./(1+exp(-(X2*theta)));

plot(1:100, prediccion, '-b');
title('Probabilidad de ser admitido con un 45 en el primer examen');
xlabel('Exam 2 score');
ylabel('Probabilidad de admitido');

%% --------------------APARTADO 2----------------------
disp('********************APARTADO 2*******************');
%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('mchip_data.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
N = length(y);
p = randperm(N); %reordena aleatoriamente los datos
X = X(p,:);
y = y(p);

[ Xtest, ytest, Xtr, ytr ] = particion( 1, 5, X, y );

%% Regresion logistica Regularizada

Xtr = mapFeature(Xtr(:,1), Xtr(:,2));
lambda = kfold(10, 4, Xtr, ytr);
options.display = 'final';
options.method = 'newton';
initial_theta = Xtr \ ytr;

%Calculamos thetas
theta = minFunc(@CosteLogReg, initial_theta, options, Xtr, ytr, 10^lambda);
thetamodelo0 = minFunc(@CosteLogReg, initial_theta, options, Xtr, ytr, 0);

%Matriz de prediccion para datos de entrenamiento con modelo bueno
prediccion = 1./(1+exp(-(Xtr*theta)));
prediccion(prediccion < 0.5) = 0;
prediccion(prediccion >= 0.5) = 1;
errorTr = mean(prediccion~=ytr)

%Matriz de prediccion para datos de test con modelo bueno
Xtest = mapFeature(Xtest(:,1), Xtest(:,2));
prediccion = 1./(1+exp(-(Xtest*theta)));
prediccion(prediccion < 0.5) = 0;
prediccion(prediccion >= 0.5) = 1;
errorTest = mean(prediccion~=ytest)

%Matriz de prediccion para datos de entrenamiento con modelo malo
prediccionmodel0 = 1./(1+exp(-(Xtr*thetamodelo0)));
prediccionmodel0(prediccionmodel0 < 0.5) = 0;
prediccionmodel0(prediccionmodel0 >= 0.5) = 1;
errorTr_modelo0 = mean(prediccionmodel0~=ytr)

%Matriz de prediccion para datos de test con modelo malo
prediccionmodel0 = 1./(1+exp(-(Xtest*thetamodelo0)));
prediccionmodel0(prediccionmodel0 < 0.5) = 0;
prediccionmodel0(prediccionmodel0 >= 0.5) = 1;
errorTest_modelo0 = mean(prediccionmodel0~=ytest)

%% Dibujar la Solucion con lambda bueno
plotDecisionBoundary(theta, Xtr, ytr);
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

%% Dibujar la Solucion con lambda 0
plotDecisionBoundary(thetamodelo0, Xtr, ytr);
title(sprintf('lambda = %g', 0))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

%% --------------------APARTADO 3----------------------
disp('********************APARTADO 3*******************');
%% Calculo Precision y Recall

%Calculo matriz de confusion
Confusion = confusionmat(ytest,prediccion);

TP=0;
FP=0;
TN=0;
FN=0;
plotconfusion(prediccion',ytest')
for i=1:size(prediccion)
    if prediccion(i)==1 && ytest(i)==1
        TP = TP+1;
    elseif prediccion(i)==1 && ytest(i)==0
        FP = FP+1;
    elseif prediccion(i)==0 && ytest(i)==0
        TN = TN+1;
    else
        FN = FN+1;
    end    
end

Precision = TP / (TP+FP)
Recall = TP / (TP+FN)

%% Cambiando el umbral podemos mejorar la precision a costa del recall, o viceversa
prediccion = 1./(1+exp(-(Xtest*theta)));
prediccion(prediccion < 0.7) = 0;
prediccion(prediccion >= 0.7) = 1;


ConfusionMOREPRECISION = confusionmat(ytest,prediccion);
TP=0;
FP=0;
TN=0;
FN=0;
for i=1:size(prediccion)
    if prediccion(i)==1 && ytest(i)==1
        TP = TP+1;
    elseif prediccion(i)==1 && ytest(i)==0
        FP = FP+1;
    elseif prediccion(i)==0 && ytest(i)==0
        TN = TN+1;
    else
        FN = FN+1;
    end    
end

Precision_newUmbral = TP / (TP+FP)
Recall_newUmbral = TP / (TP+FN)
