% load images 
% images size is 20x20. 
clear
close all

load('MNISTdata2.mat');
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);

nrows=20;
ncols=20;

nimages = size(X,1);

% Show the images
%{
for I=1:40:nimages
    imshow(reshape(X(I,:),nrows,ncols))
    pause(0.1)
end
%}

%% Perform PCA over all numbers

% TRAIN
medias = mean(X);
M = size(X,1);
for line=1:M
    Xnorm(line,:)=X(line,:)-medias;
end

covMatrix= 1/(M-1)*(Xnorm'*Xnorm);
[U, A]=eig(covMatrix);
[d,ind] = sort(diag(A),'descend');
A = A(ind,ind);
U = U(:,ind);
Upca=U(:,1:2);
z=Xnorm*Upca;

% TEST
M = size(Xtest,1);
for line=1:M
    Xtest_norm(line,:)=Xtest(line,:)-medias;
end
ztest=Xtest_norm*Upca;

% Muestra las dos componentes principales
figure(100)
clf, hold on
title('PCA con todas las clases');
xlabel ('Atributo 1');
ylabel ('Atributo 2');
plotwithcolor(z, y);

%% Clasificador Bayes Completo con todos los datos
clases=1:10;
lambda = kfold(10, 5, clases, z, y, 0);
modelo = entrenarGaussianas(z, y, clases, 0, 10^lambda);
yhat = clasificacionBayesiana(modelo, ztest, clases);

%Calculo matriz de confusion
Confusion_BayesCompleto = confusionmat(ytest',yhat);

%Calculo Precision y Recall de cada digito
Sumcolumnas=sum(Confusion_BayesCompleto)';
Sumfilas=sum(Confusion_BayesCompleto,2);
Precision_BayesCompleto = diag(Confusion_BayesCompleto) / Sumcolumnas;
[fila,columna]=find(Precision_BayesCompleto>0);
Precision_BayesCompleto=Precision_BayesCompleto(:,columna(1))
Recall_BayesCompleto = diag(Confusion_BayesCompleto) / Sumfilas;
Recall_BayesCompleto=Recall_BayesCompleto(:,1)
pause;





%% Clasificador Bayes Completo con 1-10 & 2-8
fprintf('------------------------------------------------\n');
[fila,columna]=find(y==3 | y==4 | y==5 | y==6 | y==7 | y==9);
z(fila,:)=[];
y2=y;
y2(fila,:)=[];

[fila,columna]=find(ytest==3 | ytest==4 | ytest==5 | ytest==6 | ytest==7 | ytest==9);
ztest(fila,:)=[];
y2test=ytest;
y2test(fila,:)=[];

% Muestra las dos componentes principales
figure(100)
clf, hold on
title('Aislamos las clases 1, 2, 8, 10');
xlabel ('Atributo 1');
ylabel ('Atributo 2');
plotwithcolor(z, y2);

clases=[1 2 8 10];
lambda = kfold(10, 5, clases, z, y2, 0);
modelo = entrenarGaussianas(z, y2, clases, 0, 10^lambda);
yhat = clasificacionBayesiana(modelo, ztest, clases);
yhat(yhat==3)=8;
yhat(yhat==4)=10;

%Calculo matriz de confusion
Confusion2_BayesCompleto = confusionmat(y2test',yhat);

%Calculo Precision y Recall de cada digito
Sumcolumnas=sum(Confusion2_BayesCompleto)';
Sumfilas=sum(Confusion2_BayesCompleto,2);
Precision_BayesCompleto = diag(Confusion2_BayesCompleto) / Sumcolumnas;
[fila,columna]=find(Precision_BayesCompleto>0);
Precision_BayesCompleto=Precision_BayesCompleto(:,columna(1))
Recall_BayesCompleto = diag(Confusion2_BayesCompleto) / Sumfilas;
Recall_BayesCompleto=Recall_BayesCompleto(:,1)
pause;






%% Perform PCA over numbers 1-10 & 2-8
[fila,columna]=find(y==3 | y==4 | y==5 | y==6 | y==7 | y==9);
X(fila,:)=[];
y(fila,:)=[];

[fila,columna]=find(ytest==3 | ytest==4 | ytest==5 | ytest==6 | ytest==7 | ytest==9);
Xtest(fila,:)=[];
ytest(fila,:)=[];

% TRAIN
medias = mean(X);
M = size(X,1);
for line=1:M
    Xnorm2(line,:)=X(line,:)-medias;
end

covMatrix= 1/(M-1)*(Xnorm2'*Xnorm2);
[U, A]=eig(covMatrix);
[d,ind] = sort(diag(A),'descend');
A = A(ind,ind);
U = U(:,ind);
Upca=U(:,1:2);
z=Xnorm2*Upca;

% TEST
M = size(Xtest,1);
for line=1:M
    Xtest_norm2(line,:)=Xtest(line,:)-medias;
end
ztest=Xtest_norm2*Upca;

% Muestra las dos componentes principales
figure(100)
clf, hold on
title('PCA sobre las clases 1, 2, 8, 10');
xlabel ('Atributo 1');
ylabel ('Atributo 2');
plotwithcolor(z, y);

%% Clasificador Bayes Completo con 1-10 & 2-8
fprintf('------------------------------------------------\n');
clases=[1 2 8 10];
lambda = kfold(10, 5, clases, z, y, 0);
modelo = entrenarGaussianas(z, y, clases, 0, 10^lambda);
yhat = clasificacionBayesiana(modelo, ztest, clases);
yhat(yhat==3)=8;
yhat(yhat==4)=10;

%Calculo matriz de confusion
Confusion3_BayesCompleto = confusionmat(ytest',yhat);

%Calculo Precision y Recall de cada digito
Sumcolumnas=sum(Confusion3_BayesCompleto)';
Sumfilas=sum(Confusion3_BayesCompleto,2);
Precision_BayesCompleto = diag(Confusion3_BayesCompleto) / Sumcolumnas;
[fila,columna]=find(Precision_BayesCompleto>0);
Precision_BayesCompleto=Precision_BayesCompleto(:,columna(1))
Recall_BayesCompleto = diag(Confusion3_BayesCompleto) / Sumfilas;
Recall_BayesCompleto=Recall_BayesCompleto(:,1)