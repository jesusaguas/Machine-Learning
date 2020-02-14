%% Practica 6.1: PCA
clc
clear all
close all

% Leer los datos originales en la variable X
load P61

% Graficar los datos originales
figure(1);
axis equal;
grid on;
hold on;
plot3(X(:,1),X(:,2),X(:,3),'b.');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');
pause;
close all


% Estandarizar los datos (solo hace falta centrarlos)
medias = mean(X);
M = size(X,1);
for line=1:M
    X(line,:)=X(line,:)-medias;
end

% Graficar los datos centrados
figure(2);
axis equal;
grid on;
hold on;
plot3(X(:,1),X(:,2),X(:,3),'b.');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');
pause;

% Calcular la matrix de covarianza muestral de los datos centrados
covMatrix = 1/(M-1)*(X'*X);

% Aplicar PCA para obtener los vectores propios y valores propios
[U, A]=eig(covMatrix);

% Ordenar los vectores y valores propios de mayor a menor valor propio
[d,ind] = sort(diag(A),'descend');
A = A(ind,ind);
U = U(:,ind);

% Graficar en color rojo cada vector propio * 3 veces la raiz de su 
% correspondiente valor propio
R=U*(3*sqrt(A));

figure(2);
axis equal;
grid on;
hold on;
plot3([0 0 0; R(1,1), R(1,2), R(1,3)], [0 0 0; R(2,1), R(2,2), R(2,3)], [0 0 0; R(3,1), R(3,2), R(3,3)], 'r-')
xlabel ('X');
ylabel ('Y');
zlabel ('Z');
pause;

% Graficar la variabilidad que se mantiene si utilizas los tres primeros
% vectores propios, los dos primeros, o solo el primer vector propio
figure(3);
axis equal;
grid on;
hold on;
diagonal=diag(A);
for i=1:size(diagonal,1)
    variabilidad(i)=sum(diagonal(1:i))/sum(diagonal);
end
plot(1:size(diagonal), variabilidad, 'r*');
set(gca, 'XTick', 1:size(diagonal))
xlabel ('Numero de vectores propios');
ylabel ('Variabilidad');
pause;

% Aplicar PCA para reducir las dimensiones de los datos y mantener al menos
% el 90% de la variabilidad (k=2)
bestK=size(diagonal,1);
for i=1:size(diagonal,1)
    variabilidad(i)=sum(diagonal(1:i))/sum(diagonal);
    if (variabilidad(i)>=0.9) && (i<bestK)
        bestK=i;
    end
end
Upca=U(:,2:3);
unitarios=[1 0; 0 1; 0 0];
Z=X*unitarios;

% Graficar aparte los datos z proyectados segun el resultado anterior
figure(4);
axis equal;
grid on;
hold on;
title(sprintf('Datos z para k = %g',bestK));
plot(Z(:,1),Z(:,2),'b.');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');
pause;

% Graficar en verde los datos reproyectados \hat{x} en la figura original
xhat=Z*unitarios';
figure(2);
axis equal;
grid on;
hold on;
plot3(xhat(:,1),xhat(:,2),xhat(:,3),'g.');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');