close all;
%% Cargar los datos
% Datos de entrenamiento
datos = load('PisosTrain.txt');
y = datos(:,3);  % Precio en Euros
x1 = datos(:,1); % m^2
x2 = datos(:,2); % Habitaciones
N = length(y);

% Datos de test
datostest = load('PisosTest.txt');
ytest = datostest(:,3);  % Precio en Euros
x1test = datostest(:,1); % m^2
x2test = datostest(:,2); % Habitaciones
Ntest = length(ytest);

%% Grafica de una regresion monovariable para predecir el precio de los pisos
% Datos de entrenamiento
figure;
plot(x1, y, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on; 

X = [ones(N,1) x1];
th = X \ y;  % Ecuacion normal
Xextr = [1 min(x1)  % Prediccion para los valores extremos
         1 max(x1)];
yextr = Xextr * th;  
plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de prediccion
legend('Datos Entrenamiento', 'Prediccion')

% Datos de test
figure;
plot(x1test, ytest, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on; 

Xtest = [ones(Ntest,1) x1test];
plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de prediccion
legend('Datos Test', 'Prediccion')

%% Compara los residuos obtenidos con los puntos de entrenamiento y los de test
residuotrain = mean(abs(X * th - y))
residuotest = mean(abs(Xtest * th - ytest))
[ costTrain ] = RMSE( th, X, y )
[ costTest ] = RMSE( th, Xtest, ytest )