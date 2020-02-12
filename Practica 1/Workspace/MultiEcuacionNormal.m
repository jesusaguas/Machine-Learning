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

%% Dibujo de un Ajuste con dos Variables
X = [ones(N,1) x1 x2];
Xtest = [ones(Ntest,1) x1test x2test];
th = X \ y;  % Ecuacion normal
yest = X * th;

% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');

% Generar una reticula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la reticula
Xg = [ones(size(x1g)), x1g, x2g];
yg = Xg * th;

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');
%% Compara los residuos obtenidos con los puntos de entrenamiento y los de test
residuotrain = mean(abs(X * th - y))
residuotest = mean(abs(Xtest * th - ytest))
[ costTrain ] = RMSE( th, X, y )
[ costTest ] = RMSE( th, Xtest, ytest )

%% Coste de comprar un piso de 100m cuadrados entre los dos modelos
% Coste con el modelo monovariable
Xmono = [ones(N,1) x1];
thmono = Xmono \ y;
costePisoMono = 100 * thmono(2,:)

% Coste con el modelo multivariable
costePiso2habMulti = [1,100,2] * th
costePiso3habMulti = [1,100,3] * th
costePiso4habMulti = [1,100,4] * th
costePiso5habMulti = [1,100,5] * th
