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

alfa = 0.0001; 

% Normalizamos los atributos
[ Xn, mu, sig ] = normalizar( X );

% Algoritmo de descenso de gradiente
th = [0 0 0]';  % Pongo un valor cualquiera de pesos
residuo = Xn * th - y;
gradiente = Xn' * residuo;
thNew = th - alfa * gradiente;
residuoNew = Xn * thNew - y;
gradienteNew = Xn' * residuoNew;

while( (gradienteNew-gradiente) > 0.005 )
    gradiente=gradienteNew;
    th=thNew;
    thNew = th - alfa * gradiente;
    residuoNew = Xn * thNew - y;
    gradienteNew = Xn' * residuoNew;
end

% Des-Normalizamos los pesos
[ th ] = desnormalizar( thNew, mu, sig );

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
residuotrain = median(abs(X * th - y))
residuotest = median(abs(Xtest * th - ytest))
[ costTrain ] = RMSE( th, X, y )
[ costTest ] = RMSE( th, Xtest, ytest )

%% Regresion robusta con el coste de Huber
d=residuotrain;
[HuberJtrain] = costeHuber(th,X,y,d)
[HuberJtest] = costeHuber(th,Xtest,ytest,d)