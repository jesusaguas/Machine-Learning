close all;
%% Cargar los datos
datos = load('CochesTrain.txt');
ydatos = datos(:, 1);   % Precio en Euros
Xdatos = datos(:, 2:4); % Years, Km, CV
x1dibu = linspace(min(Xdatos(:,1)), max(Xdatos(:,1)), 100)'; %para dibujar

datos2 = load('CochesTest.txt');
ytest = datos2(:,1);  % Precio en Euros
Xtest = datos2(:,2:4); % Years, Km, CV
Ntest = length(ytest);

%% --------------------APARTADO 1----------------------
disp('********************APARTADO 1*******************');
%% Seleccion del grado del polinomio para la antiguedad del coche
disp('Seleccion del grado del polinomio para la antiguedad del coche:');
best_grado = kfold(10, 10, Xdatos, ydatos);
fprintf('El mejor grado es %d \n\n', best_grado);
close all;

%% --------------------APARTADO 2----------------------
disp('********************APARTADO 2*******************');
%% Seleccion del grado del polinomio para los kilometros del coche
disp('Seleccion del grado del polinomio para los kilometros del coche:');
best_grado = kfold2(10, 10, Xdatos, ydatos);
fprintf('El mejor grado es %d \n\n', best_grado);

%% Calculo el error RMSE con los datos de test con el modelo mas basico (grado 1)
disp('Calculo el error RMSE con los datos de test:');
Xexp = expandir(Xdatos, [1 1 1]);
[ Xn, mu, sig ] = normalizar( Xexp );
w = Xn \ ydatos;  % Ecuacion normal
th = desnormalizar( w, mu, sig );
Xtestexp = expandir(Xtest, [1 1 1]);
RMSEtest = RMSE(th, Xtestexp, ytest);
fprintf('El error RMSE con los datos de tests con el modelo basico (1 1 1) es %d \n', RMSEtest);

%% Calculo el error RMSE con los datos de test con el mejor modelo
Xexp = expandir(Xdatos, [5 6 6]);
[ Xn, mu, sig ] = normalizar( Xexp );
w = Xn \ ydatos;  % Ecuacion normal
th = desnormalizar( w, mu, sig );
Xtestexp = expandir(Xtest, [5 6 6]);
RMSEtest = RMSE(th, Xtestexp, ytest);
fprintf('El error RMSE con los datos de tests con el mejor modelo (5 6 6) es %d \n\n', RMSEtest);
close all;

%% --------------------APARTADO 3----------------------
disp('********************APARTADO 3*******************');
%% Seleccion de lambda para el ajuste del modelo
disp('Seleccion de lambda para el ajuste del modelo:');
best_lambda = kfold3(10, 10, Xdatos, ydatos);
fprintf('La mejor lambda es 10^%d \n\n', best_lambda);

%% Calculo el error RMSE con los datos de test, regularizando el modelo
disp('Calculo el error RMSE con los datos de test:');
Xexp = expandir(Xdatos, [10 5 5]);
[ Xn, mu, sig ] = normalizar( Xexp );
H = Xn'*Xn + 10^best_lambda * diag([0 ones(1,20)]);
w = H \ (Xn'*ydatos);
th = desnormalizar( w, mu, sig );
Xtestexp = expandir(Xtest, [10 5 5]);
RMSEtest = RMSE(th, Xtestexp, ytest);
fprintf('El error RMSE con la mejor lambda (10^-6) es %d \n', RMSEtest);