%% Practica 6.2: PCA 
clc
clear all
close all

% Leer la imagen
I = imread('Jesus17.jpg');

% Convertirla a blanco y negro
BW = rgb2gray(I);

% Convertir los datos a double
X=im2double(BW);

% graficar la imagen
figure(1);
title('Imagen original');
colormap(gray);
imshow(X);
axis off;
pause

% Aplicar SVD
[U,S,V] = svd(X);

% Graficar las primeras 5 componentes
for k = 1:5
    figure(2);
    Xhat=U(:,k)*S(k,k)*V(:,k)';
    imshow(Xhat);
    colormap(gray);
    axis off;
    title(sprintf('Componente %g', k));
    pause
end

% Graficar la reconstruccion con las primeras 1, 2, 5, 10, 20, y total
% de componentes
for k = [1 2 5 10 20 rank(X)]
    figure(3);
    Xhat=U(:,1:k)*S(1:k,1:k)*V(:,1:k)';
    imshow(Xhat);
    colormap(gray);
    axis off;
    title(sprintf('Primeros %g componentes', k));
    pause
end

% Encontrar el valor de k que mantenga al menos el 90% de la variabilidad
figure(4);
grid on;
hold on;
diagonal=diag(S);
bestK=size(diagonal,1);
for i=1:size(diagonal,1)
    variabilidad(i)=sum(diagonal(1:i))/sum(diagonal);
    if (variabilidad(i)>=0.9) && (i<bestK)
        bestK=i;
    end
end
plot(1:size(diagonal), variabilidad, 'r.');
title('Variabilidad segun el numero de componentes');
xlabel ('Numero de componentes');
ylabel ('Variabilidad');
pause

% Graficar la reconstruccion con las primeras k componentes con las que se
% obtiene una variabilidad del 90%
figure(5);
Xhat=U(:,1:bestK)*S(1:bestK,1:bestK)*V(:,1:bestK)';
imshow(Xhat);
colormap(gray);
axis off;
title(sprintf('Primeros %g componentes', bestK));
pause

% Calcular y mostrar el ahorro en espacio
figure(6);
grid on;
hold on;
title('Ratio de compresion');
xlabel ('Numero de componentes');
ylabel ('Compresion respecto a la original');
plot(diag(S),'b.');
OriginalSize=size(U,1)^2+size(V,1)^2+size(V,1);
CompressSize=(size(U,1)+size(V,1)+1)*bestK;
dif=CompressSize/OriginalSize*100;
fprintf('Original Size = %d components\nCompression Size = %d components\n', OriginalSize,CompressSize);
fprintf('La imagen comprimida ocupa un %4.2f%% de lo ocupa la imagen original\n', dif);