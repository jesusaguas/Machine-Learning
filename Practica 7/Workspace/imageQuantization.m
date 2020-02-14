%% Practica 7: Clusters
% Jesus Aguas Acin (736935)
clc
close all;

figure(1)
%im = imread('smallparrot.jpg');
%im = imread('molinos.jpg');
im = imread('SMPTE.png');
imshow(im)
title('Imagen original');

%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));

%% dimensiones
m = size(D,1);
n = size(D,2);

%% Kmeans
K = 8;

%% Inicializar centroides
[C,ia,ic] = unique(D(:,1:3),'rows');
numColors = size(C,1);

%p = randperm(numColors,K)';
%mu = C(p,:);

mu = zeros([K,3]);
for i=1:K
    fila=round(numColors/K*i);
    mu(i,:) = C(fila,:);
end

%bucle kmeans
oldC=ones(length(D),1);
c=zeros(length(D),1);
Jdistorsiones = [];
iteraciones = 0;
while ~isequal(c,oldC)
    oldC=c;
    iteraciones = iteraciones + 1;
    [mu, c] = kmeans(D,mu,K);
    J = mean((sqrt(sum((D(:,1:2)-mu(c,1:2)).^2,2))).^2);
    Jdistorsiones = [Jdistorsiones, J];
    display(iteraciones);
end

figure(2);
grid on; hold on;
plot(1:iteraciones, Jdistorsiones, '-b');
title(sprintf('Evolucion de la distorsion(K=%g)', K));
ylabel('J (Funcion de distorsion)'); 
xlabel('Numero de iteraciones');

J = mean((sqrt(sum((D(:,1:2)-mu(c,1:2)).^2,2))).^2); 

%% reconstruir imagen
 qIM=zeros(length(c),3);
 for h=1:K
     ind=find(c==h);
     qIM(ind,:)=repmat(mu(h,:),length(ind),1);
 end
 qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
 figure(3)
 imshow(uint8(qIM));
 title(sprintf('K=%g', K));
 pause;
 
 
figure(4);
grid on; hold on;
plot((1:100), numColors./(1:100), '-b');
title('Ratio de compresion');
ylabel('Compresion'); 
xlabel('Numero de clusters');
pause;
 
 %% Kmeans encontrar K ideal (metodo del codo)
Clusters = [1 2 3 5 10 20 40];
Jdistorsiones = [];
for k = Clusters
    %Inicializar centroides
    %p = randperm(numColors,k)';
    %mu = C(p,:);
    mu = zeros([k,3]);
    for i=1:k
        fila=round(numColors/k*i);
        mu(i,:) = C(fila,:);
    end

    %bucle kmeans
    oldC=ones(length(D),1);
    c=zeros(length(D),1);
    iteraciones = 0;
    while ~isequal(c,oldC)
        oldC=c;
        iteraciones = iteraciones + 1;
        [mu, c] = kmeans(D,mu,k);
    end
    J = mean((sqrt(sum((D(:,1:2)-mu(c,1:2)).^2,2))).^2);
    Jdistorsiones = [Jdistorsiones, J];
    %% reconstruir imagen
    qIM=zeros(length(c),3);
    for h=1:k
        ind=find(c==h);
        qIM(ind,:)=repmat(mu(h,:),length(ind),1);
    end
    qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
    figure(5)
    imshow(uint8(qIM));
    title(sprintf('K=%g', k));
end

figure(6);
grid on; hold on;
plot(Clusters, Jdistorsiones, '-b');
title('Evolucion de la distorsion');
ylabel('J (Funcion de distorsion)');
xlabel('K (numero de clusters)');