function munew = updateCentroids(D,c,K)
% D((m,n), m datapoints, n dimensions
% c(m) assignment of each datapoint to a class
%
% munew(K,n) new centroids

munew = zeros([K,3]);
for k=1:K
    %Obtenemos coordenadas de los puntos que pertenecen a la clase k
    Kdata = D((c==k),:);
    %Calculamos en nuevo centroide
    munew(k,:)=mean(Kdata);
end