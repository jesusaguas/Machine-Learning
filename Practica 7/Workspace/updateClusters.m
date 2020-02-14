function Z = updateClusters(D,mu)
% D(m,n), m datapoints, n dimensions
% mu(K,n) final centroids
%
% c(m) assignment of each datapoint to a class

m=size(D,1);
Z = zeros(length(D),1);
for i=1:m
    %Para cada punto se calcula su distancia a los centroides
    distancia = sqrt(sum((D(i,1:2)-mu(:,1:2)).^2,2));
    % Se le asigna la clase k mas cercana (distancia minima)
    [value, Z(i,1)]=min(distancia.^2);
end