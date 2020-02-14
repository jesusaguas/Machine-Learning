function [mu, c] = kmeans(D,mu0, K)

% D(m,n), m datapoints, n dimensions
% mu0(K,n) K initial centroids
%
% mu(K,n) final centroids
% c(m) assignment of each datapoint to a class

c = updateClusters(D,mu0);
mu = updateCentroids(D,c,K);