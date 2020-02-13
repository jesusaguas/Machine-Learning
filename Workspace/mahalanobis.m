function [d2, R] = mahalanobis(mu, Sigma, X)
% Calcula las distancias de Mahalanobis de las muestras X a
% la Gaussiana N(mu, Sigma) y devuelve R = chol(Sigma)

%X = bsxfun(@minus, X, mu');  % Falla en Matlabs antiguos
for i = 1:size(X,2)
    X(:,i) = X(:,i)- mu(i);
end
R = chol(Sigma);
z  = X / R;
d2 = sum(z.^2,2);  %Distancia de Mahalanobis

