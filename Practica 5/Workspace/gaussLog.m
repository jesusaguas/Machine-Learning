function logp = gaussLog(mu, Sigma, X)
%Calcula el Logaritmo neperiano de la pdf Gaussiana para las muestras X

D = size(X,2);
[d2, R] = mahalanobis(mu, Sigma, X);
logden = sum(log(diag(R))) + 0.5*D*log(2*pi);
logp = -0.5*d2 - logden;
