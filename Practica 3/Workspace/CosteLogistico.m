function [J,grad,Hess] = CosteLogistico(theta,X,y) % Calcula el coste logistico, y si se piden,
% su gradiente y su Hessiano
N = size(X,2);
h = 1./(1+exp(-(X*theta)));
J = (-y'*log(h) - (1-y')*log(1-h))/N; 
if nargout > 1
    grad = X'*(h-y)/N;
end
if nargout > 2
  R = diag(h.*(1-h));
  Hess = X'*R*X/N;
end