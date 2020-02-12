function [ Xn, mu, sig ] = normalizar( X )
% Normaliza los atributos

mu = mean(X(:,2:end));
sig = std(X(:,2:end));
Xn = X;
% N = size(X,1);
% Xn(:,2:end) = ( X(:,2:end) - repmat(mu,N,1) )./ repmat(sig,N,1);
for i = 2:size(X,2)
    Xn(:,i) = (X(:,i) - mu(i-1)) / sig(i-1); 
end

end

