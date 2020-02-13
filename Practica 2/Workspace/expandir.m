function [ Xexp ] = expandir( X, grados )
% Añade unos a la izquierda y hace la expansion polinomica de los atributos
% Por ejemplo: si X = [x1 x2 x3] y grados = [2 0 3], devuelve: 
% Xexp = [1 x1 x1^2 x3 x3^2 x3^3]

if size(X,2)~=length(grados)
    error('El numero de atributos y de grados debe coincidir');
end;
N = size(X,1);
Xexp = [ones(N,1)];
for i = 1:length(grados)
    %Expande el atributo i
    if grados(i) > 0
        for g = 1:grados(i)
            Xexp = [Xexp X(:,i).^g];
        end
    end
end
end

