function [ error ] = RMSE( w, X, y )
%Calcula el error RMSE de una regresion lineal
 
error = sqrt(mean((X*w-y).^2));

end

