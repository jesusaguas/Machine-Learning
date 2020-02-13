function [ error ] = RMSE( w, X, y )
%Calcula el error RMSE de una regresión lineal
 
error = sqrt(mean((X*w-y).^2));

end

