function [ Xcv, ycv, Xtr, ytr ] = particion( i, nfolds, X, y )
% Divide las muestras (X y), separando el fold i de nfolds para
% cross-validacion (Xcv ycv), y el resto para entrenamiento (Xtr, ytr)

N = size(y,1);  %Numero de muestras totales
indices = [1:N];
cv = (indices > floor((i-1)*N/nfolds)) & (indices <= floor(i*N/nfolds));   

Xcv = X( cv,:);
Xtr = X(~cv,:);
ycv = y( cv);
ytr = y(~cv);

end

