function  verConfusiones(Xtr, ytr, yhat);
% Muestra una imagen con ejemplos de las confusiones habidas 

if size(Xtr,2) == 401
   Xtr(:,1)=[];  %Si hay columna de unos, la ignora
end
Xdib = zeros(100,400);
nc = 10;
k = 0;
for i = 1:nc
    for j = 1:nc
        k = k +1;
        conf = find((ytr==i) & (yhat==j));
        if length(conf>0)
            Xdib(k,:)=Xtr(conf(1),:);
        end
    end
end
figure        
displayData(Xdib);
title('Ejemplos de Confusiones');
drawnow;
