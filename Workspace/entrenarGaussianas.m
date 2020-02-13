function modelo = entrenarGaussianas( Xtr, ytr, nc, NaiveBayes, landa )
% Entrena una Gaussiana para cada clase y devuelve:
% modelo{i}.N     : Numero de muestras de la clase i
% modelo{i}.mu    : Media de la clase i
% modelo{i}.Sigma : Covarianza de la clase i
% Si NaiveBayes = 1, las matrices de Covarianza seran diagonales
% Se regularizaran las covarianzas mediante: Sigma = Sigma + landa*eye(D)
    for i=1:nc
        filas=find(ytr==i);
        filasX=Xtr(filas,:);
        modelo{i}.N = size(ytr(ytr==i),1);
        modelo{i}.mu = sum(filasX)/modelo{i}.N;
        if NaiveBayes==1
            d=diag(cov(filasX));
            modelo{i}.Sigma = diag(d)+landa*eye(size(Xtr,2));
        else        
            modelo{i}.Sigma = cov(filasX)+landa*eye(size(Xtr,2));
        end
    end
end
