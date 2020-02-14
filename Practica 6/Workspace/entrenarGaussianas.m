function modelo = entrenarGaussianas( Xtr, ytr, clases, NaiveBayes, landa )
% Entrena una Gaussiana para cada clase y devuelve:
% modelo{i}.N     : Numero de muestras de la clase i
% modelo{i}.mu    : Media de la clase i
% modelo{i}.Sigma : Covarianza de la clase i
% Si NaiveBayes = 1, las matrices de Covarianza seran diagonales
% Se regularizaran las covarianzas mediante: Sigma = Sigma + landa*eye(D)
    for i=1:size(clases,2)
        filas=find(ytr==clases(i));
        filasX=Xtr(filas,:);
        modelo{clases(i)}.N = size(ytr(ytr==clases(i)),1);
        modelo{clases(i)}.mu = sum(filasX)/modelo{clases(i)}.N;
        if NaiveBayes==1
            d=diag(cov(filasX));
            modelo{clases(i)}.Sigma = diag(d)+landa*eye(size(Xtr,2));
        else        
            modelo{clases(i)}.Sigma = cov(filasX)+landa*eye(size(Xtr,2));
        end
    end
end
