function yhat = clasificacionBayesiana(modelo, X, clases)
% Con los modelos entrenados, predice la clase para cada muestra X
    h=[];
    for i=1:size(clases,2)
        h=[h gaussLog(modelo{clases(i)}.mu, modelo{clases(i)}.Sigma, X)];
    end
    [A,yhat] = max(h');
end

