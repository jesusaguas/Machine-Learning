function yhat = clasificacionBayesiana(modelo, X)
% Con los modelos entrenados, predice la clase para cada muestra X
    h=[];
    for i=1:10
        h=[h gaussLog(modelo{i}.mu, modelo{i}.Sigma, X)];
    end
    [A,yhat] = max(h');
end

