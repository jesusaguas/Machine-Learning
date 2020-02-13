function [ best_lambda ] = kfold(N, k, X, y, NaiveBayes)
    best_lambda=0;
    best_error=9999999;
    erroresCv = [];
    erroresTr = [];
    for lambda=-10:N
        errorTr=0;
        errorCv=0;
        for fold=1:k
            % Separamos entre datos de entrenamiento y datos de validacion
            [ Xcv, ycv, Xtr, ytr ] = particion( fold, k, X, y );
            
            % Obtenemos modelo
            modelo = entrenarGaussianas( Xtr, ytr, 10, NaiveBayes, 10^lambda);
            
            % Error de prediccion para los datos de entrenamiento
            yhatTr = clasificacionBayesiana(modelo, Xtr);
            errorTr = errorTr + mean(yhatTr'~=ytr);
            
            % Error de prediccion para los datos de validacion
            yhatCv = clasificacionBayesiana(modelo, Xcv);
            errorCv = errorCv + mean(yhatCv'~=ycv);
            
        end
        errorTr = errorTr / k;
        errorCv = errorCv / k;
        erroresTr = [erroresTr, errorTr];
        erroresCv = [erroresCv, errorCv];
        if errorCv < best_error
            best_lambda = lambda;
            best_error = errorCv;
        end
    end
    figure;
    grid on; hold on;
    plot(-10:N, erroresTr, '-b');
    plot(-10:N, erroresCv, '-r');
    title('Evolucion de los errores');
    ylabel('Error prediccion'); xlabel('Lambda Value');
    legend('Trainig', 'Validation')
end