function [ best_lambda ] = kfold(N, k, X, y)
    best_lambda=0;
    best_error=9999999;
    erroresCv = [];
    erroresTr = [];
    for lambda=-10:N
        errorTr=0;
        errorCv=0;
        for fold=1:k
            [ Xcv, ycv, Xtr, ytr ] = particion( fold, k, X, y );
            options.display = 'final';
            options.method = 'newton';
            initial_theta = Xtr \ ytr;
            th = minFunc(@CosteLogReg, initial_theta, options, Xtr, ytr, 10^lambda);
            prediccion = 1./(1+exp(-(Xtr*th)));
            prediccion(prediccion < 0.5) = 0;
            prediccion(prediccion >= 0.5) = 1;
            errorTr = errorTr + mean(prediccion~=ytr);
            prediccion = 1./(1+exp(-(Xcv*th)));
            prediccion(prediccion < 0.5) = 0;
            prediccion(prediccion >= 0.5) = 1;
            errorCv = errorCv + mean(prediccion~=ycv);
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