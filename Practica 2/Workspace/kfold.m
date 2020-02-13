function [ best_grado ] = kfold(N, k, X, y)
    best_grado=0;
    best_error=9999999;
    erroresCv = [];
    erroresTr = [];
    for grado=1:N
        Xexp = expandir( X, [grado 1 1]);
        [ Xn, mu, sig ] = normalizar( Xexp );
        errorTr=0;
        errorCv=0;
        for fold=1:k
            [ Xcv, ycv, Xtr, ytr ] = particion( fold, k, Xn, y );
            th = Xtr \ ytr;
            %th = desnormalizar( w, mu, sig );
            errorTr = errorTr + RMSE( th, Xtr, ytr );
            errorCv = errorCv + RMSE( th, Xcv, ycv );
        end
        errorTr = errorTr / k;
        errorCv = errorCv / k;
        erroresTr = [erroresTr, errorTr];
        erroresCv = [erroresCv, errorCv];
        if errorCv < best_error
            best_grado = grado;
            best_error = errorCv;
        end
    end
    figure;
    grid on; hold on;
    plot(1:N, erroresTr, '-b');
    plot(1:N, erroresCv, '-r');
    title('Evolucion de los errores');
    ylabel('RMSE'); xlabel('Grado del polinomio');
    legend('Trainig', 'Validation')
end