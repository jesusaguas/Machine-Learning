function [ best_lambda ] = kfold3(N, k, X, y)
    best_lambda=0;
    best_error=9999999;
    erroresCv = [];
    erroresTr = [];
    for lambda=-10:N
        Xexp = expandir(X, [10 5 5]);
        [ Xn, mu, sig ] = normalizar( Xexp );
        errorTr=0;
        errorCv=0;
        for fold=1:k
            [ Xcv, ycv, Xtr, ytr ] = particion( fold, k, Xn, y );
            H = Xtr'*Xtr + 10^lambda * diag([0 ones(1,20)]);
            th = H \ (Xtr'*ytr);
            %th = desnormalizar( th, mu, sig );
            errorTr = errorTr + RMSE( th, Xtr, ytr );
            errorCv = errorCv + RMSE( th, Xcv, ycv );
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
    ylabel('RMSE'); xlabel('Lambda Value');
    legend('Trainig', 'Validation')
end