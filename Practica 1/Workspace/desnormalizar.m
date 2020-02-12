function [ wdes ] = desnormalizar( w, mu, sig )
% Desnormaliza los pesos de la regresion
wdes = w(2:end)./sig';
wdes = [w(1)-(mu*wdes); wdes];

end

