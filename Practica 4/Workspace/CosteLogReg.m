function [J,grad,Hess] = CosteLogReg(theta,X,y,lambda)
[N, D] = size(X);
h = 1./(1+exp(-(X*theta)));
th = theta; 
th(1)= 0;   %Para no penalizar theta_0
J =(-y'*log(h) - (1-y')*log(1-h))/N +(lambda/2)*(th'*th);
if nargout > 1
    grad = X'*(h-y)/N + lambda*th;
end
if nargout > 2
    R = diag(h.*(1-h));
    Hess = X'*R*X/N + diag([0 lambda*ones(1,D-1)]);
end
end