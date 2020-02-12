function [J,grad] = costeHuber(theta,X,y,d)
r = X*theta-y;
good = abs(r) <= d;
J = (1/2)*sum(r(good).^2) + d*sum(abs(r(~good))) - (1/2)*sum(~good)*d^2; 
grad = X(good,:)'*r(good) + d*X(~good,:)'*sign(r(~good));
end