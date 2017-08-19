function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hthta mx1
% list_cost m*1
% theta n*1
htheta=sigmoid(X*theta); 
list_cost = -y.*log(htheta) -(1-y).*log(1-htheta);
J 	= 1/m * sum(list_cost) + lambda/2/m * (sum(theta.^2)-theta(1)^2) ;

for i = 1:m
	% htheta = mx1 column vector
	% y = mx1 column vector
	% X = mxn matrix
	grad = grad + ( htheta(i) - y(i) ) * X(i, :)' + lambda/m*theta;
end

grad(1) = 0;
for i = 1:m
	% htheta = mx1 column vector
	% y = mx1 column vector
	% X = mxn matrix
	grad(1) = grad(1) + ( htheta(i) - y(i) ) * X(i, 1)' ;
end

% grad n*1 vector
grad = grad/m;

% =============================================================

end
