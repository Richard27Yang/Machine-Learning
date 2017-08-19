function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta

htheta=sigmoid(X*theta); % hthta m*1
list_cost = -y.*log(htheta) -(1-y).*log(1-htheta);
J 	= 1/m * sum(list_cost);

%grad1 = 1/m * sum((htheta-y).* X(:, 1));
%grad2 = 1/m * sum((htheta-y).* X(:, 2));
%grad3 = 1/m * sum((htheta-y).* X(:, 3));
%grad(1) = grad1;
%grad(2) = grad2;
%grad(3) = grad3;

for i = 1:m
	% hypothesis = mx1 column vector
	% y = mx1 column vector
	% X = mxn matrix
	grad = grad + ( htheta(i) - y(i) ) * X(i, :)';
end

grad = grad/m;

% =============================================================

end
