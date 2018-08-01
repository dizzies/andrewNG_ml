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


H = sigmoid(X * theta);
%H = X * theta;
%printf("%d\n", size(H));

y1 = transpose(y);
theta1 = theta;
theta1(1,:) = [0];

[m,n] = size(X);

cost = (sum(theta1.^2) * lambda / 2 + (-y1 * log(H) - (1-y1) * log(1-H)))/ m;
%printf("%d\n", cost);

%printf("theta1 = %d\n", theta1);
H1 = sigmoid(X * theta1);
J = cost;
grad = ((transpose(X) * (H-y)) + lambda * theta1) / m;
 



% =============================================================

end
