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
%d
% Note: grad should have the same dimensions as theta
%

H = sigmoid(X * theta);
%H = X * theta;
%printf("%d\n", size(H));

y1 = transpose(y);
[m,n] = size(X);


cost = (-y1 * log(H) - (1-y1) * log(1-H)) / m;
J = cost;
%printf("%d\n", cost);
grad = (transpose(X) * (H-y)) / m;






% =============================================================

end
