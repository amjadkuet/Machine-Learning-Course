function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

X_xl_x0 = X(:, 2:size(X,2));
theta_xl_th0 = theta(2:length(theta));

yp = X*theta;
diff = yp-y;

diff_sq = diff.*diff;
J_part1 = sum(diff_sq)/(2*m);

J_part2 = (lambda/(2*m))*sum(theta_xl_th0.*theta_xl_th0);
J = J_part1+J_part2;

grad0 = sum(diff)/m;
grad_rest = (sum(diff.*X_xl_x0)/m)' .+ (lambda/m)*theta_xl_th0;
grad = [grad0;grad_rest];









% =========================================================================

grad = grad(:);

end
