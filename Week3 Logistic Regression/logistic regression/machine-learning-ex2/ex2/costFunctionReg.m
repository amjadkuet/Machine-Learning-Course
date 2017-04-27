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



z = zeros(m,1);
I = ones(m,1);
z = X*theta;
h_theta_X = sigmoid(z);

theta_xl_th0 = theta(2:length(theta));  % theta excluding theta0

diff = h_theta_X - y;

grad0 = sum(diff)/m;

X_xl_x0 = X(:, 2:size(X,2)); % X excluding x0

grad_rest = (sum(diff.*X_xl_x0)/m)' .+ (lambda/m)*theta_xl_th0;
grad = [grad0;grad_rest];


J_part1 = y.*log(h_theta_X);
J_part2 = (I.-y).*log(I.-h_theta_X);
J_part3 = (lambda/(2*m))*sum(theta_xl_th0.*theta_xl_th0);
J = -1*sum(J_part1.+J_part2)/m +J_part3;


% =============================================================

end
