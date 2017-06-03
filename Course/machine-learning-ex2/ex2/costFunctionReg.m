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
reg_theta = [0; theta(2:size(theta))];
% Calculate regularization parameters 
J_regularization = (lambda / (2*m)) * transpose(reg_theta) * reg_theta;
theta_regularization = (lambda/m) * reg_theta;
% Calculate cost
[J, grad] = costFunction(theta, X, y);
% Complute regularized cost
J = J + J_regularization;
grad = grad + theta_regularization;
% =============================================================
end
