function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
% ====================== YOUR CODE HERE ======================
% Instructions: Perform a single gradient step on the parameter vector
%               theta. 
%
% Hint: While debugging, it can be useful to print out the values
%       of the cost function (computeCost) and gradient here.
% X * theta - y is 'difference' or a sort of cost (non abs)
% (72 * 2)' * 72 * 1 = 2 * 1
for i = 1:num_iters
theta = theta - (alpha / m) * (transpose(X) * ((X * theta) - y));
J_history(i) = computeCost(X, y, theta);
endfor
end
