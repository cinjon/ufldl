function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

# Compute the weight decay
weight_decay = lambda/2 * sum((theta.^2)(:));

# Compute theta * inputData;
class_probabilities = theta * data;
# Subtract the max value to make the computations more palatable
class_probabilities = bsxfun(@minus, class_probabilities, max(class_probabilities, [], 1));
# Exponentiate the values
class_probabilities = exp(class_probabilities);
# Normalize each training set (one per column)
class_probabilities = bsxfun(@rdivide, class_probabilities, sum(class_probabilities));
# Log the values
indicator_cost = groundTruth .* log(class_probabilities);
# Sum and normalize the cost
indicator_cost = -1/numCases * sum(indicator_cost(:));

# Total the cost
cost = indicator_cost + weight_decay;

# ThetaGrad
thetagrad = -1/numCases * ((groundTruth - class_probabilities) * data') + \
            (lambda * theta);

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
