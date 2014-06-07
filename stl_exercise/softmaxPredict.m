function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix
% pred, where pred(i) is argmax_c P(y(c) | x(i)).

% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start
%                from 1.

# Compute theta * inputData;
class_probabilities = theta * data;
# Subtract the max value to make the computations more palatable
class_probabilities = bsxfun(@minus, class_probabilities, max(class_probabilities, [], 1));
# Exponentiate the values
class_probabilities = exp(class_probabilities);
# Normalize each training set (one per column)
class_probabilities = bsxfun(@rdivide, class_probabilities, \
                             sum(class_probabilities));

# Now we have the P(y(c) | x(i)) for each input datum.
# Get the argmax_c of that:

[max_elements, pred] = max(class_probabilities);


% ---------------------------------------------------------------------

end
