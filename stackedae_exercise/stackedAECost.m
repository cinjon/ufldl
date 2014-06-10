function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)

% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.

% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

stackDepth = numel(stack);

% vars{d}.s = delta, vars{d}.z = result, vars{d}.a = activation of result
vars = cell(stackDepth+1);

% Compute the forward activations
vars{1}.a = data;
for d = 1:stackDepth
    vars{d+1}.z = stack{d}.w * vars{d}.a + repmat(stack{d}.b, [1, size(vars{d}.a, 2)]);
    vars{d+1}.a = sigmoid(vars{d+1}.z);
end

% Compute the vector of conditional probabilities, direct from softmaxCost calculation
P = softmaxTheta * vars{stackDepth + 1}.a;
P = exp(bsxfun(@minus, P, max(P)));
P = bsxfun(@rdivide, P, sum(P));

% Compute the errors due to each node
vars{stackDepth+1}.s = -softmaxTheta' * (groundTruth - \
                                        P) .* \
                         sigmoidDeriv(vars{stackDepth+1}.a);
for d = stackDepth:-1:2
    vars{d}.s = stack{d}.w' * vars{d+1}.s .* sigmoidDeriv(vars{d}.a);
end

% Compute the partial derivatives
for d = 1:stackDepth
    stackgrad{d}.w = vars{d+1}.s * vars{d}.a' / numClasses;
    stackgrad{d}.b = vars{d+1}.s * ones(size(vars{d+1}.s, 2), 1) / numClasses;
end

indicator_cost = -1/numClasses * groundTruth(:)' * log(P(:));
cost = indicator_cost + lambda/2 * sum(softmaxTheta(:) .^2); % Weight decay

softmaxThetaGrad = -1/numClasses * (groundTruth - P) * vars{stackDepth+1}.a' + lambda * softmaxTheta;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function deriv = sigmoidDeriv(x)
  deriv = x .* (1-x);
end
