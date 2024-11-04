close all;
clear;
clc;

%% Initialise network parameters

input_size  = 2;    % Number of input neurons
hidden_size1 = 12;    % Number of neurons in hidden layer 1
hidden_size = 12;    % Number of neurons in hidden layer 2
output_size = 1;    % Number of output neurons
learning_rate = 3;
epochs      = 50000;

%% Random weight and bias initialisation
% Range between -0.5 to 0.5 for stability and symmetry

W1 = rand(hidden_size1, input_size) -.5;     % Input to hidden weights
b1 = rand(hidden_size1, 1) -.5;              % bias hidden layer

W2 = rand(hidden_size, hidden_size1) -.5;     % Input to hidden weights
b2 = rand(hidden_size, 1) -.5;              % bias hidden layer

W3 = rand(output_size, hidden_size) -.5;    % Hidden to output weights
b3 = rand(output_size, 1) -.5;              % bias output layer

%% Define activation functions

sigmoid             = @(x) 1./(1+exp(-x));  % Sigmoid activation
sigmoid_derivative  = @(x) x .* (1 - x);        % Derivative of sigmoid

%% Training loops

for epoch=1:1
    % Training example (XOR problem)
    X = [0, 0; 0, 1; 1, 0; 1, 1]';
    Y = [0; 1; 1; 0]';

    % Forward Propagation
    Z1 = W1 * X + b1;
    A1 = sigmoid(Z1);
    Z2 = W2 * A1 + b2;
    A2 = sigmoid(Z2);
    Z3 = W3 * A2 + b3;
    A3 = sigmoid(Z3);
    
    % Compute loss (Mean Squared Error)
    
    loss = sum((Y - A3).^2) / length(Y);
    
    % Backpropagation
    
    dZ3 = (A3-Y) .* sigmoid_derivative(A3); % Error at ouput
    dW3 = dZ3 * A2';                        % Gradient for W2
    db3 = sum(dZ3, 2);                      % Gradient for b2

    dA2 = W3' * dZ3;
    dZ2 = dA2 .* sigmoid_derivative(A2);
    dW2 = dZ2 * A1';
    db2 = sum(dZ2, 2);

    dA1 = W2' * dZ2;
    dZ1 = dA1 .* sigmoid_derivative(A1);    % Error at hidden layer
    dW1 = dZ1 * X';                         % Gradient for W1
    db1 = sum(dZ1, 2);                      % Gradient for b1
    
    % Update weights and biases
    
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    W3 = W3 - learning_rate * dW3;
    b3 = b3 - learning_rate * db3;

    % Display loss every 100 epochs

    if mod(epoch, 100) == 0
        disp(['Epoch ' num2str(epoch) ', Loss: ' num2str(loss)])
    end
    
end

%% Test the MLP on input data

output = sigmoid(W3 * sigmoid(W2 * sigmoid(W1 * X + b1) + b2)+b3);
disp('Predicted output: ');
disp(output);
