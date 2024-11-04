close all;
clear;
clc;

%% Initialise network parameters

input_size  = 2;    % Number of input neurons
hidden_size = 10;    % Number of neurons in hidden layer
output_size = 1;    % Number of output neurons
learning_rate = .01;
epochs      = 1000;

%% Random weight and bias initialisation
% Range between -0.5 to 0.5 for stability and symmetry

W1 = rand(hidden_size, input_size) -.5;     % Input to hidden weights
b1 = rand(hidden_size, 1) -.5;              % bias hidden layer
W2 = rand(output_size, hidden_size) -.5;    % Hidden to output weights
b2 = rand(output_size, 1) -.5;              % bias output layer

%% Define activation functions

sigmoid             = @(x) 1./(1+exp(-x));  % Sigmoid activation
sigmoid_derivative  = @(x) x .* (1 - x);        % Derivative of sigmoid

%% Training loops

for epoch=1:epochs
    % Training example (XOR problem)
    X = [0, 0; 0, 1; 1, 0; 1, 1]';
    Y = [0; 1; 1; 0]';

    % Forward Propagation
    Z1 = W1 * X + b1;
    A1 = sigmoid(Z1);
    Z2 = W2 * A1 + b2;
    A2 = sigmoid(Z2);
    
    % Compute loss (Mean Squared Error)
    
    loss = sum((Y - A2).^2) / length(Y);
    
    % Backpropagation
    
    dZ2 = (A2-Y) .* sigmoid_derivative(A2); % Error at ouput
    dW2 = dZ2 * A1';                        % Gradient for W2
    db2 = sum(dZ2, 2);                      % Gradient for b2

    dA1 = W2' * dZ2;
    dZ1 = dA1 .* sigmoid_derivative(A1);    % Error at hidden layer
    dW1 = dZ1 * X';                         % Gradient for W1
    db1 = sum(dZ1, 2);                      % Gradient for b1
    
    % Update weights and biases
    
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;

    % Display loss every 100 epochs

    if mod(epoch, 100) == 0
        disp(['Epoch ' num2str(epoch) ', Loss: ' num2str(loss)])
    end
    
end

%% Test the MLP on input data

output = sigmoid(W2 * sigmoid(W1 * X + b1) + b2);
disp('Predicted output: ');
disp(output);
