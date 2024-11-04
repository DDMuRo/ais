% Initialize Network Parameters
input_size = 2;       % Number of input neurons
hidden_size = 8;      % Increased to 8 neurons in hidden layer
output_size = 1;      % Number of output neurons
learning_rate = 0.5;  % Increased learning rate
epochs = 5000;        % Increased epoch count for better convergence

% Random weight and bias initialization
W1 = rand(hidden_size, input_size) - 0.5; % Input to hidden weights
b1 = rand(hidden_size, 1) - 0.5;          % Hidden layer bias
W2 = rand(output_size, hidden_size) - 0.5; % Hidden to output weights
b2 = rand(output_size, 1) - 0.5;          % Output layer bias

% Sigmoid Activation Functions
sigmoid = @(x) 1 ./ (1 + exp(-x));       % Sigmoid activation
sigmoid_derivative = @(x) x .* (1 - x);  % Derivative of Sigmoid

% Training Loop
for epoch = 1:epochs
    % XOR Input and Output
    X = [0, 0; 0, 1; 1, 0; 1, 1]';
    Y = [0; 1; 1; 0]';

    % Forward Propagation
    Z1 = W1 * X + b1;      % Hidden layer linear combination
    A1 = sigmoid(Z1);      % Hidden layer activation
    Z2 = W2 * A1 + b2;     % Output layer linear combination
    A2 = sigmoid(Z2);      % Output layer activation

    % Loss (Mean Squared Error)
    loss = sum((Y - A2).^2) / length(Y);

    % Backpropagation
    dZ2 = (A2 - Y) .* sigmoid_derivative(A2); % Output layer error
    dW2 = dZ2 * A1';                          % Gradient for W2
    db2 = sum(dZ2, 2);                        % Gradient for b2

    dA1 = W2' * dZ2;                          % Backprop error for hidden layer
    dZ1 = dA1 .* sigmoid_derivative(A1);      % Hidden layer error
    dW1 = dZ1 * X';                           % Gradient for W1
    db1 = sum(dZ1, 2);                        % Gradient for b1

    % Update Weights and Biases
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;

    % Display loss every 500 epochs
    if mod(epoch, 500) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end

% Test the MLP on input data
output = sigmoid(W2 * sigmoid(W1 * X + b1) + b2);
disp('Predicted output:');
disp(output);
