% Prepare the dataset
a0_values = linspace(-1, 1, 10); % Example range, replace with actual values
a1_values = linspace(-1, 1, 80); % Example range, replace with actual values
u_values = linspace(0, 1, 8);    % Example range, replace with actual values

% Generate all possible input combinations and corresponding b values
inputs = [];
outputs = [];
for a0 = a0_values
    for a1 = a1_values
        for u = u_values
            b = a0 + a1 * (1 - u);
            inputs = [inputs; a0, a1, u];
            outputs = [outputs; b];
        end
    end
end

% Define the neural network
net = feedforwardnet([10, 10]);  % Example with two hidden layers with 10 neurons each

% Split data into training and testing sets
[trainInd,~,testInd] = dividerand(size(inputs, 1), 0.7, 0, 0.3);
trainInputs = inputs(trainInd, :)';
trainOutputs = outputs(trainInd, :)';
testInputs = inputs(testInd, :)';
testOutputs = outputs(testInd, :)';

% Train the neural network
net = train(net, trainInputs, trainOutputs);

% Test the network
predictedOutputs = net(testInputs);

% Calculate the mean squared error
mseError = mse(net, testOutputs, predictedOutputs);

% Display the MSE
disp(['Mean Squared Error: ', num2str(mseError)]);
