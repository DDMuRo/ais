close all; clear; clc;

%% Initial values for trajectories

l       = 360;
theta   = 1 : l;
f       = 2 / l;
w       = 2 * pi * f * theta;

%% Experimental data

inputs  = load('inputs.mat').inputs;
data    = load('mean_TorqueAndOtherData_16participants.mat');
fns     = fieldnames(data);
ix      = find(contains(fns, 'mean_p'));
traj    = cell2mat(arrayfun(@(i) data.(fns{i}), ix, 'UniformOutput', 0));
[trI, ~, teI]   = dividerand(size(traj, 1), .7, 0, .3);
Y       = traj(trI, :);

%% Network parameters

inputSize   = size(inputs, 2);
hiddenSize  = 10;
hiddenSize1 = 10;
outputSize  = 9;

% Training parameters
alpha       = 1e-8;
epochs      = 1e5;
nSamples    = size(trI, 2);

% Initialise weights and biases
rng(1);     % set the seed for the random number (Marsenne Twister) generator to 1
W1 = randn(hiddenSize, inputSize) * .01;
b1 = zeros(hiddenSize, 1);
W2 = randn(outputSize, hiddenSize) * .01;
b2 = zeros(outputSize, 1);

% Training data
p_values = inputs(trI, :)';
target_output = Y';

record = [];

% Training loop for batch processing
for epoch = 1 : epochs
    totalLoss = 0;
    
    % Initialise cumulative gradients
    dW1_total = zeros(size(W1));
    db1_total = zeros(size(b1));
    dW2_total = zeros(size(W2));
    db2_total = zeros(size(b2));

    % Loop over all training samples
    for j = 1 : nSamples
        % Forward propagation for each sample
        Z1 = W1 * p_values(:, j) + b1;
        A1 = max(0, Z1);    % ReLU activation
        Z2 = W2 * A1 + b2;
        predictions = Z2(1) + ...
            p_values(1, j) * Z2(2) * sin(w + Z2(3)) + ...
            p_values(2, j) * Z2(4) * sin(w + Z2(5)) + ...
            p_values(3, j) * Z2(6) * sin(w + Z2(7)) + ...
            p_values(4, j) * Z2(8) * sin(w + Z2(9));
         
        % Compute loss for this sample
        loss = mean((predictions - target_output(:, j)').^2);
%         loss = sum((predictions - target_output(:, j)').^2);
        totalLoss = totalLoss + loss;

        % Backpropagation for this sample
        dl_pred = (2 / l) * (predictions - target_output(:, j)');
%         dl_pred = (2) * (predictions - target_output(:, j)');

        dZ2 = [sum(dl_pred);...
            dl_pred * p_values(1, j) * sin(w + Z2(3))';...
            dl_pred * Z2(2) * p_values(1, j) * cos(w + Z2(3))';...
            dl_pred * p_values(2, j) * sin(w + Z2(5))';...
            dl_pred * Z2(4) * p_values(2, j) * cos(w + Z2(5))';...
            dl_pred * p_values(3, j) * sin(w + Z2(7))';...
            dl_pred * Z2(6) * p_values(3, j) * cos(w + Z2(7))';...
            dl_pred * p_values(4, j) * sin(w + Z2(9))';...
            dl_pred * Z2(8) * p_values(4, j) * cos(w + Z2(9))'];
        dW2 = dZ2 * A1';
        db2 = dZ2;

        dA1 = W2' * dZ2;
        dZ1 = dA1 .* (Z1 > 0);  % ReLU derivative
        dW1 = dZ1 * p_values(:, j)';
        db1 = dZ1;

        % Accumulative gradients for batch update
        dW1_total = dW1_total + dW1;
        db1_total = db1_total + db1;
        dW2_total = dW2_total + dW2;
        db2_total = db2_total + db2;

    end

    % Average loss across all samples
    totalLoss = totalLoss / nSamples;

    % Gradient Descent Update (batch update)
    W1 = W1 - alpha * (dW1_total / nSamples);
    b1 = b1 - alpha * (db1_total / nSamples);
    W2 = W2 - alpha * (dW2_total / nSamples);
    b2 = b2 - alpha * (db2_total / nSamples);

    % Print loss every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, totalLoss);
    end

    record = [record totalLoss];
    figure(1)
    plot(record)

end

%%

trajectories = get_traj2(inputs, Z2, w);

for i = 1 : size(traj,1)
    if any(trI == i)
        word = 'train';
    else
        word = 'test';
    end
    str_title = ['Subject ' num2str(i) ', ' word];
    subplot(8, 2, i)
    hold on;
    plot(trajectories(i,:))
    plot(traj(i, :))
    hold off;
    title(str_title)
    xlim([0 360])
end
