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
hiddenSize  = 12;
hiddenSize1 = 12;
hiddenSize2 = 12;
outputSize  = 9;

% Training parameters
alpha       = 2.55e-5;
epochs      = 1e6;
nSamples    = size(trI, 2);

% Initialise weights and biases
rng(1);     % set the seed for the random number (Marsenne Twister) generator to 1
W1 = randn(hiddenSize, inputSize) * .01;
b1 = zeros(hiddenSize, 1);
W2 = randn(hiddenSize1, hiddenSize) * .01;
b2 = zeros(hiddenSize1, 1);
W3 = randn(hiddenSize2, hiddenSize1) * .01;
b3 = zeros(hiddenSize2, 1);
W4 = randn(outputSize, hiddenSize2) * .01;
b4 = zeros(outputSize, 1);

% Training data
p_values = inputs(trI, :)';
target_output = Y';

Zmax = zeros(outputSize, 1);
ref = 1e10;

record = [];

% Training loop for batch processing
for epoch = 1 : epochs
    totalLoss = 0;
    
    % Initialise cumulative gradients
    dW1_total = zeros(size(W1));
    db1_total = zeros(size(b1));
    dW2_total = zeros(size(W2));
    db2_total = zeros(size(b2));
    dW3_total = zeros(size(W3));
    db3_total = zeros(size(b3));
    dW4_total = zeros(size(W4));
    db4_total = zeros(size(b4));

    % Loop over all training samples
    for j = 1 : nSamples
        % Forward propagation for each sample
        Z1 = W1 * p_values(:, j) + b1;
        A1 = max(0, Z1);    % ReLU activation
        Z2 = W2 * A1 + b2;
        A2 = max(0, Z2);
        Z3 = W3 * A2 + b3;
        A3 = max(0, Z3);
        Z4 = W4 * A3 + b4;
        predictions = Z4(1) + ...
            p_values(1, j) * Z4(2) * sin(w + Z4(3)) + ...
            p_values(2, j) * Z4(4) * sin(w + Z4(5)) + ...
            p_values(3, j) * Z4(6) * sin(w + Z4(7)) + ...
            p_values(4, j) * Z4(8) * sin(w + Z4(9));
         
        % Compute loss for this sample
        loss = mean((predictions - target_output(:, j)').^2);
%         loss = sum((predictions - target_output(:, j)').^2);
        totalLoss = totalLoss + loss;

        % Backpropagation for this sample
        dl_pred = (2 / l) * (predictions - target_output(:, j)');
%         dl_pred = (2) * (predictions - target_output(:, j)');

        dZ4 = [sum(dl_pred);...
            dl_pred * p_values(1, j) * sin(w + Z4(3))';...
            dl_pred * Z4(2) * p_values(1, j) * cos(w + Z4(3))';...
            dl_pred * p_values(2, j) * sin(w + Z4(5))';...
            dl_pred * Z4(4) * p_values(2, j) * cos(w + Z4(5))';...
            dl_pred * p_values(3, j) * sin(w + Z4(7))';...
            dl_pred * Z4(6) * p_values(3, j) * cos(w + Z4(7))';...
            dl_pred * p_values(4, j) * sin(w + Z4(9))';...
            dl_pred * Z4(8) * p_values(4, j) * cos(w + Z4(9))'];
        dW4 = dZ4 * A3';
        db4 = dZ4;

        dA3 = W4' * dZ4;
        dZ3 = dA3 .* (Z3 > 0);
        dW3 = dZ3 * A2';
        db3 = dZ3;

        dA2 = W3' * dZ3;
        dZ2 = dA2 .* (Z2 > 0);
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
        dW3_total = dW3_total + dW3;
        db3_total = db3_total + db3;
        dW4_total = dW4_total + dW4;
        db4_total = db4_total + db4;

    end

    % Average loss across all samples
    totalLoss = totalLoss / nSamples;

    if totalLoss < ref
        ref = totalLoss;
        Zmax = Z4;
    end

    % Gradient Descent Update (batch update)
    W1 = W1 - alpha * (dW1_total / nSamples);
    b1 = b1 - alpha * (db1_total / nSamples);
    W2 = W2 - alpha * (dW2_total / nSamples);
    b2 = b2 - alpha * (db2_total / nSamples);
    W3 = W3 - alpha * (dW3_total / nSamples);
    b3 = b3 - alpha * (db3_total / nSamples);
    W4 = W4 - alpha * (dW4_total / nSamples);
    b4 = b4 - alpha * (db4_total / nSamples);

    % Print loss every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, totalLoss);
    end

    record = [record totalLoss];
    figure(1)
    plot(record)

end

%%

trajectories = get_traj2(inputs, Zmax, w);

figure(2);

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
