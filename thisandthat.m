close all;
clear;
clc;

%% Experimental data

theta = 1:360;
f = 2/360;

w = 2 * pi * f * theta;

%%

inputs = load('inputs.mat').inputs;

data = load('mean_TorqueAndOtherData_16participants.mat');

P1 = data.mean_p01_t_v;
P2 = data.mean_p02_t_v;
P3 = data.mean_p03_t_v;
P4 = data.mean_p04_t_v;
P5 = data.mean_p05_t_v;
P6 = data.mean_p06_t_v;
P7 = data.mean_p07_t_v;
P8 = data.mean_p08_t_v;
P9 = data.mean_p09_t_v;
P10 = data.mean_p10_t_v;
P11 = data.mean_p11_t_v;
P12 = data.mean_p12_t_v;
P13 = data.mean_p13_t_v;
P14 = data.mean_p14_t_v;
P15 = data.mean_p15_t_v;
P16 = data.mean_p16_t_v;

Y = [P1' P2' P3' P4' P5' P6' P7' P8' P9' P10' P11' P12' P13' P14' P15' P16']';
clear('P*')

[trI, ~, teI] = dividerand(size(Y, 1), .7, 0, .3);
Ytr = Y(trI, :);

%% Initialise network parameters

input_size  = 9;    % Number of input neurons
hidden_size1 = 12;    % Number of neurons in hidden layer 1
hidden_size2 = 12;    % Number of neurons in hidden layer 2
hidden_size3 = 12;    % Number of neurons in hidden layer 3
hidden_size4 = 12;    % Number of neurons in hidden layer 4
output_size = 9;    % Number of output neurons
learning_rate = .0001;
epochs      = 50000;

%% Random weight and bias initialisation
% Range between -0.5 to 0.5 for stability and symmetry

W1 = rand(hidden_size1, input_size) -.5;     % Input to hidden 1 weights
b1 = rand(hidden_size1, 1) -.5;              % bias hidden layer 1

W2 = rand(hidden_size2, hidden_size1) -.5;     % Hidden 1 to hidden 2 weights
b2 = rand(hidden_size2, 1) -.5;              % bias hidden layer 2

W3 = rand(hidden_size3, hidden_size2) -.5;     % Hidden 2 to hidden 3 weights
b3 = rand(hidden_size3, 1) -.5;              % bias hidden layer 3

W4 = rand(hidden_size3, hidden_size2) -.5;     % Hidden 3 to hidden 4 weights
b4 = rand(hidden_size3, 1) -.5;              % bias hidden layer 4

W5 = rand(output_size, hidden_size3) -.5;    % Hidden 4 to output weights
b5 = rand(output_size, 1) -.5;              % bias output layer

% W1 = W1 * .01;
% b1 = b1 * .01;
% W2 = W2 * .01;
% b2 = b2 * .01;
% W3 = W3 * .01;
% b3 = b3 * .01;
% W4 = W4 * .01;
% b4 = b4 * .01;
% W5 = W5 * .01;
% b5 = b5 * .01;


%% Define activation functions

sigmoid             = @(x) 1./(1+exp(-x));  % Sigmoid activation
sigmoid_derivative  = @(x) x .* (1 - x);        % Derivative of sigmoid

%% Training loops

% Training sample

    m0      = rand();
    m1      = rand();
    m2      = rand();
    m3      = rand();
    m4      = rand();
    phi1    = -pi/2;
    phi2    = -pi/2;
    phi3    = -pi/2;
    phi4    = -pi/2;

    X = [m0; m1; m2; m3; m4; phi1; phi2; phi3; phi4];

%%

for epoch=1:2


    % Forward Propagation
    % Input layer
    Z1 = W1 * X + b1;
    A1 = sigmoid(Z1);

    % Hidden layer 2
    Z2 = W2 * A1 + b2;
    A2 = sigmoid(Z2);

    % Hidden layer 3
    Z3 = W3 * A2 + b3;
    A3 = sigmoid(Z3);

    % Hidden layer 4
    Z4 = W4 * A3 + b4;
    A4 = sigmoid(Z4);

    % output layer
    Z5 = W5 * A4 + b5;
    A5 = Z5;

    % Compute trajectories
    trajectories = get_traj(inputs, A5, theta, f);
    traj_tr = trajectories(trI,:); 

    
    % Compute loss (Mean Squared Error)
    
    loss = mean(sum((traj_tr - Ytr).^2) / size(Ytr, 1));

    dl_dA = (2 / size(Ytr, 1)) * sum(traj_tr - Ytr);

    % Gradient with respect to m0
    dl_dm0 = sum(dl_dA);

    % Gradients with respect to m1 : m4
    dl_dm1 = (dl_dA * mean((inputs(trI,1) ./ 10) * sin(w + A5(6)))');
    dl_dm2 = (dl_dA * mean((inputs(trI,2) ./ 10) * sin(w + A5(7)))');
    dl_dm3 = (dl_dA * mean((inputs(trI,3) ./ 10) * sin(w + A5(8)))');
    dl_dm4 = (dl_dA * mean((inputs(trI,4) ./ 10) * sin(w + A5(9)))');

    % Gradients with respect to m5 : m9
    dl_dm5 = (dl_dA * mean(A5(2) * (inputs(trI,1) ./ 10) * cos(w + A5(6)))');
    dl_dm6 = (dl_dA * mean(A5(3) * (inputs(trI,2) ./ 10) * cos(w + A5(7)))');
    dl_dm7 = (dl_dA * mean(A5(4) * (inputs(trI,3) ./ 10) * cos(w + A5(8)))');
    dl_dm8 = (dl_dA * mean(A5(5) * (inputs(trI,4) ./ 10) * cos(w + A5(9)))');

    % Back propagation

    dZ5 = [dl_dm0 dl_dm1 dl_dm2 dl_dm3 dl_dm4 dl_dm5 dl_dm6 dl_dm7 dl_dm8]';
    dW5 = dZ5 * A4';
    db5 = sum(dZ5, 2);

    dA4 = W5' * dZ5;
    dZ4 = dA4 .* sigmoid_derivative(A4);
    dW4 = dZ4 * A3';
    db4 = sum(dZ4, 2);

    dA3 = W4' * dZ4;
    dZ3 = dA3 .* sigmoid_derivative(A3);
    dW3 = dZ3 * A2';
    db3 = sum(dZ3, 2); 

    dA2 = W3' * dZ3;
    dZ2 = dA2 .* sigmoid_derivative(A2);
    dW2 = dZ2 * A1';
    db2 = sum(dZ2, 2);

    dA1 = W2' * dZ2;
    dZ1 = dA1 .* sigmoid_derivative(A1);    % Error at hidden layer
    dW1 = dZ1 * X';                         % Gradient for W1
    db1 = sum(dZ1, 2);   
     
%     % Update weights and biases
    
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    W3 = W3 - learning_rate * dW3;
    b3 = b3 - learning_rate * db3;
    W4 = W4 - learning_rate * dW4;
    b4 = b4 - learning_rate * db4;
    W5 = W5 - learning_rate * dW5;
    b5 = b5 - learning_rate * db5;
 
    % Display loss every 100 epochs

    if mod(epoch, 1) == 0
        disp(['Epoch ' num2str(epoch) ', Loss: ' num2str(loss)])
    end
    
end

%% Calculating trajectories



% trajectories = get_traj(inputs, A5, theta, f);

%% Plots

myplots = figure;

for i = 1 : size(Y,1)
    subplot(8, 2, i)
    hold on;
    plot(trajectories(i,:))
    plot(Y(i, :))
    hold off;
    xlim([0 360])
end

myplots.WindowState = 'Maximized';


