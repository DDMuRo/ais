close all;
clear;
clc;

%% 1. inputs & outputs

theta = 0:360;
f = 2/360;
phi = -pi/2;

% figure;
% plot(theta,sin(2*pi*f*theta+phi))

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

outputs = [P1' P2' P3' P4' P5' P6' P7' P8' P9' P10' P11' P12' P13' P14' P15' P16']';
clear('P*')

%% Neural network definition

net = fitnet([10,10]);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 30/100;

% net.inputs{1}.size = 4;
% net.outputs{2}.size = size(outputs,2);

% [Itrain,~,Itest] = dividerand(size(inputs,1),.7,0,.3);
% Intrain = inputs(Itrain,:);
% Outtrain = outputs(Itrain,:);
% Intest = inputs(Itest,:);
% Outtest = outputs(Itest,:);

%% Train the neural network
net = train(net, Intrain, Outtrain);


% Test the network
predictedOutputs = net(Intest);

% Calculate the mean squared error
mseError = mse(net, Outtest, predictedOutputs);

% Display the MSE
disp(['Mean Squared Error: ', num2str(mseError)]);
