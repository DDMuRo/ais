close all;
clear;
clc;

% Symbolic regression script applied to calculate crank torque in function
% of candence, power of the pedalling and weight and height of the subject
% [21/20/2024]

% Initialise parameters

pSize = 100;
Ngen = 50;

% Operators and variables

operators = {'+', '-', '*', '/'};
variables = {'height', 'weight', 'cadence', 'power'};

% Inputs

inputs = load('inputs.mat').inputs;


%% Outputs

[n, m] = size(inputs);

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

outputs = [P1' P2' P3' P4' P5' P6' P7' P8' P9' P10' P11' P12' P13' P14' P15' P16'];

clear('P*')

%%

% f = @(weight, cadence) eval(pop{1});
f = @(height,weight,cadence,power) eval('height+weight+cadence+power');
% B = arrayfun(@(i,j) f(test(i,1),test(j,2)), 1:size(test, 1), 1:size(test,1));
B = arrayfun(@(i) f(inputs(i,1),inputs(i,2),inputs(i,3),inputs(i,4)), 1:size(test, 1));


%% Generate initial population

pop = genPop(pSize, operators, variables);

for gen = 1:1      %1:pSize
    %
    % Step 2: Evaluate fitness

    % Step 3: Selection (tournament selection, roulette wheel, etc)

    % Step 4: Crossover and Mutation

    % Step 5: Update population

end

% The best expression is found at the end




%% Functions

function pop = genPop(pSize, operators, variables)

    pop = cell(pSize, 1); % Preallocate cell arrat for population

    for i = 1:pSize
        % Randomly generate expressions (trees) for each individual
        pop{i} = genRandE(operators, variables, 3); % Depth of
    end
end

function expr = genRandE(operators, variables, maxDepth)
    if maxDepth == 1
        % If at maximum depth, choose either a variable or a constant
        if rand<.5
            expr = variables{randi(numel(variables))};  % Choose random variable
        else
            expr = num2str(randn);  % Random constant (can be tuned)
        end
    else
        % Randomly choose and operator abd recursively generate operands
        operator = operators{randi(numel(operators))};
        left = genRandE(operators, variables, maxDepth-1);
        right = genRandE(operators, variables, maxDepth-1);
        expr = ['(', left, operator, right, ')']; % Construct binary tree expression
    end
end

function fitness = evalFitness(pop, data)
    numInd = length(pop);
    fitness = zeros(numInd, 1);

    % For each individual (expression), compute fitness
    for i = 1:numInd
        expression = pop{i};    % Get the symbolic expression
        fitness(i) = computeFitness(expression, data);  % Calculate error (fitness)
    end
end

function error = computeFitness(expression, data)
    % Convert expression into MATLAB function
    f = @(x) eval(expression);

    % Compute prediction for each data point
    actual = data(:,end);   % Last column as actual output
    predictions = arrayfun(@(row) f(data(row, 1:end-1)), 1:size(data,1));   % Evaluate function

    % Compute mean squared error
    error = mean((predictions - actual).^2);
end
