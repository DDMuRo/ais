%% Calculating torques in function of pedal forces
% [25/09/2024]

close all; clear; clc;

% Data

kint = importdata('P01rra_cycling_Kinematics_q.sto'); 

kin = kint.data;
headers = kint.colheaders;

kphix = find(contains(headers, 'knee_angle'));

kphir = kin(:,kphix(1));

p1 = find(islocalmax(kphir)==1, 1, 'first');
tk = kin(1:p1,1);

figure(1)
plot(tk, kphir(1:p1))


%%

P01t = readtable('P01pedal_force_data_filtered.csv');

P01 = table2array(P01t);

time = P01(:,1);
[n,~] = size(P01);

t1 = find(time >= tk(1), 1, 'first');
t2 = find(time >= tk(end), 1, 'first');
t = t1:t2-1;

theta = linspace(0, 360, length(t));
theta = theta*(pi/180);

%%

xright = P01(t,2);
yright = P01(t,3);

xleft = P01(t,8);
yleft = P01(t,9);



% figure(1)
% subplot(2,1,1)
% plot(xright)
% subplot(2,1,2)
% plot(yright)
% 
% figure(2)
% subplot(2,1,1)
% plot(xleft)
% subplot(2,1,2)
% plot(yleft)

r = .175;

fr = 450;

%% Equation

Txr      = xright.*r.*cos(theta)';

Tyr      = yright.*r.*sin(theta)';
Tcrankr  = sqrt(Txr.^2 + Tyr.^2);

Txl      = xleft.*r.*cos(theta)';
Tyl      = yleft.*r.*sin(theta)';
Tcrankl  = sqrt(Txl.^2 + Tyl.^2);

Tcrank   = Tcrankr+Tcrankl;

%%
figure(2)
plot(Tcrank)

