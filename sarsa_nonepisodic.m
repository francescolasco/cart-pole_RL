clear all
close all
clc

% number of actions
A = 2;
% number of episodes
numEpisodes = 5000;
maxTime = 50000;
% exploration parameter
epsilon = 0.5;
% foresight parameter
gamma = 0.99;
% update parameter
alpha = 1e-1;

% parametri del sistema
mm = 0.5;
MM = 2.5;
L = 2;
g = -9.81;
dd = 1;

Ts = 0.005;

% size of the state space
X = [-5 5];
V = [-25 25];
THETA = [pi - pi/4 pi + pi/4];
OMEGA = [-25 25];

% parameters
M = 10; % number of cells per grid
N = 10; % number of grids

% dimension of the weight vector
d = (M+1)^4*N; % N ipercubi sovrapposti

% VALUTARE SE CONVIENE FARE GLI IPERCUBI CON LATI DI DIMENSIONI DIVERSE

% initialize the weigth vector
w = randn(d,A);

% construct grids
[cellX, cellV, cellTHETA, cellOMEGA] = get_cells(X, V, THETA, OMEGA, M, N);

% total return
G = zeros(numEpisodes,1);

for e = 1:numEpisodes
    disp(e);

    % initialize the episode: parto da condizioni iniziali distinte per
    % velocizzare l'apprendimento
    s0 = [(rand*2*X(2) - X(2)) / 5; (rand*2*V(2) - V(2)) / 5; pi + ((2 * rand * (pi/4)) - (pi/4)); (rand*2*OMEGA(2) - OMEGA(2)) / 5];
    % s0 = [0; 0; pi + pi/8; 0];
    s = s0;

    % get feature for initial state
    Fac = get_features(s, cellX, cellV, cellTHETA, cellOMEGA, M, N); % indici delle featur attive, 1 per ogni griglia

    % get quality function
    Q = sum(w(Fac,:));

    % take epsilon greedy action
    if rand < epsilon
        a = randi(A); % take random action
    else
        a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q
    end

    % at the beginning is not terminal
    isTerminal = 0;

    for k = 1:maxTime
        % disp(k)
        [sp, r] = dinamica(s, s0, mm, MM, L, g, dd, a, Ts, X, V, THETA, OMEGA);

        % drawpend(sp,mm,MM,L);

        % get active features at next state
        Facp = get_features(sp, cellX, cellV, cellTHETA, cellOMEGA, M, N);
        % compute next q function
        Qp = sum(w(Facp,:));
        % take epsilon greedy action
        if rand < epsilon
            ap = randi(A); % take random action
        else
            ap = find(Qp == max(Qp), 1, 'first'); % take greedy action 
        end
        % compute temporal difference error
        delta = r + gamma*Qp(ap) - sum(w(Fac,a));
 
        % update weigth vector
        w(Fac,a) = w(Fac,a) + alpha*delta;
        
        % update state, action and features
        s = sp;
        a = ap;
        Fac = Facp;
    end

    epsilon = max(0.2, epsilon * 0.9975);
end

save("w.mat","w");

%% plot
load("w.mat");

historyX = [s(1)];
historyV = [s(2)];
historyTHETA = [s(3)];
historyOMEGA = [s(4)];

% get feature for initial state
Fac = get_features(s, cellX, cellV, cellTHETA, cellOMEGA, M, N); % indici delle featur attive, 1 per ogni griglia

% get quality function
Q = sum(w(Fac,:));

a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q

s0 = [0; 0; pi + ((2 * rand * (pi/4)) - (pi/4)); 0];
% s0 = [0; 0; pi + pi/8; 0];
s = s0;

while true
    % take action a and observe sp and r
    [sp, ~] = dinamica(s, s0, mm, MM, L, g, dd, a, Ts, X, V, THETA, OMEGA);

    drawpend(sp,mm,MM,L);

    historyX = [historyX, sp(1)];
    historyV = [historyV, sp(2)];
    historyTHETA = [historyTHETA, sp(3)];
    historyOMEGA = [historyOMEGA, sp(4)];

    % get active features at next state
    Facp = get_features(sp, cellX, cellV, cellTHETA, cellOMEGA, M, N);
    % compute next q function
    Qp = sum(w(Facp,:));
    % take greedy action
    ap = find(Qp == max(Qp), 1, 'first'); % take greedy action 
    
    % update state, action and features
    s = sp;
    a = ap;
    Fac = Facp;

end

figure(1);
plot(historyX,historyV);
axis([X V]);

figure(2);
plot(historyTHETA,historyOMEGA);
axis([THETA OMEGA]);