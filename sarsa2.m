clear all
close all
clc
% training log
% 10 k ep lim(2m 0.07rad) eps=.1 steps=1k u = +-20
% 50 k ep lim(2m 0.07rad) eps=.05 steps=2k u = +-20 ->w2
% 100 k ep lim(2m 0.07rad) eps=.9g steps=2k u = +-10 

% number of actions
A = 2;
% number of episodes
numEpisodes = 200000;
% exploration parameter
epsilon = 0.05;
% foresight parameter
gamma = 1;
% update parameter
alpha = 1e-2;

% parametri del sistema
mm = 0.5;
MM = 1;
L = 1.5;
g = -9.81;
dd = 1; % questo non serve piu nel nuovo modello

% Il tempo di campionamento dev'essere abbastanza basso per avere una
% simulazione liscia, e abbastanza alto per  non rallentare troppo gli
% episodi. Se fosse troppo alto, si rischia di saltare il punto di
% equilibrio.
Ts = 0.025; 

% size of the state space
X = [-5 5];
V = [-20 20];
marg = 0.05;
THETA = [pi-pi/6 pi+pi/6];
OMEGA = [-20 20];

% parameters
M = 10; % number of cells per grid
N = 10; % number of grids

% dimension of the weight vector
d = (M+1)^4*N; % N ipercubi sovrapposti

% VALUTARE SE CONVIENE FARE GLI IPERCUBI CON LATI DI DIMENSIONI DIVERSE

% initialize the weigth vector
% w = randn(d,A);
load("w2.mat");

% construct grids
[cellX, cellV, cellTHETA, cellOMEGA] = get_cells(X, V, THETA, OMEGA, M, N);

% total return
G = zeros(numEpisodes,1);

maxSteps = 4000;

%%

for e = 1:numEpisodes
    s0 = [ 0; 0; pi + ((2 * rand * marg) - marg); 0];
    s = s0;
    
    disp(e);
    % disp(s0);

    % get feature for initial state
    Fac = get_features(s, cellX, cellV, cellTHETA, cellOMEGA, M, N); % indici delle featur attive, 1 per ogni griglia

    % get quality function
    Q = sum(w(Fac,:));

    % take epsilon greedy actions
    if rand < epsilon
        a = randi(A); % take random action
    else
        a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q
    end

    % at the beginning is not terminal
    isTerminal = 0;
    steps = 0;
    
    while (~isTerminal) && (steps <= maxSteps)
        % take action a and observe sp and r
        [sp, r, isTerminal] = dinamica2(s, s0, mm, MM, L, g, dd, a, Ts, X, V, THETA, OMEGA);
        steps = steps + 1;
        % update total return
        G(e) = G(e) + r;
        if isTerminal
            % impose that next value is 0, delta = r + gamma*Qp(ap) -
            % sum(w(Fac,a)) quindi gamma*Qp(ap) è 0
            delta = r - sum(w(Fac,a)); 
        else
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
        end
        % update weigth vector
        w(Fac,a) = w(Fac,a) + alpha*delta;
        
        if ~isTerminal
            % update state, action and features
            s = sp;
            a = ap;
            Fac = Facp;
        end
    end
    % if(mod(e,200) == 0)
    %     epsilon = epsilon*0.99;
    % end
       
    % epsilon = max(0.2, epsilon * ((numEpisodes - 2.5)/numEpisodes));
end

save("w.mat","w");

%% plot
load("w2.mat");

s0 = [ 0; 0; pi + ((2 * rand * 0.05) - 0.05); 0];
s = s0;

historyX = [s(1)];
historyV = [s(2)];
historyTHETA = [s(3)];
historyOMEGA = [s(4)];

% get feature for initial state
Fac = get_features(s, cellX, cellV, cellTHETA, cellOMEGA, M, N); % indici delle feature attive, 1 per ogni griglia

% get quality function
Q = sum(w(Fac,:));

a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q

% at the beginning is not terminal
isTerminal = false;

steps = 0;
maxSteps = 500;
while steps < maxSteps
    [sp, r, isTerminal] = dinamica2(s, s0, mm, MM, L, g, dd, a, Ts, X, V, THETA, OMEGA, 0);

    drawpend(sp,mm,MM,L);

    disp(sp);

    historyX = [historyX, sp(1)];
    historyV = [historyV, sp(2)];
    historyTHETA = [historyTHETA, sp(3)];
    historyOMEGA = [historyOMEGA, sp(4)];

    Facp = get_features(sp, cellX, cellV, cellTHETA, cellOMEGA, M, N);
    % compute next q function
    Qp = sum(w(Facp,:));
    % take greedy action
    ap = find(Qp == max(Qp), 1, 'first');
    % compute temporal difference error
    delta = r + gamma*Qp(ap) - sum(w(Fac,a));
    % update weigth vector
    w(Fac,a) = w(Fac,a) + alpha*delta;

    % update state, action and features
    s = sp;
    a = ap;
    Fac = Facp;
    steps = steps + 1;
end

figure(1);
plot(historyX,historyV);
axis([X V]);

figure(2);
plot(historyTHETA,historyOMEGA);
axis([THETA OMEGA]);
