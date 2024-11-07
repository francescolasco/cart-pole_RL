clear all
close all
clc

% number of actions
A = 2;
% number of episodes
numEpisodes = 10000;
% exploration parameter
epsilon = 0.3;
% foresight parameter
gamma = 1;
% update parameter
alpha = 1e-2;

% parametri del sistema dinamico
mm = 1;
MM = 5;
L = 2;
g = -10;
dd = 1;

Ts = 0.001;

% size of the state space
X = [-1 1];
V = [-2 2];
THETA = [pi/2 - 0.2 pi/2 + 0.2];
OMEGA = [-1.5 1.5];

% parameters
M = 15; % number of cells per grid
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

s0 = [0; 0; pi/2 + 0.1; 0];

for e = 1:numEpisodes
    % initialize the episode 
    %s = rand(4,1) - 0.5;
    s = s0; % per ora parto sempre dallo stesso stato

    disp(e)

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
    isTerminal = false;
    
    % k = 0;
    while ~isTerminal
        % stampo lo stato
        % if mod(k,1000) == 0
        %     disp(s);
        % end
        % k = k + 1;

        % take action a and observe sp and r
        [sp, r, isTerminal] = dinamica(s, s0, mm, MM, L, g, dd, a, Ts, X, V, THETA, OMEGA);

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

    % if mod(e,100) == 0
    %     epsilon = epsilon*0.995;
    %     disp(epsilon);
    % end
end

%% plot
s = [0; 0; pi/2 + 0.05; 0];

historyX = [s(1)];
historyV = [s(2)];
historyTHETA = [s(3)];
historyOMEGA = [s(4)];

% get feature for initial state
Fac = get_features(s, cellX, cellV, cellTHETA, cellOMEGA, M, N); % indici delle featur attive, 1 per ogni griglia

% get quality function
Q = sum(w(Fac,:));

a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q

% at the beginning is not terminal
isTerminal = false;

while ~isTerminal
    % take action a and observe sp and r
    
    [sp, r, isTerminal] = dinamica(s, mm, MM, L, g, dd, a, Ts, X, V, THETA, OMEGA);

    historyX = [historyX, sp(1)];
    historyV = [historyV, sp(2)];
    historyTHETA = [historyTHETA, sp(3)];
    historyOMEGA = [historyOMEGA, sp(4)];

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

figure(1);
plot(historyX,historyV);
axis([X V]);

figure(2);
plot(historyTHETA,historyOMEGA);
axis([THETA OMEGA]);

