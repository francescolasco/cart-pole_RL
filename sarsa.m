clear all
close all
clc

% number of actions
A = 2;
% number of episodes
numEpisodes = 100000;
% exploration parameter
epsilon = 0.2;
% foresight parameter
gamma = 1;
% update parameter
alpha = 1e-2;

% provare 0.01 ma acc = 1
tt = 0.01;

% size of the state space
% X = [-1.2, 0.5];
% V = [-0.07, 0.07];
X = [-0.5, 0.5];
V = [-0.5, 0.5];

% parameters
M = 15; % number of cells per grid
N = 10; % number of grids

% dimension of the weight vector
d = (M+1)^2*N; 

% initialize the weigth vector
w = randn(d,A);

% construct grids
[cellX, cellV] = get_cells(X, V, M, N);

% total return
G = zeros(numEpisodes,1);

for e = 1:numEpisodes
    % initialize the episode 
    % s = [1/2*(X(2)+X(1));
    %     0];
    s = rand(2,1) - 0.5;

    disp(e)

    % get feature for initial state
    Fac = get_features(s, cellX, cellV, M, N); % indici delle featur attive, 1 per ogni griglia

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

    while ~isTerminal
        % take action a and observe sp and r
        
        [sp, r, isTerminal] = dinamica(s, a, tt, X, V);

        % update total return
        G(e) = G(e) + r;
        if isTerminal
            % impose that next value is 0, delta = r + gamma*Qp(ap) -
            % sum(w(Fac,a)) quindi gamma*Qp(ap) è 0
            delta = r - sum(w(Fac,a)); 
        else
            % get active features at next state
            Facp = get_features(sp, cellX, cellV, M, N);
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
    if mod(e,100) == 0
        epsilon = epsilon*0.995;
        disp(epsilon);
    end
end

%% plot
s = rand(2,1) - 0.5;

historyX = [s(1)];
historyV = [s(2)];

% get feature for initial state
Fac = get_features(s, cellX, cellV, M, N); % indici delle featur attive, 1 per ogni griglia

% get quality function
Q = sum(w(Fac,:));

a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q

% at the beginning is not terminal
isTerminal = false;

while ~isTerminal
    % take action a and observe sp and r
    
    [sp, r, isTerminal] = dinamica(s, a, tt, X, V);

    historyX = [historyX, sp(1)];
    historyV = [historyV, sp(2)];

    % update total return
    G(e) = G(e) + r;
    if isTerminal
        % impose that next value is 0, delta = r + gamma*Qp(ap) -
        % sum(w(Fac,a)) quindi gamma*Qp(ap) è 0
        delta = r - sum(w(Fac,a)); 
    else
        % get active features at next state
        Facp = get_features(sp, cellX, cellV, M, N);
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
