clear all %#ok<CLALL>
close all
clc
    
% numero azioni
A = 2;
% actions
actions = [-10 10];
% number of episodes
numEpisodes = 5000;
% exploration parameter
epsilon = 1;
epsilonDecay = 0.9999;
% foresight parameter
gamma = 0.999;
% update parameter
alpha = 1e-2;

maxSteps = 500;

env = rlPredefinedEnv("CartPole-Discrete");

% size of the state space
X = [-2.41 2.41];
V = [-20 20];
THETA = [-0.2095 0.2095];
OMEGA = [-20 20];

% parameters
M = 10; % number of cells per grid
N = 10; % number of grids

% dimension of the weight vector
d = (M+1)^4*N; % N ipercubi sovrapposti

% VALUTARE SE CONVIENE FARE GLI IPERCUBI CON LATI DI DIMENSIONI DIVERSE

% initialize the weigth vector
w = randn(d,A);
% load("w.mat");

% construct grids
[cellX, cellV, cellTHETA, cellOMEGA] = get_cells(X, V, THETA, OMEGA, M, N);

% total return
G = zeros(numEpisodes,1);

mean = 0;
tau = 0.95;
means = zeros(numEpisodes,1);

for e = 1:numEpisodes
    fprintf('Episodio: %d\n', e);

    % Stato iniziale
    s = env.reset();

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

    step = 0;
    
    while ~isTerminal && step < maxSteps
        step = step + 1;
        
        action = actions(a);
        [sp, r, isTerminal, ~] = env.step(action);

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

        epsilon = max(0.01, epsilon * epsilonDecay);
    end

    mean = (1-tau) * G(e) + tau * mean;
    means(e) = mean;

    fprintf('Reward cumulativo: %d\n', G(e));
    fprintf('Reward cumulativo medio: %f\n', mean);
    fprintf('Epsilon: %f\n', epsilon);
    fprintf('-----\n\n');
    
    if means(e) >= 0.9*maxSteps % metto una tolleranza
        break;
    end
end

hold on
plot(G);
plot(means);
hold off

% save("w.mat","w");

%% plot

close all

s = env.reset();

for i=1:maxSteps
    % get feature for initial state
    Fac = get_features(s, cellX, cellV, cellTHETA, cellOMEGA, M, N);
    
    % get quality function
    Q = sum(w(Fac,:));
    
    a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q
    action = actions(a);

    [sp, ~, isTerminal, ~] = env.step(action);

    plot(env);

    % update state, action and features
    s = sp;

    % Interrompe la simulazione se l'episodio è finito
    if isTerminal
        break;
    end
end