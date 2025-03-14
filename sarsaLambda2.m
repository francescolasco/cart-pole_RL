clear all 
close all
clc
    
% numero azioni
A = 2;
% actions
actions = [-10 10];
% number of episodes
numEpisodes = 25000;
% exploration parameter
epsilon = 1;
epsilonDecay = 0.99999;
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

% initialize eligibility traces
et = zeros(d,A); % tante quante le coppie feature/stato

% construct grids
[cellX, cellV, cellTHETA, cellOMEGA] = get_cells(X, V, THETA, OMEGA, M, N);

% total return
G = zeros(numEpisodes,1);

mean = 0;
tau = 0.95;
means = zeros(numEpisodes,1);
lambda = 0.8; % provare altri valori di lambda

%% addestramento
counter = 0;
rng(4321);
for e = 1:numEpisodes
    
    % initialize eligibility traces
    et = zeros(d,A); % tante quante le coppie feature/stato

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
            % update eligibility traces
            et(Fac,a) = et(Fac,a) + 1; %accumulating traces
            
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

            et = gamma*lambda*et;
        end
        % update weigth vector
        w = w + alpha*delta*et;
                
        if ~isTerminal
            % update state, action and features
            s = sp;
            a = ap;
            Fac = Facp;
        end

        epsilon = max(0.01, epsilon * epsilonDecay);
    end

    if G(e) == 500
        counter = counter + 1;
    else
        counter = 0;
    end

    mean = (1-tau) * G(e) + tau * mean;
    means(e) = mean;

    fprintf('Reward cumulativo: %d\n', G(e));
    fprintf('Reward cumulativo medio: %f\n', mean);
    fprintf('Epsilon: %f\n', epsilon);
    fprintf('-----\n\n');
    
    % if means(e) >= 0.9*maxSteps % metto una tolleranza
    %     break;
    % end
    
    if counter >= 500
        break;
    end
     
end

hold on
plot(G(1:e));
plot(means(1:e));
hold off

save("w_sarsalambda.mat","w");

%% plot

close all

load("w_sarsalambda.mat");
rng(1)

s = env.reset();

% while true
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


%%
[xx, yy] = meshgrid(THETA(1):0.01:THETA(2),OMEGA(1):0.1:OMEGA(2));

zz = zeros(size(xx));
for i = 1:size(xx,1)
    for j = 1:size(xx,2)
        zz(i,j) = max(sum(w(get_features([0,0,xx(i,j),yy(i,j)], cellX, cellV, cellTHETA, cellOMEGA, M, N),:)));
    end
end

surf(xx,yy,zz)

%%
thetatheta = THETA(1):0.01:THETA(2);

act = zeros(size(thetatheta));
for i = 1:length(thetatheta)
    Fac = get_features([0,0,thetatheta(i),0], cellX, cellV, cellTHETA, cellOMEGA, M, N);
    Q = sum(w(Fac,:));
    a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q
    act(i) = actions(a);
end

plot(thetatheta,act,'r')
title("Azione di Controllo nell'origine")
ylabel("u")
xlabel("theta")
%%
xx = X(1):0.01:X(2);

act = zeros(size(xx));
for i = 1:length(xx)
    Fac = get_features([xx(i),0,0,0], cellX, cellV, cellTHETA, cellOMEGA, M, N);
    Q = sum(w(Fac,:));
    a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q
    act(i) = actions(a);
end

plot(xx,act,'r')
title("Azione di Controllo nell'origine")
ylabel("u")
xlabel("x")
%%

[xx, thetatheta] = meshgrid(X(1):0.01:X(2),THETA(1):0.1:THETA(2));

act = zeros(size(xx));
for i = 1:size(xx,1)
    for j = 1:size(xx,2)
        Fac = get_features([xx(i,j),0,thetatheta(i,j),0], cellX, cellV, cellTHETA, cellOMEGA, M, N);
        Q = sum(w(Fac,:));
        a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q
        act(i,j) = actions(a);
    end
end

surf(xx,thetatheta,act)