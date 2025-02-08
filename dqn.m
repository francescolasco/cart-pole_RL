clear all %#ok<CLALL>
close all
clc
    
% numero azioni
numActions = 2;
% actions
actions = [-10 10];
% number of episodes
numEpisodes = 1000;
% exploration parameter
epsilon = 1;
epsilonDecay = 0.999;
% foresight parameter
gamma = 0.999;
% metric parameters (mean)
mean = 0;
alpha = 0.95;

maxSteps = 500;

env = rlPredefinedEnv("CartPole-Discrete");

%% carico la rete neurale

% iperparametri
experienceBufferSize = 10000;
batchSize = 128;
lr = 0.001;
lrDecay = 0.999;
freqUpdate = 1;
targetUpdate = 4;
% softUpdate = 0;
% tau = 0.1;

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
initOpts = rlAgentInitializationOptions(NumHiddenUnit=128);
agent = rlDQNAgent(obsInfo,actInfo,initOpts);

criticNet = getModel(getCritic(agent));
targetNet = criticNet;

% inizializzo le code per mantenere le esperienze
experienceBufferS = zeros(4,experienceBufferSize);
experienceBufferA = zeros(1,experienceBufferSize);
experienceBufferR = zeros(1,experienceBufferSize);
experienceBufferSp = zeros(4,experienceBufferSize);
experienceBufferT = zeros(1,experienceBufferSize);

% target
y = zeros(1,batchSize);

% momenti per adam
averageGrad = [];
averageSqGrad = [];

%% inizio addestramento

% total return
G = zeros(numEpisodes,1);
means = zeros(numEpisodes,1);

% Loss
losses = zeros(numEpisodes,1);

m = 0; % contatore per riempimento code e step totali dall'inizio
for e = 1:numEpisodes
    fprintf('Episodio: %d\n', e);

    % Stato iniziale
    s = env.reset();
 
    % All'inizio lo stato non è mai terminale
    isTerminal = 0;
    
    n = 0; % contatore per numero passi simulazione
    while ~isTerminal && n < maxSteps
        n = n + 1;

        % Prende la Q facendo uno step di forward-propagation nella rete
        Q = forward(criticNet,dlarray(s,'CB'));

        % Prendo l'azione secondo il metodo epsilon-greedy
        if rand < epsilon
            a = randi(numActions);
        else
            [~,a] = max(Q);
        end
        action = actions(a);

        [sp, r, isTerminal, ~] = env.step(action);
       
        % plot(env);

        % ritorno complessivo
        G(e) = G(e) + r;
        
        % accumulo le informazioni in una coda (accesso circolare)
        experienceBufferS(:,mod(m,experienceBufferSize)+1) = s;
        experienceBufferA(:,mod(m,experienceBufferSize)+1) = a;
        experienceBufferR(:,mod(m,experienceBufferSize)+1) = r;
        experienceBufferSp(:,mod(m,experienceBufferSize)+1) = sp;
        experienceBufferT(:,mod(m,experienceBufferSize)+1) = isTerminal;
        
        m = m + 1;
        % aggiorno la rete ogni freqUpdate passi ma solo se prima ho 
        % esperienze a sufficienza
        if m >= batchSize
            % estraggo batchSize campioni casuali dalla coda per evitare la
            % correlazione fra campioni contigui
            index = randperm(min(m,experienceBufferSize),batchSize);
            S = experienceBufferS(:,index); % stati
            A = experienceBufferA(:,index); % azioni
            R = experienceBufferR(:,index); % rewards
            Sp = experienceBufferSp(:,index); % stati successivi
            T = experienceBufferT(:,index); % flag stati terminali
            
            % Calcolo la Q per gli stati successivi
            Qp = forward(targetNet,dlarray(Sp,'CB'));
            
            % Aggiorno il target
            y(:,T==1) = R(:,T==1);
            y(:,T==0) = R(:,T==0) + gamma * max(Qp(:,T==0));
            
            %y = gpuArray(dlarray(y,'CB'));
            %S = gpuArray(dlarray(S,'CB'));
            y = dlarray(y,'CB');
            S = dlarray(S,'CB');

            % Aggiorno la rete
            colIndex = linspace(1,batchSize,batchSize);
            %indices = gpuArray(dlarray(A + (colIndex-1)*2,'CB'));
            indices = dlarray(A + (colIndex-1)*2,'CB');
            [loss,g] = dlfeval(@optimize,criticNet,S,y,indices);

            losses(e) = losses(e) + loss;
            
            [criticNet,averageGrad,averageSqGrad] = adamupdate(criticNet,g,averageGrad,averageSqGrad,m,lr);
        end

        if mod(m,targetUpdate) == 0
            targetNet.Learnables.Value = criticNet.Learnables.Value;
        end

        s = sp;
           
        epsilon = max(0.01, epsilon * epsilonDecay); 
    end

    mean = (1-alpha) * G(e) + alpha * mean;
    means(e) = mean;

    losses(e) = losses(e) / n;

    fprintf('Reward cumulativo: %d\n', G(e));
    fprintf('Reward cumulativo medio: %f\n', mean);
    fprintf('Loss: %f\n', losses(e));
    fprintf('epsilon: %f\n', epsilon);
    fprintf('learning-rate: %f\n', lr);
    fprintf('-----\n\n');

    if G(e) >= maxSteps
        break;
    end
end

% save('net.mat', 'net');

%% test
close all

load("trainednet.mat");

s = env.reset();
for i=1:maxSteps
    
    Q = predict(criticNet,s');
    [~,a] = max(Q);
    action = actions(a);

    [sp, ~, isTerminal, ~] = env.step(action);

    plot(env);

    % pause(0.05);

    s = sp;

    % Interrompe la simulazione se l'episodio è finito
    if isTerminal
        break;
    end
end

%% optimization function

function [f,g] = optimize(net,S,y,indices)
% Calculate objective using supported functions for dlarray
    Q = forward(net,S);
    % questo mi seleziona solamente le righe corrispondenti alle azioni
    % effettivamente scelte in un dato stato
    f = mse(dlarray(Q(indices),'CB'),y);
    g = dlgradient(f,net.Learnables);
end