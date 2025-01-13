clear all %#ok<CLALL>
close all
clc
    
% numero azioni
A = 2;
% actions
actions = [-10 10];
% number of episodes
numEpisodes = 500;
% exploration parameter
epsilon = 1;
epsilonDecay = 0.9999; % ho raggiunto il miglior risultato con 0.9995
% foresight parameter
gamma = 0.99;

maxSteps = 500;

env = rlPredefinedEnv("CartPole-Discrete");

%% carico la rete neurale

load("dqnnet.mat");

targetNet = criticNet;

options = trainingOptions("adam", ...
    MaxEpochs=1, ...
    InitialLearnRate=0.01, ...
    Verbose=false);

experienceBufferSize = 10000;
batchSize = 250;
freqUpdate = 1;
targetUpdate = 4;

% inizializzo le code
experienceBufferS = zeros(4,experienceBufferSize);
experienceBufferY = zeros(2,experienceBufferSize);

%% inizio addestramento

% total return
G = zeros(numEpisodes,1);

% Loss
losses = zeros(numEpisodes,1);

m = 0; % contatore per riempimento code
k = 0; % contatore per frequenza aggiornamento rete 
j = 0; % contatore per frequenza aggiornamento rete target
for e = 1:numEpisodes
    fprintf('Episodio: %d\n', e);

    % Stato iniziale
    s = env.reset();

    % Prende la Q facendo uno step di forward-propagation nella rete
    Q = predict(targetNet,s');
        
    % All'inizio lo stato non è mai terminale
    isTerminal = 0;
    
    n = 0; % contatore per numero passi simulazione
    while ~isTerminal && n < maxSteps
        k = k + 1;
        n = n + 1;
        j = j + 1;

        % Prendo l'azione secondo il metodo epsilon-greedy
        if rand < epsilon
            a = randi(A);
        else
            a = find(Q == max(Q), 1, 'first');
        end

        action = actions(a);

        [sp, r, isTerminal, ~] = env.step(action);
       
        % plot(env);

        % ritorno complessivo
        G(e) = G(e) + r;
        
        % calcolo il target secondo la legge del DQN
        y = Q;
        if isTerminal
            y(a) = r;   
        else
            Qp = predict(targetNet,sp');
            y(a) = r + gamma*max(Qp);            
        end
        
        losses(e) = losses(e) + norm(y-Q,2);
        
        % accumulo le informazioni in una coda (accesso circolare)
        experienceBufferS(:,mod(m,experienceBufferSize)+1) = s;
        experienceBufferY(:,mod(m,experienceBufferSize)+1) = y;

        m = m + 1;

        % aggiorno la rete ogni freqUpdate passi ma solo se prima ho 
        % esperienze a sufficienza
        if m >= batchSize && (k >= freqUpdate)
            % estraggo batchSize campioni casuali dalla coda per evitare la
            % correlazione fra campioni contigui
            index = randperm(min(m,experienceBufferSize),batchSize);
            S = experienceBufferS(:,index);
            Y = experienceBufferY(:,index);

            % addestro la rete sui campioni casuali
            criticNet = trainnet(S', Y', criticNet, 'mse', options);
            % criticNet = trainNetwork(S', Y', criticNet.Layers, options);
          
            % resetto l'indice di batch
            k = 0;
        end
        
        % aggiorno la rete target, copiando i pesi a quella critic
        if mod(j,targetUpdate) == 0
            targetNet.Learnables.Value = criticNet.Learnables.Value;
        end
        
        s = sp;
        Q = Qp;
        
        % ho raggiunto un buon risultato combinando min 0.1 e decay 0.9995
        epsilon = max(0.01, epsilon * epsilonDecay); 
    end
    
    fprintf('Reward cumulativo: %d\n', G(e));
    fprintf('-----\n\n');

    if G(e) == 500
        disp("Terminazione: convergenza raggiunta");
        break;
    end
end

% save('net.mat', 'net');

%% plot
close all

s = env.reset();
for i=1:1000
    
    Q = predict(criticNet,s');
    a = find(Q == max(Q), 1, 'first');
    action = actions(a);

    [sp, ~, isTerminal, ~] = env.step(action);

    plot(env);

    

    s = sp;

    % Interrompe la simulazione se l'episodio è finito
    if isTerminal
        break;
    end
end