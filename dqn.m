clear all
close all
clc

% number of actions
A = 2;
% number of episodes
numEpisodes = 5000;
% exploration parameter
epsilon = 0.5;
% foresight parameter
gamma = 0.95;

maxSteps = 1000;

% parametri del sistema
mm = 0.5;
MM = 1;
L = 1.5;
g = -9.81;

% Dimensione dello spazio di stato
X = [-5 5];
V = [-25 25];
THETA = [pi - pi/4 pi + pi/4];
OMEGA = [-20 20];

% Il tempo di campionamento dev'essere abbastanza basso per avere una
% simulazione liscia, e abbastanza alto per  non rallentare troppo gli
% episodi. Se fosse troppo alto, si rischia di saltare il punto di
% equilibrio.
Ts = 0.025;

replayBufferSize = 5000;
batchSize = 100;

inputLayer = 4;
layer1 = 50;
layer2 = 25;
outputLayer = 2;

% % costruisco rete neurale
net = network;

% Imposta il numero di input e output
net.numInputs = 1;
net.numLayers = 3;
net.inputConnect = [1; 0; 0];
net.layerConnect = [0 0 0; 1 0 0; 0 1 0];
net.outputConnect = [0 0 1]; 

net.input.size = inputLayer;
net.layers{1}.dimensions = layer1;
net.layers{2}.dimensions = layer2;
net.layers{3}.dimensions = outputLayer;

% Funzioni di attivazione (poslin sarebbe ReLU)
net.layers{1}.transferFcn = 'poslin'; 
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'purelin'; 

net.performFcn = 'mse'; % imposta mse come metrica d'errore

net.trainFcn = 'traingd'; % imposta gradiente come alg. di ott.
net.trainParam.showWindow = false;  % Disattiva la finestra di riepilogo
net.trainParam.lr = 0.05;

% Inizializzazione di He dei pesi
net.IW{1} = sqrt(2 / inputLayer) * randn(layer1,inputLayer);
net.LW{2,1} = sqrt(2 / layer1) * randn(layer2,layer1);
net.LW{3,2} = sqrt(2 / layer2) * randn(outputLayer,layer2);

net_target = net;  % Copia della rete principale
update_target_steps = 50;  % Ogni 50 episodi aggiorna net_target

% Costruisco rete neurale con altro metodo (MA NON CAMBIA NIENTE)
% layers = [
%     featureInputLayer(4)
%     fullyConnectedLayer(50)
%     reluLayer
%     fullyConnectedLayer(50)
%     reluLayer
%     fullyConnectedLayer(2)];
% 
% options = trainingOptions('rmsprop', ...
%     'InitialLearnRate', 0.0005, ...
%     'ExecutionEnvironment', 'parallel-gpu', ...
%     'Verbose', false);
% 
% net = dlnetwork(layers);

%% inizio addestramento

% inizializzo le code
replayBufferS = zeros(inputLayer,replayBufferSize);
replayBufferY = zeros(outputLayer,replayBufferSize);

% questo contatore indica quanto è riempita la coda. All'inizio è vuota
m = 0;

% warning('off', 'all');
s0 = [0; 0; pi + 0.05; 0];

for e = 1:numEpisodes
    disp(e);

    % ogni 10 episodi vedo come sta andando
    if mod(e,10) == 0
        % Stato iniziale
        s = s0;
        while true
            sNorm = normalize(s);
            Q = net(sNorm);
            a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q
        
            sp = dinamica(s, mm, MM, L, g, a, Ts);
            sp = sp(:);
        
            if sp(1) < X(1) || sp(1) > X(2) || sp(2) < V(1) || sp(2) > V(2) || sp(3) < THETA(1) || sp(3) > THETA(2) || sp(4) < OMEGA(1) || sp(4) > OMEGA(2)
                break;
            end
        
            drawpend(sp,mm,MM,L);
        
            % update state
            s = sp;
        end
    end

    % Stato iniziale
    s = s0;

    % normalizzo lo stato in valori compresi fra -1 e 1
    sNorm = normalizeState(s);

    % Prende la Q facendo uno step di forward-propagation nella rete
    Q = net(sNorm); 
        
    % All'inizio lo stato non è mai terminale
    isTerminal = 0;
    
    % Questi sono contatori che servono per gestire le strutture di dati 
    % usate per memorizzare gli stati visitati
    k = 0;
    n = 0;
    elapsed = 0;
    while ~isTerminal && n < maxSteps
        k = k + 1;
        n = n + 1;

        % Prendo l'azione secondo il metodo epsilon-greedy
        if rand < epsilon
            a = randi(A);
        else
            a = find(Q == max(Q), 1, 'first');
        end

        % Faccio un passo della dinamica e prendo lo stato successivo sp
        sp = dinamica(s, mm, MM, L, g, a, Ts);
        sp = sp(:);
        spNorm = normalize(sp);
         
        % Questo è un modo per mostrare a schermo il cart-pole senza 
        % rallentare troppo la simulazione
        if mod(k,2) == 0
            % drawpend(sp,mm,MM,L);
        end
        
        % assegno il reward nello stato terminale
        % if ((sp(3)-pi)^2 + sp(4)^2) < 0.001
        %     isTerminal = 1;
        %     r = 0;
        % else
        %     isTerminal = 0;
        %     r = -10;
        % end

        r = -10;
        isTerminal = 0;
        
        % se raggiungo lo stato d'equilibrio, assegno reward positivo
        if ((sp(3)-pi)^2 + sp(4)^2) < 0.001
            r = 1;
        end
        
        % se raggiungo una configurazione fuori dallo spazio di stato,
        % termino l'episodio
        if sp(1) < X(1) || sp(1) > X(2) || sp(2) < V(1) || sp(2) > V(2) || sp(3) < THETA(1) || sp(3) > THETA(2) || sp(4) < OMEGA(1) || sp(4) > OMEGA(2)
            isTerminal = 1;
        end

        % calcolo il target secondo la legge del DQN
        y = Q;
        % Qp = net(spNorm);
        Qp = net_target(spNorm);
        y(a) = r + gamma*max(Qp);
        
        % accumulo le informazioni in una coda (accesso circolare)
        replayBufferS(:,mod(m,replayBufferSize)+1) = sNorm;
        replayBufferY(:,mod(m,replayBufferSize)+1) = y;

        m = m + 1;

        % aggiorno la rete ogni batchSize passi oppure quando raggiungo lo
        % stato terminale, ma solo se prima ho esperienze a sufficienza
        if m >= batchSize && (isTerminal || k == batchSize)
            % disp('updating network');

            % estraggo batchSize campioni casuali dalla coda per evitare la
            % correlazione fra campioni contigui
            index = randperm(min(m,replayBufferSize),batchSize);
            S = replayBufferS(:,index);
            Y = replayBufferY(:,index);
            
            
            tic;
            
            % Questo lo facevo per velocizzare, ma genera un errore e non
            % ho capito il perchè. Praticamente sposto manualmente i dati
            % su gpu
            % S = dlarray(s, 'CB');  % Converti lo stato in deep learning array
            % S = gpuArray(S);       % Sposta l'array sulla GPU
            % Y = dlarray(Y, 'CB');
            % Y = gpuArray(Y);

            % addestro la rete sui campioni casuali, usando calcolo
            % parallelo sui core della sola gpu
            net = train(net, S, Y);
          
            % resetto l'indice di batch
            k = 0;

            disp(toc);
        end

        if mod(e, update_target_steps) == 0
            net_target = net;
        end

        s = sp;
        sNorm = spNorm;
        Q = Qp;
    end
       
    epsilon = max(0.05, epsilon * 0.9995);
end

save('net.mat', 'net');

%% plot
close all

% load("net.mat");

% s0 = [0; 0; pi + ((2 * rand * (pi/6)) - (pi/6)); 0];
s0 = [0; 0; pi + 0.05; 0];
s = s0;

while true
    sNorm = normalize(s);
    Q = net(sNorm);
    a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q

    sp = dinamica(s, mm, MM, L, g, a, Ts);
    sp = sp(:);

    if sp(1) < X(1) || sp(1) > X(2) || sp(2) < V(1) || sp(2) > V(2) || sp(3) < THETA(1) || sp(3) > THETA(2) || sp(4) < OMEGA(1) || sp(4) > OMEGA(2)
        sp = s0;
    end

    drawpend(sp,mm,MM,L);

    % update state
    s = sp;
end
