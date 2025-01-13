clear all
close all
clc

% number of actions
A = 2;
% number of episodes
numEpisodes = 10000;
% exploration parameter
epsilon = 1;
% foresight parameter
gamma = 1;

maxSteps = 1000;

% parametri del sistema
mm = 0.5;
MM = 1;
L = 1.5;
g = -9.81;

% Dimensione dello spazio di stato
X = [-5 5];
V = [-25 25];
THETA = [pi - pi/8 pi + pi/8];
OMEGA = [-20 20];

% Il tempo di campionamento dev'essere abbastanza basso per avere una
% simulazione liscia, e abbastanza alto per  non rallentare troppo gli
% episodi. Se fosse troppo alto, si rischia di saltare il punto di
% equilibrio.
Ts = 0.02;

%% Definisco un polinomio di 3 grado completo

% PROVARE UN POLINOMIO DI 3 GRADO COMPLETO, CI SONO POCHI PARAMETRI!
% modelfun1 = @(a,x) a(1) + a(2)*x(:,1) + a(3)*x(:,2) + a(4)*x(:,3) + a(5)*x(:,4) ...                 
%     + a(6)*x(:,1).*x(:,1) + a(7)*x(:,1).*x(:,2) + a(8)*x(:,1).*x(:,3) + a(9)*x(:,1).*x(:,4) ...
%     + a(10)*x(:,2).*x(:,2) + a(11)*x(:,2).*x(:,3) + a(12)*x(:,2).*x(:,4) ...
%     + a(13)*x(:,3).*x(:,3) + a(14)*x(:,3).*x(:,4) ...
%     + a(15)*x(:,4).*x(:,4) ...
%     + a(16)*x(:,1).^3 + a(17)*x(:,1).*x(:,1).^2.*x(:,2) + a(18)*x(:,1).^2.*x(:,3) + a(19)*x(:,1).^2.*x(:,4) ...
%     + a(20)*x(:,1).*x(:,2).^2 + a(21)*x(:,1).*x(:,2).*x(:,3) + a(22)*x(:,1).*x(:,2).*x(:,4) ...
%     + a(23)*x(:,1).*x(:,3).^2 + a(24)*x(:,1).*x(:,3).*x(:,4) + a(25).*x(:,1).*x(:,4).^2 ...
%     + a(26)*x(:,2).^3 + a(27)*x(:,2).*x(:,2).^2.*x(:,3) + a(28)*x(:,2).*x(:,2).^2.*x(:,4) ...
%     + a(29)*x(:,2).*x(:,3).^2 + a(30)*x(:,2).*x(:,3).*x(:,4) + a(31)*x(:,2).*x(:,4).^2 ...
%     + a(32)*x(:,3).^3 + a(33)*x(:,3).^2.*x(:,4) + a(34)*x(:,3).*x(:,4).^2 ...
%     + a(35)*x(:,4).^3;
% 
% modelfun2 = @(b,x) b(1) + b(2)*x(:,1) + b(3)*x(:,2) + b(4)*x(:,3) + b(5)*x(:,4) ...                 
%     + b(6)*x(:,1).*x(:,1) + b(7)*x(:,1).*x(:,2) + b(8)*x(:,1).*x(:,3) + b(9)*x(:,1).*x(:,4) ...
%     + b(10)*x(:,2).*x(:,2) + b(11)*x(:,2).*x(:,3) + b(12)*x(:,2).*x(:,4) ...
%     + b(13)*x(:,3).*x(:,3) + b(14)*x(:,3).*x(:,4) ...
%     + b(15)*x(:,4).*x(:,4) ...
%     + b(16)*x(:,1).^3 + b(17)*x(:,1).*x(:,1).^2.*x(:,2) + b(18)*x(:,1).^2.*x(:,3) + b(19)*x(:,1).^2.*x(:,4) ...
%     + b(20)*x(:,1).*x(:,2).^2 + b(21)*x(:,1).*x(:,2).*x(:,3) + b(22)*x(:,1).*x(:,2).*x(:,4) ...
%     + b(23)*x(:,1).*x(:,3).^2 + b(24)*x(:,1).*x(:,3).*x(:,4) + b(25).*x(:,1).*x(:,4).^2 ...
%     + b(26)*x(:,2).^3 + b(27)*x(:,2).*x(:,2).^2.*x(:,3) + b(28)*x(:,2).*x(:,2).^2.*x(:,4) ...
%     + b(29)*x(:,2).*x(:,3).^2 + b(30)*x(:,2).*x(:,3).*x(:,4) + b(31)*x(:,2).*x(:,4).^2 ...
%     + b(32)*x(:,3).^3 + b(33)*x(:,3).^2.*x(:,4) + b(34)*x(:,3).*x(:,4).^2 ...
%     + b(35)*x(:,4).^3;

% modelfun1 = @(a,x) a(1) + a(2)*x(:,1) + a(3)*x(:,2) + a(4)*x(:,3) + a(5)*x(:,4) ...                 
%     + a(6)*x(:,1).*x(:,1) + a(7)*x(:,1).*x(:,2) + a(8)*x(:,1).*x(:,3) + a(9)*x(:,1).*x(:,4) ...
%     + a(10)*x(:,2).*x(:,2) + a(11)*x(:,2).*x(:,3) + a(12)*x(:,2).*x(:,4) ...
%     + a(13)*x(:,3).*x(:,3) + a(14)*x(:,3).*x(:,4) ...
%     + a(15)*x(:,4).*x(:,4);

% modelfun2 = @(b,x) b(1) + b(2)*x(:,1) + b(3)*x(:,2) + b(4)*x(:,3) + b(5)*x(:,4) ...                 
%     + b(6)*x(:,1).*x(:,1) + b(7)*x(:,1).*x(:,2) + b(8)*x(:,1).*x(:,3) + b(9)*x(:,1).*x(:,4) ...
%     + b(10)*x(:,2).*x(:,2) + b(11)*x(:,2).*x(:,3) + b(12)*x(:,2).*x(:,4) ...
%     + b(13)*x(:,3).*x(:,3) + b(14)*x(:,3).*x(:,4) ...
%     + b(15)*x(:,4).*x(:,4);

% modelfun1 = @(a,x) a(1) + a(2)*x(:,1) + a(3)*x(:,2) + a(4)*x(:,3) + a(5)*x(:,4);
% modelfun2 = @(b,x) b(1) + b(2)*x(:,1) + b(3)*x(:,2) + b(4)*x(:,3) + b(5)*x(:,4);

% modelfun1 = @(a,x) a(1) + a(2)*x(:,1) + a(3)*x(:,2) + a(4)*x(:,3) + a(5)*x(:,4) ...
%     + a(6)*x(:,1).^2 + a(7)*x(:,2).^2 + a(8)*x(:,3).^2 + a(9)*x(:,4).^2 ...
%     + a(10)*x(:,1).^3 + a(11)*x(:,2).^3 + a(12)*x(:,3).^3 + a(13)*x(:,4).^3 ...
%     + a(14)*x(:,1).^4 + a(15)*x(:,2).^4 + a(16)*x(:,3).^4 + a(17)*x(:,4).^4;
% 
% modelfun2 = @(b,x) b(1) + b(2)*x(:,1) + b(3)*x(:,2) + b(4)*x(:,3) + b(5)*x(:,4) ...
%     + b(6)*x(:,1).^2 + b(7)*x(:,2).^2 + b(8)*x(:,3).^2 + b(9)*x(:,4).^2 ...
%     + b(10)*x(:,1).^3 + b(11)*x(:,2).^3 + b(12)*x(:,3).^3 + b(13)*x(:,4).^3 ...
%     + b(14)*x(:,1).^4 + b(15)*x(:,2).^4 + b(16)*x(:,3).^4 + b(17)*x(:,4).^4;

modelfun1 = @(a,x) a(2)*exp(-((x(:,1)-a(3))/a(1)).^2) ...
    + a(4)*exp(-((x(:,1)-a(5))/a(1)).^2) ...
    + a(6)*exp(-((x(:,1)-a(7))/a(1)).^2) ...
    + a(8)*exp(-((x(:,2)-a(9))/a(1)).^2) ...
    + a(10)*exp(-((x(:,2)-a(11))/a(1)).^2) ...
    + a(12)*exp(-((x(:,2)-a(13))/a(1)).^2) ...
    + a(14)*exp(-((x(:,3)-a(15))/a(1)).^2) ...
    + a(16)*exp(-((x(:,3)-a(17))/a(1)).^2) ...
    + a(18)*exp(-((x(:,3)-a(19))/a(1)).^2) ...
    + a(20)*exp(-((x(:,4)-a(21))/a(1)).^2) ...
    + a(22)*exp(-((x(:,4)-a(23))/a(1)).^2) ...
    + a(24)*exp(-((x(:,4)-a(25))/a(1)).^2);

modelfun2 = @(b,x) b(2)*exp(-((x(:,1)-b(3))/b(1)).^2) ...
    + b(4)*exp(-((x(:,1)-b(5))/b(1)).^2) ...
    + b(6)*exp(-((x(:,1)-b(7))/b(1)).^2) ...
    + b(8)*exp(-((x(:,2)-b(9))/b(1)).^2) ...
    + b(10)*exp(-((x(:,2)-b(11))/b(1)).^2) ...
    + b(12)*exp(-((x(:,2)-b(13))/b(1)).^2) ...
    + b(14)*exp(-((x(:,3)-b(15))/b(1)).^2) ...
    + b(16)*exp(-((x(:,3)-b(17))/b(1)).^2) ...
    + b(18)*exp(-((x(:,3)-b(19))/b(1)).^2) ...
    + b(20)*exp(-((x(:,4)-b(21))/b(1)).^2) ...
    + b(22)*exp(-((x(:,4)-b(23))/b(1)).^2) ...
    + b(24)*exp(-((x(:,4)-b(25))/b(1)).^2);

modelfun = {modelfun1, modelfun2};

param = 0.1*randn(25,2);
% param = zeros(15,2);

options = statset('MaxIter', 100,'TolFun',1e-6);

%% inizio addestramento

% load("net.mat");

warning("off","all");

replayBufferSize = 10000;

% inizializzo le code
replayBufferS = zeros(replayBufferSize,4);
replayBufferY = zeros(replayBufferSize,2);

batchSize = 100;

S = zeros(batchSize,4);
target = zeros(batchSize,1);

m = 0;

% total return
G = zeros(numEpisodes,1);

% Loss
losses = zeros(numEpisodes,1);

% disp(param(1,:));

% lo stato iniziale è un punto vicino alla zona di tolleranza 
s0 = [0; 0; pi + ((rand*0.3)-0.15); 0];
k = 0;
for e = 1:numEpisodes
    disp(e);

    % Stato iniziale
    s = s0;

    % normalizzo lo stato in valori compresi fra -1 e 1
    sNorm = normalizeState(s);

    % Prende la Q facendo uno step di forward-propagation nella rete
    Q = [modelfun{1}(param(:,1),sNorm') modelfun{2}(param(:,2),sNorm')];
        
    % All'inizio lo stato non è mai terminale
    isTerminal = 0;
    
    % Questi sono contatori che servono per gestire le strutture di dati 
    % usate per memorizzare gli stati visitati
    
    n = 0;
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
        spNorm = normalizeState(sp);

        if mod(n,3) == 0
            % drawpend(sp,mm,MM,L);
        end

        % se raggiungo lo stato d'equilibrio, assegno reward positivo
        % if (0.01*sp(1)^2 + (sp(3)-pi)^2) < 0.005
        if sp(1)^2 < 1 && (sp(3)-pi)^2 < 0.05 && sp(4)^2 < 1
            isTerminal = 0;
            r = 1;
        elseif sp(1) < X(1) || sp(1) > X(2) || sp(2) < V(1) || sp(2) > V(2) || sp(3) < THETA(1) || sp(3) > THETA(2) || sp(4) < OMEGA(1) || sp(4) > OMEGA(2)
            isTerminal = 1;
            r = 0;
        else
            isTerminal = 0;
            r = 0;
        end
        
        % ritorno complessivo
        G(e) = G(e) + r;
        
        % calcolo il target secondo la legge del Q-learning
        y = Q;
        if isTerminal
            y(a) = r;   
        else
            Qp = [modelfun{1}(param(:,1),spNorm') modelfun{2}(param(:,2),spNorm')];
            y(a) = r + gamma*max(Qp);
        end
        
        losses(e) = losses(e) + norm(y-Q,2);
        
        % accumulo le informazioni in una coda (accesso circolare)
        replayBufferS(mod(m,replayBufferSize)+1,:) = sNorm';
        replayBufferY(mod(m,replayBufferSize)+1,:) = y';

        m = m + 1;

        if m >= batchSize
            index = randperm(min(m,replayBufferSize),batchSize);
            S = replayBufferS(index,:);
            Y = replayBufferY(index,a);

            mdl = fitnlm(S,target,modelfun{a},param(:,a),"Options",options);
        
            % aggiorno i parametri del polinomio corrispondente all'azione
            param(:,a) = mdl.Coefficients.Estimate;
            % disp(param(1,:));
        
            % resetto i batch
            S = zeros(batchSize,4);
            target = zeros(batchSize,1);

            k = 0;
        end

        % addestro il regressore sui campioni
        % tbl = table(sNorm(1),sNorm(2),sNorm(3),sNorm(4),y(1));        
        

        % if mod(e,update_target_steps) == 0
        %     net_target = net;
        % end

        s = sp;
        sNorm = spNorm;
        Q = Qp;

    end
  
    disp(param);
    epsilon = max(0.5, epsilon * 0.999);
end

% save('net.mat', 'net');

%% plot
close all

% load("net.mat");

s0 = [0; 0; pi + ((rand*0.3)-0.15); 0];
s = s0;

isTerminal = 0;

while true
    sNorm = normalizeState(s);
    Q = [modelfun{1}(param(:,1),sNorm') modelfun{2}(param(:,2),sNorm')];
    a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q

    sp = dinamica(s, mm, MM, L, g, a, Ts);
    sp = sp(:);

    if sp(1) < X(1) || sp(1) > X(2) || sp(2) < V(1) || sp(2) > V(2) || sp(3) < THETA(1) || sp(3) > THETA(2) || sp(4) < OMEGA(1) || sp(4) > OMEGA(2)
        % isTerminal = 1;
        pause(1);
        sp = [0; 0; pi + ((rand*0.3)-0.15); 0];
    end

    drawpend(sp,mm,MM,L);

    % update state
    s = sp;
end

%% Plot della Q-function 

x = linspace(-10,10,100);
y = zeros(4,length(x));
state = zeros(4,1);
for i = 1:length(x)
    for j = 1:4
        state(j) = x(i);
        Q = [modelfun{1}(param(:,1),state') modelfun{2}(param(:,2),state')];
        a = find(Q == max(Q), 1, 'first');
        y(j,i) = Q(a);
        state = zeros(4,1);
    end
end

hold on
plot(x,y(1,:));
plot(x,y(2,:));
plot(x,y(3,:));
plot(x,y(4,:));
hold off
