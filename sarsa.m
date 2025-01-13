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
Ts = 0.02;

% size of the state space
X = [-5 5];
V = [-20 20];
THETA = [pi - pi/8 pi + pi/8];
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

maxSteps = 10000;

tau = 0.1;

s0 = [0; 0; pi + ((rand*0.3)-0.15); 0];

for e = 1:numEpisodes
    % Questo è importante: quando all'interno di un episodio il cart-pole
    % finisce in uno stato che è fuori dalla griglia, non riparte in uno
    % stato casuale ma in quello da cui è partito, e questo finchè non
    % termina l'episodio. Se così non fosse, potrebbe succedere che il
    % cart-pole dopo essere finito in uno stato fuori dalla griglia riparte
    % da uno stato terminale e questo viene ricompensato positivamente,
    % facendo apprendere al modello un'informazione errata.
    
    % Stati iniziali per la prima parte
    % s0 = [0; 0; pi + ((2 * rand * (pi/6)) - (pi/6)); 0];
    

    % Stati iniziali per la seconda parte
    % s0 = [rand*8 - 4; 0; pi; 0];

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

    step = 0;
    
    while ~isTerminal && step < maxSteps
        step = step + 1;

        % take action a and observe sp and r
        sp = dinamica(s, mm, MM, L, g, a, Ts);

        % r = -((sp(3)-pi)^2 + sp(4)^2);

        % se raggiungo lo stato d'equilibrio, assegno reward positivo
        if sp(1)^2 < 1 && (sp(3)-pi)^2 < 0.01 && sp(4)^2 < 1
            isTerminal = 0;
            r = 1;
        elseif sp(1) < X(1) || sp(1) > X(2) || sp(2) < V(1) || sp(2) > V(2) || sp(3) < THETA(1) || sp(3) > THETA(2) || sp(4) < OMEGA(1) || sp(4) > OMEGA(2)
            isTerminal = 1;
            r = 0;
        else
            isTerminal = 0;
            r = 0;
        end
        % se raggiungo una configurazione fuori dallo spazio di stato,
        % termino l'episodio

        % if ((sp(3)-pi)^2 + sp(4)^2) <= 0.001
        %     isTerminal = 1;
        %     r = 0;
        % else
        %     isTerminal = 0;
        %     r = -10;
        % end

        % if ((sp(3)-pi)^2 + sp(4)^2) <= 0.01 && step > MaxSteps 
        %     isTerminal = 1;
        %     r = 0;
        % elseif ((sp(3)-pi)^2 + sp(4)^2) <= 0.01 && step <= MaxSteps 
        %     step = step + 1;
        %     isTerminal = 0;
        %     r = -1;
        % else
        %     step = 0;
        %     isTerminal = 0;
        %     r = -1;
        % end
        % 
        % disp(step);

        % integralError = integralError + (sp(3)-pi)^2;
        % 
        % if integralError < 0.1
        %     isTerminal = 1;
        %     integralError = 0;
        % else
        %     isTerminal = 0;
        % end
        
        % % Aggiorno il vettore thetas (sarebbero i valori di theta dell'ultima parte della
        % % simulazione); è una finestra scorrevole.
        % for i = 2:length(errors)
        %     errors(i-1) = errors(i);
        % end
        % errors(length(errors)) = (sp(3)-pi)^2 + sp(4)^2;
        % % thetas(length(thetas)) = (sp(3)-pi)*Ts;
        % % disp(thetas);
        % 
        % % Condizioni sullo stato terminale: voglio che il pendolo rimanga in
        % % posizione verticale con un errore < 0.3 rad per 1 secondo
        % integralErrors = sum(errors);
        % 
        % if integralErrors < 1
        %     isTerminal = 1;
        % else
        %     isTerminal = 0;
        % end

        if mod(step,3) == 0
            % drawpend(sp,mm,MM,L);
        else
            % close all;
        end

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
       
    % epsilon = max(0.2, epsilon * ((numEpisodes - 2.5)/numEpisodes));
    epsilon = max(0.5, epsilon * 0.9999);
end

% save("w-softmax.mat","w");

%% plot
close all
clc

load("w.mat");

s0 = [0; 0; pi + ((rand*0.3)-0.15); 0];

% get feature for initial state
Fac = get_features(s, cellX, cellV, cellTHETA, cellOMEGA, M, N);

% get quality function
Q = sum(w(Fac,:));

a = find(Q == max(Q), 1, 'first'); % take greedy action wrt Q

% at the beginning is not terminal
isTerminal = false;

s = s0;

while true
    sp = dinamica(s, mm, MM, L, g, a, Ts);

    drawpend(sp,mm,MM,L);

    % disp(sp);

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
end

% while ~isTerminal
%     % take action a and observe sp and r
%     [sp, r, isTerminal] = dinamica(s, s0, mm, MM, L, g, dd, a, Ts, X, V, THETA, OMEGA);
% 
%     % Aggiorno il vettore thetas (sarebbero i valori di theta dell'ultima parte della
%     % simulazione); è una finestra scorrevole.
%     % for i = 2:length(thetas)
%     %     thetas(i-1) = thetas(i);
%     % end
%     % thetas(length(thetas)) = (sp(3)-pi)*Ts;
%     % 
%     % % Condizioni sullo stato terminale
%     % integralTheta = sum(abs(thetas));
%     % 
%     % if integralTheta < 0.05
%     %     isTerminal = 1;
%     % else
%     %     isTerminal = 0;
%     % end
% 
%     drawpend(sp,mm,MM,L);
% 
%     historyX = [historyX, sp(1)];
%     historyV = [historyV, sp(2)];
%     historyTHETA = [historyTHETA, sp(3)];
%     historyOMEGA = [historyOMEGA, sp(4)];
% 
%     % update total return
%     G(e) = G(e) + r;
%     if isTerminal
%         % impose that next value is 0, delta = r + gamma*Qp(ap) -
%         % sum(w(Fac,a)) quindi gamma*Qp(ap) è 0
%         delta = r - sum(w(Fac,a)); 
%     else
%         % get active features at next state
%         Facp = get_features(sp, cellX, cellV, cellTHETA, cellOMEGA, M, N);
%         % compute next q function
%         Qp = sum(w(Facp,:));
%         % take greedy action
%         ap = find(Qp == max(Qp), 1, 'first');
%         % compute temporal difference error
%         delta = r + gamma*Qp(ap) - sum(w(Fac,a));
%     end
%     % update weigth vector
%     w(Fac,a) = w(Fac,a) + alpha*delta;
% 
%     if ~isTerminal
%         % update state, action and features
%         s = sp;
%         a = ap;
%         Fac = Facp;
%     end
% end
