close all; clear; clc;

% matrici dell'uscita y=C*x+D*u
C = eye(4);     % ipotizzo di avere accesso a tutto lo stato
D = [0;0;0;0];  % // di non avere legame diretto i/o

nu = 1; % # ingressi
no = 4; % # uscite

% parametri del sistema dinamico
m = 1;
M = 5;
L = 2;
g = -10;
d = 1;

% plot del cartpole
visual = 1;

% stato iniziale x0
x = [0; 0; pi-0.1; 0]; 

% inizializzo l'ingresso u0 = 0
ui = 0;


% tempi
Ts = 0.001;         % campionamento    
Tfinal = 2;         % tempo missione


ref = [0; 0; pi; 0];    % riferimento 

y  = zeros(no,Tfinal/Ts); % vettore delle uscite
uh = zeros(nu,Tfinal/Ts); % vettore degli ingressi

% definizione della funzione per il modello
mmodel = @(t,x,u) pendcart(x,m,M,L,g,d,u); 

% simulazione
for t=1:Tfinal/Ts

    y(:,t) = C*x+D*ui;       % y(k*Ts) 
  
    [foo, xx] = ode45(@(t, x) mmodel(t, x, ui), [t,t+Ts], x); % simulo per un intervallo Ts
    x = xx(end,:)';  % stato alla fine della simulazione 

    error = x-ref;
    ui = -80*(error(3))-10*(error(4));  % Controllo PD su angolo

    uh(:,t) = ui;     % salvo l'ingresso
end


% plot

if visual == 1
    for t=1:10:Tfinal/Ts
    drawpend(y(1:4,t),m,M,L)
    end
end

tt = Ts:Ts:Tfinal;
% figure(2)
% subplot(5,1,1)
% plot(tt ,y(1,:),'b');
% title('x_1(t)');
% 
% subplot(5,1,2)
% plot(tt ,y(2,:),'b');
% title('x_2(t)');
% 
% subplot(5,1,3)
% plot(tt ,y(3,:),'b');
% title('x_3(t)');
% 
% subplot(5,1,4)
% plot(tt ,y(4,:),'b');
% title('x_4(t)');
% 
% subplot(5,1,5)
% plot(tt ,uh);
% title('u(t)');
% 

figure(2)
subplot(3,1,1)
plot(tt ,y(3,:),'b');
title('x_3(t)');
 
subplot(3,1,2)
plot(tt ,y(4,:),'b');
title('x_4(t)');

subplot(3,1,3)
plot(tt ,uh);
title('u(t)');
