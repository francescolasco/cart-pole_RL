clc;
close all;

s0 = [0; 0; pi - pi/5; 0];
Ts = 0.025;

% parametri del sistema
m = 0.5; % massa del pendolo
M = 1; % massa del carrello
L = 1.5; % lunghezza del pendolo
g = -9.81;

d = 100; % questo non serve più (era nel vecchio modello)

s = s0;
for i = 1:500
    %u = ((randi(2) * 2) - 3) * 100;
    u = 0;

    mmodel = @(t,x,u) model(s,m,M,L,g,d,u); 
    [~, x] = ode45(@(t, x) mmodel(t, s, u), [0,Ts], s);
    sp = x(end,:);

    drawpend(s,m,M,L);
   
    % disp(pi-sp(3));
    % if (pi-sp(3))^2 < 0.01
    %     break;
    % end
     
    s = sp;
end

