s0 = [0; 0; pi + 0.01; 0];
Ts = 0.001;

% parametri del sistema
m = 0.5;
M = 2.5;
L = 2;
g = -9.81;
d = 1;

s = s0;
for i = 1:500
    %u = ((randi(2) * 2) - 3) * 100;
    u = ((mod(i,2)*2)-1) * 200;
    mmodel = @(t,x,u) model(s,m,M,L,g,d,u); 
    [~, x] = ode45(@(t, x) mmodel(t, s, u), [0,Ts], s);
    sp = x(end,:);

    drawpend(s,m,M,L);

    s = sp;
end

