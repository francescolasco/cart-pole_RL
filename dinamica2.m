function [sp, r, isTerminal] = dinamica2(s, s0, m, M, L, g, d, a, Ts, X, V, THETA, OMEGA, z)

% transform action in acceleration
switch a
    case 1
        u = 20;
    case 2
        u = -20;
end

mmodel = @(t,x,u) model(s,m,M,L,g,d,u); 
[~, x] = ode45(@(t, x) mmodel(t, s, u), [0,Ts], s);
sp = x(end,:);

if (sqrt(sp(1)^2) > 2)||(sp(3)> pi+0.07)||(sp(3) < pi-0.07)
    isTerminal = 1;
    r = -5;
else
    isTerminal = 0;
    r = 1;
end

% if (sp(3)> pi+0.07)||(sp(3) < pi-0.07)
%     isTerminal = 1;
%     r = -5;
% elseif (sqrt(sp(1)^2) > 2)
%     isTerminal = 1;
%     r = -7;
% else
%     isTerminal = 0;
%     r = 1;
% end