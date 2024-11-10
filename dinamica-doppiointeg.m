function [sp, r, isTerminal] = dinamica(s, a, tt, X, V)

% get states
x = s(1);
v = s(2);

% transform action in acceleration
switch a
    case 1
        acc = 1;
    case 2
        acc = -1;
end

% % we should integrate continuous-time dynamics
% Ts = 1; % sampling time
% % integrate differential equations
% odefun = @(t,s) [s(2); 0.001*a - 0.0025*cos(3*x(1))];
% [t, st] = ode45(odefun, [0, Ts], s);
% % get next states as the last one
% sp = st(end,:)';

% to simplify, we use the forward Euler discretization
% xp = x + v;
% vp = v + 0.001*acc - 0.0025*cos(3*x);
xp = x + tt*v;
vp = v + tt*acc;  

% saturate the next state
xp = max(min(xp, X(2)),X(1));
vp = max(min(vp, V(2)),V(1));

% % implement the impact dynamics
% if xp == X(1) && vp < 0
%     vp = 0;
% end

% define next state
sp = [xp; vp];

% define reward
r = -1;

% define isTerminal
if (xp^2 + vp^2) <= 0.001
    isTerminal = 1;
else
    isTerminal = 0;
end