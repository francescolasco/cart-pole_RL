clear all;
close all;

% position in the state space
x = 0.7;
y = -2.3;

N = 10; % number of cells
M = 10; % number of grids

xMax = 2.4; % assumo che lo spazio sia da -xMax a xMax
yMax = 20; % assumo che lo spazio sia da -yMax a yMax

dx = 2*xMax/N; % larghezza quadratino
dy = 2*yMax/N; % altezza quadratino

displacement = [1 3]; % spiazzamento relativo su asse x e su asse y
displacement = displacement/max(displacement); % normalizzazione

mx = 2*xMax/N/M*displacement(1);
my = 2*yMax/N/M*displacement(2);

for k = 0:M
    clr = rand(1, 3);
    % dd = displacement * k;
    dd = [mx my]*k;
    % disegna bordo esterno
    plot([-xMax+dd(1) xMax+dd(1)], [-yMax+dd(2) -yMax+dd(2)], "Color", clr); % base inferiore
    hold on
    plot([-xMax+dd(1) xMax+dd(1)], [yMax+dd(2) yMax+dd(2)], "Color", clr); % base superiore
    plot([-xMax+dd(1) -xMax+dd(1)], [-yMax+dd(2) yMax+dd(2)], "Color", clr); % parete sinistra
    plot([xMax+dd(1) xMax+dd(1)], [-yMax+dd(2) yMax+dd(2)], "Color", clr); % parete destra

    
    % disegna le righe
    for i = -yMax:dy:yMax
        plot([-xMax+dd(1) xMax+dd(1)], [i+dd(2) i+dd(2)], "Color", clr); % base inferiore
    end

    % disegna le colonne
    for i = -xMax:dx:xMax
        plot([i+dd(1) i+dd(1)], [-yMax+dd(2) yMax+dd(2)], "Color", clr); % base inferiore
    end

end

plot(x,y,'.','MarkerSize',20)