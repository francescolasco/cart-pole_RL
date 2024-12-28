function sNormalized = normalizeState(s)
    sNormalized = zeros(4,1);
    
    sNormalized(1) = s(1) / 5;
    sNormalized(2) = s(2) / 25;
    sNormalized(3) = (s(3) - pi) / pi/4;
    sNormalized(4) = s(4) / 20;
end