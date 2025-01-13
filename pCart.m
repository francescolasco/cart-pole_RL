clc
clear all
close all
%%

previousRngState = rng(0,"twister");
env = cartPoleDQN;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);


initOpts = rlAgentInitializationOptions(NumHiddenUnit=20);

agentOpts = rlDQNAgentOptions( ...
    MiniBatchSize            = 256,...
    TargetSmoothFactor       = 1, ...
    TargetUpdateFrequency    = 4,...
    UseDoubleDQN             = false);

agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-3;
agent = rlDQNAgent(obsInfo,actInfo,initOpts,agentOpts);
getAction(agent,{rand(obsInfo.Dimension)})

%%
trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=500, ...
    Verbose= true, ...
    StopTrainingCriteria="EpisodeCount",...
    StopTrainingValue=500);

% agent evaluator
evl = rlEvaluator(EvaluationFrequency=20, NumEpisodes=5);
%%
clc
trainingStats = train(agent,env,trainOpts,Evaluator=evl);

%%
out=runEpisode(env,agent)

for i = 1:length(out.AgentData.Experiences)
    
    sp = out.AgentData.Experiences(i).Observation{1};
    drawpend(sp,mm,MM,L);

    historyX = [historyX, sp(1)];
    historyV = [historyV, sp(2)];
    historyTHETA = [historyTHETA, sp(3)];
    historyOMEGA = [historyOMEGA, sp(4)];

    steps = steps + 1;
end

X = [-5 5];
V = [-20 20];
marg = 0.05;
THETA = [pi-pi/6 pi+pi/6];
OMEGA = [-20 20];
figure(1);
plot(historyX,historyV);
axis([X V]);

figure(2);
plot(historyTHETA,historyOMEGA);
axis([THETA OMEGA]);
