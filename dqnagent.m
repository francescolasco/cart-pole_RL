clear all
close all
clc

env = rlPredefinedEnv("CartPole-Discrete");

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

initOpts = rlAgentInitializationOptions(NumHiddenUnit=128);

agentOpts = rlDQNAgentOptions( ...
    MiniBatchSize            =  128,...
    TargetSmoothFactor       = 1, ...
    TargetUpdateFrequency    = 4,...
    UseDoubleDQN             = false);

agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-3;

agent = rlDQNAgent(obsInfo,actInfo,initOpts,agentOpts);

% training options
trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=500, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="EvaluationStatistic",...
    StopTrainingValue=500);

% agent evaluator
evl = rlEvaluator(EvaluationFrequency=20, NumEpisodes=5);

%% inizio addestramento

trainingStats = train(agent,env,trainOpts,Evaluator=evl);

dqnAgentNet = getModel(getCritic(agent));

save('trainednet_dqnagent.mat', 'dqnAgentNet');

%% plot

state = env.reset();
for i=1:500
    
    action = agent.getAction(state);

    [nextState, reward, isDone, info] = env.step(action{1});

    plot(env);

    state = nextState;

    % Interrompe la simulazione se l'episodio è finito
    if isDone
        break;
    end
end