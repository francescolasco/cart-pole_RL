clear all
close all
clc

env = rlPredefinedEnv("CartPole-Discrete");

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

% training options
trainOpts = rlTrainingOptions(...
    MaxEpisodes=100, ...
    MaxStepsPerEpisode=250, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="EvaluationStatistic",...
    StopTrainingValue=250);

% agent evaluator
evl = rlEvaluator(EvaluationFrequency=20, NumEpisodes=5);

%% inizio addestramento

% trainingStats = train(agent,env,trainOpts,Evaluator=evl);
simOptions = rlSimulationOptions(MaxSteps=1);
for e = 1:100
    state = env.reset();
    
    isDone = 0;
    n = 0;
    while ~isDone || n < 250
        n = n + 1;

        action = agent.getAction(state);
    
        experience = sim(env,agent,simOptions);
        
        isDone = experience.IsDone.Data;
        nextState = experience.Observation.CartPoleStates.Data(:,:,2);
        
        learn(agent, experience);

        state = nextState;
    end
end

%% plot

state = env.reset();
for i=1:1000
    
    action = agent.getAction(state);

    [nextState, reward, isDone, info] = env.step(action{1});

    plot(env);

    state = nextState;

    % Interrompe la simulazione se l'episodio è finito
    if isDone
        break;
    end
end