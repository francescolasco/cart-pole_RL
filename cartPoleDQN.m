classdef cartPoleDQN < rl.env.MATLABEnvironment
    
    properties
        Gravity = 9.8000
        MassCart = 1
        MassPole =  0.5
        Length =  1.5000
        Friction = 0.45
        MaxForce = 20
        Ts = 0.0200
        ThetaThresholdRadians = 0.2094
        XThreshold = 2.4000
        RewardForNotFalling = 1
        PenaltyForFalling = -5
    end

    properties
        % [x,dx,theta,dtheta]'
        State = zeros(4,1)
    end
    
    properties(Access = protected)
        IsDone = false        
    end

    methods
        function this = cartPoleDQN()
            ObservationInfo = rlNumericSpec([4 1]);
            ObservationInfo.Name = 'States';
            ObservationInfo.Description = 'x v theta omega';

            ActionInfo = rlFiniteSetSpec([-1 1]);
            ActionInfo.Name = 'CartPole Action';

            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            %updateActionInfo(this);
        end
        
        % Reset environment to initial state and return initial observation
        function InitialObservation = reset(this)
            % Theta (+- .05 rad)
            T0 = pi+ 2 * 0.05 * rand - 0.05;  
            % Thetadot
            Td0 = 0;
            % X 
            X0 = 0;
            % Xdot
            Xd0 = 0;
        
            InitialObservation = [X0;Xd0;T0;Td0];
            this.State = InitialObservation;
        
            % (Optional) Use notifyEnvUpdated to signal that the 
            % environment is updated (for example, to update the visualization)
            notifyEnvUpdated(this);
        end

        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = []; 

            % transform action in acceleration
            switch Action
                case 1
                    u = 20;
                case -1
                    u = -20;
            end
           
            mmodel = @(t,x,u) model( this.State, this.MassCart, this.MassPole, this.Length, this.Gravity , this.Friction , u); 
            [~, x] = ode45(@(t, x) mmodel(t, this.State, u), [0,this.Ts], this.State);
            Observation = x(end,:)';
            
            % Update system states
            this.State = Observation;
        
            % Check terminal condition
            X = Observation(1);
            Theta = Observation(3);
            IsDone = abs(X) > this.XThreshold || abs(pi-Theta) > this.ThetaThresholdRadians;
            this.IsDone = IsDone;
        
            % Get reward
            Reward = getReward(this);
        
            % (Optional) Use notifyEnvUpdated to signal that the 
            % environment has been updated (for example, to update the visualization)
            notifyEnvUpdated(this);
        end

       function Reward = getReward(this)
            if ~this.IsDone
                Reward = this.RewardForNotFalling;
            else
                Reward = this.PenaltyForFalling;
            end          
        end
    end

end