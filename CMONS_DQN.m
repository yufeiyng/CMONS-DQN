classdef CMONS_DQN < ALGORITHM
    % <multi/many> <real/integer/label/binary/permutation> <constrained>
    % Deep reinforcement learning assisted constrained multi-objective evolutionary algorithm with novelty search

    methods

        function main(Algorithm, Problem)
            % Parameters for CVT
            numCentroids = 30;
            maxSizePerCell = 15;
            numSamples = 10;

            %% Initialize the CVT by Latin hypercube sampling
            PopDec = UniformPoint(numCentroids * 100, Problem.D, 'Latin');
            PopDec = repmat(Problem.upper - Problem.lower, numCentroids * 100, 1) .* PopDec + repmat(Problem.lower, numCentroids * 100, 1);
            CVT = CVTMAPElites(numCentroids, maxSizePerCell, PopDec);

            Population1 = Problem.Evaluation(CVT.centroids);
            Population2 = Population1;

            newConv = zeros(numCentroids, 1);
            newDiv = zeros(numCentroids, 1);
            [CVT, newConv, newDiv] = CVT.AddToCVT(Population1, newConv, newDiv);

            % Parameters for epsilon
            CVmax = max(sum(max(0, Population2.cons), 2));
            cp = 2; alpha = 0.95; tao = 0.05;
            % Set epsilon to inf for ignoring constraints in search mode 1
            searchMode = 1; epsilon = inf; gen = 1;
            % Parameters for change detection
            maxChange = 1; threshold = 1e-2;

            Fitness1 = CalFitness(Population1.objs, Population1.cons, 0);
            Fitness2 = CalFitness(Population2.objs, Population2.cons, epsilon);

            %% For DQL
            greedy = 0.95; gamma = 0.9;
            built = 0; count = 0; EP = [];

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                CV2 = overall_cv(Population2.cons);
                CVmax = max([max(CV2), CVmax]);
                rf = sum(CV2 <= 1e-6) / Problem.N;

                if rand > greedy || gen <= 300 || length(EP) < 200
                    [~, rank] = sort(CVT.nst);
                    action = rank(1:numSamples);
                else
                    % Choose an action based on the trained net
                    if ~built
                        % Build model
                        rEP = randperm(length(EP), 200);
                        trX = EP(rEP, 1:3);
                        trY = EP(rEP, 4:6);
                        [net, Params] = trainmodel(trX, trY, []);
                        built = 1;
                        action = randi([1, numCentroids], [numSamples, 1]);
                    else
                        % Use the model to choose action
                        x = [newConv, newDiv, (1:numCentroids)'];
                        x = mapminmax(x', 0, 1)';
                        reward = testNet(x, net, Params);
                        [~, action] = maxk(reward(:, 1), numSamples);
                    end

                end

                % Novelty Search
                [CVT, Offspring3] = CVT.NoveltySearch(Problem, action);

                % Generate offspring
                NP = ceil(Problem.N / 2);
                [FrontNo1, ~] = NDSort(Population1.objs, Population1.cons, inf);
                [FrontNo2, ~] = NDSort(Population2.objs, inf);
                CrowdDis1 = CrowdingDistance(Population1.objs, FrontNo1);
                CrowdDis2 = CrowdingDistance(Population2.objs, FrontNo2);
                Mate1 = TournamentSelection(2, NP, -CrowdDis1);
                Mate2 = TournamentSelection(2, NP, -CrowdDis2);

                % Novelty Search with DE/current-to-rand/1
                [CVT, Offspring1_1] = CVT.CentroidsGeneration(Problem, Population1(Mate1), 0);
                [CVT, Offspring2_1] = CVT.CentroidsGeneration(Problem, Population2(Mate2), epsilon);

                % Fitnesss-Driven Search with DE/rand-to-best/1
                Offspring1_2 = DE_rand_to_best_1(Problem, Population1.decs, Fitness1, NP, 0.1);
                Offspring2_2 = DE_rand_to_best_1(Problem, Population2.decs, Fitness2, NP, 0.1);

                Offspring1 = [Offspring1_1; Offspring1_2];
                Offspring2 = [Offspring2_1; Offspring2_2];

                mapPop = CVT.GetAllIndividuals();
                Offspring1 = setdiff(Offspring1, [Population1.decs; Population2.decs; mapPop.decs], "stable", "rows");

                if ~isempty(Offspring1)
                    Offspring1 = Problem.Evaluation(Offspring1);
                    Offspring2 = setdiff(Offspring2, [Population1.decs; Population2.decs; mapPop.decs; Offspring1.decs], "stable", "rows");
                    Offspring3 = setdiff(Offspring3, [Population1.decs; Population2.decs; mapPop.decs; Offspring1.decs], "stable", "rows");
                end

                if ~isempty(Offspring2)
                    Offspring2 = Problem.Evaluation(Offspring2);
                    Offspring3 = setdiff(Offspring3, Offspring2.decs, "stable", "rows");
                end

                if ~isempty(Offspring3)
                    Offspring3 = Problem.Evaluation(Offspring3);
                end

                if searchMode == 1
                    Objvalues(gen) = sum(sum(Population2.objs, 1));
                    [FrontNo2, ~] = NDSort(Population2.objs, size(Population2.objs, 1));
                    NC2 = sum(FrontNo2 == 1);

                    if gen ~= 1
                        maxChange = abs(Objvalues(gen) - Objvalues(gen - 1));
                    end

                    if maxChange < threshold && NC2 == Problem.N
                        epsilon = max(CV2);
                        searchMode = 2;
                    end

                else
                    epsilon = UpdateEpsilon(Problem, tao, epsilon, CVmax, rf, alpha, cp);
                end

                Offspring = [Offspring1, Offspring2, Offspring3];

                % Save the old state before updating CVT
                oldConv = newConv; oldDiv = newDiv;

                % Update CVT
                [CVT, newConv, newDiv] = CVT.AddToCVT(Offspring, newConv, newDiv);

                % Environmental selection
                [Population1, Fitness1] = EnvironmentalSelection([Population1, Offspring], Problem.N, true, 0);
                [Population2, Fitness2] = EnvironmentalSelection([Population2, Offspring], Problem.N, false, epsilon);

                %% Update experience replay
                reward = (oldConv + oldDiv) - (newConv + newDiv);
                record = [oldConv, oldDiv, (1:numCentroids)', reward, newConv, newDiv];
                record = mapminmax(record', 0, 1)';
                EP = [record(reward ~= 0, :); EP];

                if size(EP, 1) > 400
                    EP(401:end, :) = [];
                end

                %% Update Q-net
                % Update net model every 50 generations
                if built
                    count = count + 1;

                    if count > 50
                        % Update model
                        rEP = randperm(length(EP), 200);
                        trX = EP(rEP, 1:3);
                        reward = testNet(trX, net, Params);
                        succ = reward(:, 1);
                        trY = EP(rEP, 4) + gamma * max(succ);
                        trY = mapminmax(trY', 0, 1)';
                        net = updatemodel(trX, trY, Params, net);
                        count = 0;
                    end

                end

                gen = gen + 1;
                CVT = CVT.UpdateCentroids();
            end

        end

    end

end

function result = overall_cv(cv)
    cv(cv <= 0) = 0;
    cv = abs(cv);
    result = sum(cv, 2);
end

% Update the value of epsilon
function epsilon = UpdateEpsilon(Problem, tao, epsilon_k, epsilon_0, rf, alpha, cp)

    if Problem.FE > 0.8 * Problem.maxFE
        epsilon = 0;
    else

        if rf < alpha
            epsilon = (1 - tao) * epsilon_k;
        else
            epsilon = epsilon_0 * ((1 - (Problem.FE / (0.8 * Problem.maxFE))) ^ cp);
        end

    end

end
