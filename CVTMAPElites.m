classdef CVTMAPElites

    properties
        niche
        nst % novelty search times
        centroids
        numCentroids
        maxSize
    end

    methods

        function obj = CVTMAPElites(numCentroids, maxSize, solutions)
            obj.numCentroids = numCentroids;
            obj.maxSize = maxSize;
            obj.niche = cell(numCentroids, 1);
            obj.nst = zeros(numCentroids, 1);
            [~, obj.centroids] = kmeans(solutions, numCentroids, 'MaxIter', 1000); % K-means for classification
        end

        function obj = UpdateCentroids(obj)

            for i = 1:obj.numCentroids
                nichePop = obj.niche{i};
                obj.centroids(i, :) = mean(nichePop.decs);
            end

        end

        function [obj, Offspring] = CentroidsGeneration(obj, Problem, Population, epsilon)
            Offspring = [];

            indices = knnsearch(obj.centroids, Population.decs);

            for i = unique(indices)'
                groupIndices = find(indices == i);
                tempPop = [];

                for i1 = groupIndices'
                    tempPop = [Population(i1), tempPop];
                end

                if ~isempty(tempPop)
                    obj.nst(i) = obj.nst(i) + 1;

                    if length(tempPop) == 1
                        % random perturbation when only one solution
                        if rand < 0.5
                            tempOff = Problem.lower + Problem.upper - rand(1, Problem.D) .* tempPop.decs;
                        else
                            tempOff = Problem.lower + rand(1, Problem.D) .* (Problem.upper - tempPop.decs);
                        end

                    else
                        tempFit = CalFitness(tempPop.objs, tempPop.cons, epsilon);
                        tempOff = DE_current_to_rand_1(Problem, tempPop.decs, tempFit, length(tempPop));
                    end

                    Offspring = [Offspring; tempOff];
                end

            end

        end

        function [obj, new_conv, new_div] = AddToCVT(obj, Population, new_conv, new_div)
            indices = knnsearch(obj.centroids, Population.decs);

            for i = unique(indices)'
                groupIndices = find(indices == i);
                tempPop = obj.niche{i};

                for i1 = groupIndices'
                    tempPop = [Population(i1), tempPop];
                end

                FrontNo = NDSort(tempPop.objs, tempPop.cons, 1);
                tempPop = tempPop(FrontNo == 1);

                if length(tempPop) > obj.maxSize
                    CrowdDis = CalCrowdDis(tempPop.objs);
                    [~, rank] = sort(CrowdDis, 'descend');
                    tempPop = tempPop(rank(1:obj.maxSize));
                end

                obj.niche{i} = tempPop;
                existPi = Deduplicate(Population(groupIndices).decs, tempPop.decs);

                if size(existPi, 1) ~= length(groupIndices)
                    [new_conv(i), new_div(i)] = CalculateState(tempPop);
                end

            end

        end

        function [obj, Offspring] = NoveltySearch(obj, Problem, index)
            Offspring = [];

            for i = 1:length(index)
                obj.nst(index(i)) = obj.nst(index(i)) + 1;
                tempPop = obj.niche{index(i)};

                if length(tempPop) == 1
                    % random perturbation when only one solution
                    if rand < 0.5
                        tempOff = Problem.lower + Problem.upper - rand(1, Problem.D) .* tempPop.decs;
                    else
                        tempOff = Problem.lower + rand(1, Problem.D) .* (Problem.upper - tempPop.decs);
                    end

                else
                    nicheFit = CalFitness(tempPop.objs, tempPop.cons, 0);
                    tempOff = DE_current_to_rand_1(Problem, tempPop.decs, nicheFit, length(tempPop));
                end

                Offspring = [Offspring; tempOff];
            end

        end

        function individuals = GetAllIndividuals(obj)
            individuals = [];

            for i = 1:numel(obj.niche)
                individuals = [individuals obj.niche{i}];
            end

        end

    end

end

function [new_conv, new_div] = CalculateState(Pop)
    new_conv = sum(sum(Pop.objs, 2)) / length(Pop);
    f_max = max(Pop.objs, [], 1);
    f_min = min(Pop.objs, [], 1);
    new_div = 1 ./ sum(f_max - f_min);

    % only one non-donimated solution
    if isinf(new_div)
        new_div = 0;
    end

end
