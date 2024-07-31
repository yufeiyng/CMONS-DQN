function Offspring = OperatorHOA(Problem, Population, Fitness)

    POpDec = Population.decs;
    Offspring = POpDec;
    [N, D] = size(Offspring);

    [~, rank] = sort(Fitness);
    % Best = Population(rank(1:ceil(0.1 * N))).decs;
    Best = Population(rank(1)).decs;
    Best = repmat(Best, ceil(N / size(Best, 1)), 1);
    Best = Best(1:N, :);

    s = tan(randi([0 50], N, 1)); % randomize elevation angle of hiker
    SF = randi([1 2], N, 1);
    Vel = 6 .* exp(-3.5 .* abs(s + 0.05)); % Compute walking velocity based on Tobler's Hiking Function
    newVel = Vel + rand(1, D) .* (Best - SF .* Offspring); % determine new position of hiker
    Offspring = Offspring + newVel;

    %% Polynomial mutation
    [proM, disM] = deal(1, 20);
    Lower = repmat(Problem.lower, N, 1);
    Upper = repmat(Problem.upper, N, 1);
    Site = rand(N, D) < proM / D;
    mu = rand(N, D);
    temp = Site & mu <= 0.5;
    Offspring = min(max(Offspring, Lower), Upper);
    Offspring(temp) = Offspring(temp) + (Upper(temp) - Lower(temp)) .* ((2 .* mu(temp) + (1 - 2 .* mu(temp)) .* ...
        (1 - (Offspring(temp) - Lower(temp)) ./ (Upper(temp) - Lower(temp))) .^ (disM + 1)) .^ (1 / (disM + 1)) - 1);
    temp = Site & mu > 0.5;
    Offspring(temp) = Offspring(temp) + (Upper(temp) - Lower(temp)) .* (1 - (2 .* (1 - mu(temp)) + 2 .* (mu(temp) - 0.5) .* ...
        (1 - (Upper(temp) - Offspring(temp)) ./ (Upper(temp) - Lower(temp))) .^ (disM + 1)) .^ (1 / (disM + 1)));

end
