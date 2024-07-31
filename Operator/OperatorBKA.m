function Offspring = OperatorBKA(Problem, Population, Fitness)

    [~, rank] = sort(Fitness);
    NB = ceil(0.1 * length(Population));
    leaderPos = Population(rank(1:NB)).decs;

    Offspring = Population.decs;
    [N, D] = size(Offspring);

    for i = 1:N

        % Attacking behavior
        n = 0.05 * exp(-2 * (Problem.FE / Problem.maxFE) ^ 2);

        r = rand;

        if r > 0.9
            Offspring(i, :) = Offspring(i, :) + n .* (1 + sin(r)) * Offspring(i, :);
        else
            Offspring(i, :) = Offspring(i, :) .* (n * (2 * rand(1, D) - 1) + 1);
        end

        % Migration behavior
        m = 2 * sin(r + pi / 2);
        s = randi(length(Fitness));
        cauchy_value = tan((rand(1, D) - 0.5) * pi);

        if Fitness(i) < Fitness(s)
            Offspring(i, :) = Offspring(i, :) + cauchy_value(:, D) .* (Offspring(i, :) - leaderPos(randi(NB), :));
        else
            Offspring(i, :) = Offspring(i, :) + cauchy_value(:, D) .* (leaderPos(randi(NB), :) - m .* Offspring(i, :));
        end

    end

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
