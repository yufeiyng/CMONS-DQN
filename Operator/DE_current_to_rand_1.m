function Offspring = DE_current_to_rand_1(Problem, Population, Fitness, N)
    % DE/current-to-rand/1 for diversity (also used in niches)

    [proM, disM] = deal(1, 20);

    if isa(Population(1), 'SOLUTION')
        evaluated = true;
        PopDec = Population.decs;
    else
        evaluated = false;
        PopDec = Population;
    end

    D = Problem.D;

    Fm = [0.6, 0.8, 1.0];
    CRm = [0.1, 0.2, 1.0];
    index = randi([1, length(Fm)], N, 1);
    F = Fm(index);
    F = F';
    F = F(:, ones(1, D));
    index = randi([1, length(CRm)], N, 1);
    CR = CRm(index);
    CR = CR';
    %% Differental evolution
    Site = rand(N, D) < repmat(CR, 1, D);

    Offspring = PopDec(1:N, :);

    Mate = TournamentSelection(2, 3 * N, Fitness);
    Parent1 = PopDec(Mate(1:N), :);
    Parent2 = PopDec(Mate(N + 1:2 * N), :);
    Parent3 = PopDec(Mate(2 * N + 1:3 * N), :);

    Offspring(Site) = Offspring(Site) + rand .* (Parent1(Site) - Offspring(Site)) + F(Site) .* (Parent2(Site) - Parent3(Site));

    %% Polynomial mutation
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

    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end

end
