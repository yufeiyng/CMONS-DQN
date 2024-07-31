function Offspring = OperatorDCS(Problem, Population, Fitness)

    lb = Problem.lower;
    ub = Problem.upper;
    PopDec = Population.decs;
    Offspring = PopDec;
    [N, D] = size(Offspring);
    eta_qKR = zeros(1, N);

    % Golden ratio
    gr = 2 / (1 + sqrt(5));

    % High-performing individuals
    ngS = max(6, round(N * (gr / 3)));
    pc = 0.5;

    % Ranking-based self-improvement
    phi_qKR = 0.25 + 0.55 * ((0 + ((1:N) / N)) .^ 0.5);

    % Reset
    [~, rank] = sort(Fitness);
    NB = ceil(0.1 * length(Population));

    % Compute social impact factor
    lamda_t = 0.1 + (0.518 * ((1 - (Problem.FE / Problem.maxFE) ^ 0.5)));

    for i = 1:N

        % Compute differentiated knowledge-acquisition rate
        eta_qKR(i) = (round(rand * phi_qKR(i)) + (rand <= phi_qKR(i))) / 2;
        jrand = floor(D * rand + 1);

        if i == N && rand < pc
            % Low-performing
            Offspring(i, :) = lb + rand * (ub - lb);

        elseif i <= ngS
            % High-performing
            while true, r1 = round(N * rand + 0.5); if r1 ~= i && r1 ~= rank(1), break, end, end

            for d = 1:D

                if rand <= eta_qKR(i) || d == jrand
                    Offspring(i, d) = PopDec(r1, d) + LnF3(gr, 0.05, 1, 1);
                end

            end

        else
            % Average-performing

            while true, r1 = round(N * rand + 0.5); if r1 ~= i && r1 ~= rank(1), break, end, end
            while true, r2 = ngS + round((N - ngS) * rand + 0.5); if r2 ~= i && r2 ~= rank(1) && r2 ~= r1, break, end, end

            % Compute learning ability
            omega_it = rand;

            for d = 1:D

                if rand <= eta_qKR(i) || d == jrand
                    Offspring(i, d) = PopDec(rank(1), d) + ((PopDec(r2, d) - PopDec(i, d)) * lamda_t) + ((PopDec(r1, d) - PopDec(i, d)) * omega_it);
                end

            end

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

function Y = LnF3(alpha, sigma, m, n)
    Z = laplacernd(m, n);
    Z = sign(rand(m, n) - 0.5) .* Z;
    ub = rand(m, n);
    R = sin(0.5 * pi * alpha) .* tan(0.5 * pi * (1 - alpha * ub)) - cos(0.5 * pi * alpha);
    Y = sigma * Z .* (R) .^ (1 / alpha);
end

function x = laplacernd(m, n)
    u1 = rand(m, n);
    u2 = rand(m, n);
    x = log(u1 ./ u2);
end
