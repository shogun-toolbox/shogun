% Blind Source Separation using the Jade Algorithm with Shogun
%
% Based on the example from scikit-learn
% http://scikit-learn.org/
%
% Kevin Hughes 2013

% Generate sample data
n_samples = 2000;
time = linspace(0,10,n_samples);

% Source Signals
S = zeros(2, length(time));
S(1,:) = sin(2*time);
S(2,:) = sign(sin(3*time));
S += 0.2*rand(size(S));

% Standardize data
S = S ./ (std(S,0,2) * ones(1,n_samples));

% Mixing Matrix
A = [1 0.5; 0.5 1]

% Mix Signals
X = A*S;
mixed_signals = X;

% Separating
sg('set_converter', 'jade');
sg('set_features', 'TRAIN', mixed_signals);
S_ = sg('apply_converter');


% Plot
figure();
subplot(311);
plot(time, S(1,:), 'b');
hold on;
plot(time, S(2,:), 'g');
set(gca, 'xtick', [])
title("True Sources");

subplot(312);
plot(time, X(1,:), 'b');
hold on;
plot(time, X(2,:), 'g');
set(gca, 'xtick', [])
title("Mixed Sources");

subplot(313);
plot(time, S_(1,:), 'b');
hold on;
plot(time, S_(2,:), 'g');
title("Estimated Sources");
