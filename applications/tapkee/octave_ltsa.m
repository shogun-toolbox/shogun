n = 1000;
noise = 0.0;
t = (3 * pi / 2) * (1 + 2 * rand(n, 1));
height = 30 * rand(n, 1);
X = [t .* cos(t) height t .* sin(t)] + noise * randn(n, 3);

sg('set_features','TRAIN',X');
sg('set_converter','ltsa',10);
embedding = sg('embed',2);
plot(embedding(:,1),embedding(:,2),'@');

