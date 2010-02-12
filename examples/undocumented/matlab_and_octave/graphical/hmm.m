clear all;
leng=50;
rep=5;
weight=1;
num_hmms=30;

% generate a sequence with characters 1-6 drawn from 3 loaded cubes
for i = 1:3,
    a{i}= [ ones(1,ceil(leng*rand)) 2*ones(1,ceil(leng*rand)) 3*ones(1,ceil(leng*rand)) 4*ones(1,ceil(leng*rand)) 5*ones(1,ceil(leng*rand)) 6*ones(1,ceil(leng*rand)) ];
    a{i}= a{i}(randperm(length(a{i})));
end

s=[];
for i = 1:size(a,2),
    s= [ s i*ones(1,ceil(rep*rand)) ];
end
s=s(randperm(length(s)));
sequence={''};
for i = 1:length(s),
    f(i)=ceil(((1-weight)*rand+weight)*length(a{s(i)}));
    t=randperm(length(a{s(i)}));
    r=a{s(i)}(t(1:f(i)));
    sequence{1}=[sequence{1} char(r+'0')];
end

% train 30 hmms on sequence with pseudocount 1e-10
sg('loglevel', 'ERROR');
sg('pseudo', 1e-10);
hmms=[];
liks=[];
for i=1:num_hmms,
	sg('new_hmm', 3, 6);
	sg('set_features','TRAIN',sequence,'CUBE');
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', 1);
	sg('bw');
	[hmms(i).p, hmms(i).q, hmms(i).a, hmms(i).b]=sg('get_hmm');
	hmms(i).lik=sg('hmm_likelihood');
end

% take the most likely one
[v,idx]=max([hmms.lik]);
 sg('set_hmm', hmms(idx).p,hmms(idx).q,hmms(idx).a,hmms(idx).b);

% compute viterbi path
sg('set_features','TEST',sequence,'CUBE');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', 1);
[path,lik]=sg('get_viterbi_path',0);
path=path+1; %path is zero based in shogun but one based in matlab


% fancy output when which cube was estimated to be drawn
f_est=[]; s_est=[]; f_est=1; s_est=path(1); i=1;

for j=2:length(path),
	if s_est(i)==path(j)
		f_est(i)=f_est(i)+1;
	else
		i=i+1;
		f_est = [ f_est 1 ];
		s_est = [ s_est path(j) ];
	end
end

fprintf('estimated:\n');
fprintf('========\n\n');
fprintf('sequence:\n');
for i = 1:length(s_est),
    fprintf('\tcube: %dx%d\n',f_est(i), s_est(i));
end
for i= 1:size(a,2),
    fprintf('Estimated Distribution: Dice %d:', i);
    fprintf('%f ', exp(hmms(idx).b(i,:)));
    fprintf('\n');
end

fprintf('solution:\n');
fprintf('========\n\n');
fprintf('sequence:\n');
for i = 1:length(s),
    fprintf('\tcube: %dx%d\n',f(i), s(i));
end
for i= 1:size(a,2),
    fprintf('Distribution: Cube %d:', i);
    fprintf('%f ', [sum(a{i}==1),sum(a{i}==2),sum(a{i}==3),sum(a{i}==4),sum(a{i}==5),sum(a{i}==6)]/length(a{i}));
    fprintf('\n');
end

%plot the segmentation
truepath=[];
for i = 1:length(s),
	truepath = [truepath ones(1,f(i))*s(i)];
end
figure(1);
clf
plot(path,'r-')
hold on
plot(truepath+0.02,'b-')
legend('estimated path', 'true path');
