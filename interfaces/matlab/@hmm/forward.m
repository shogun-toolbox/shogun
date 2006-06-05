function [ fwd, prob ] = forward(hmm, obs)
%compute FULL forward variables
%for observation vector obs of size 1,T
fwd=[];

%init
for i=1:hmm.N,
    fwd(1,i)=hmm.p(i)*hmm.b(i,obs(1));
end

%induction
for t=1:size(obs,2)-1,
    for j=1:hmm.N,
	s=0;
	for i=1:hmm.N,
	    s=s+fwd(t,i)*hmm.a(i,j)*hmm.b(j,obs(t+1));
	end
	fwd(t+1,j)=s;
    end
end

prob=0;
for i=1:hmm.N,
    prob=prob+fwd(size(obs,2),i)*hmm.q(i);
end
hmm.fwd=fwd;
hmm.probability=prob;

