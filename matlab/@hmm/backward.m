function [ bwd, prob ] = backward(hmm, obs)
%compute FULL backward variables
%for observation vector obs of size 1,T
bwd=[];

%init
for i=1:hmm.N,
    bwd(size(obs,2),i)=hmm.q(i);
end

%induction
for t=size(obs,2)-1:-1:1,
    %%
    for i=1:hmm.N,
	s=0;
	for j=1:hmm.N,
	    s=s+hmm.a(i,j)*hmm.b(j,obs(t+1))*bwd(t+1,j);
	end
	bwd(t,i)=s;
    end
    %% this is the same upto %% bwd(t,:)= (hmm.a*(bwd(t+1,:)'.*hmm.b(:,obs(t+1))))';
end

prob=0;
for i=1:hmm.N,
    prob=prob+bwd(1,i)*hmm.p(i)*hmm.b(i,obs(1));
end

hmm.bwd=bwd;
hmm.probability=prob;
