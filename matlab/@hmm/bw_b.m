function new_hmm = bw_b(hmm, gammas, obs)
%baum welch estimate a's 
new_hmm=hmm;
for j=1:hmm.N,
    for k=1:hmm.M,
	n=0;
	d=0;
	for t=1:size(obs,2)
	    if (obs(t)==k),
		n=n+gammas(t,j,k)
	    end
	    d=d+gamms(t,k)
	end
	new_hmm.b(i,j)=n/d;
    end
end
