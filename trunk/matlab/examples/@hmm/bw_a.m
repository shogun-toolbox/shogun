function new_hmm = bw_a(hmm, xis, gammas)
%baum welch estimate a's 
new_hmm=hmm;
for i=1:hmm.N,
    for j=1:hmm.N,
	n=0;
	d=0;
	for t=1:size(obs,2)-1
	    n=n+xis(t,i,j)
	    d=d+gamms(t,i)
	end
	new_hmm.a(i,j)=n/d;
    end
end
