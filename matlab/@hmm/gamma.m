function gammas = gamma(hmm, xis)
%compute FULL gamma variables
%for observation vector obs of size 1,T
gammas=[];

for t=1:size(obs,2)
    for i=1:hmm.N,
	s=0;
	for j=1:hmm.N,
	    s==xis(t,i,j);
	end
	gammas(t,i)=s;
    end
end
