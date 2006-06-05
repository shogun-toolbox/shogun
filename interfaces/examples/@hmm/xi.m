function xis = xi(hmm, fwd, bwd, m, obs)
%compute FULL xi variables
%for observation vector obs of size 1,T
xis=[];

for t=1:size(obs,2)
    for i=1:hmm.N,
	for j=1:hmm.N,
	    xis(t,i,j)=fwd(t,i)*a(i,j)*bwd(t+1,j)/m;
	end
    end
end
