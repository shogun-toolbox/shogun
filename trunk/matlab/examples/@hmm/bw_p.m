function new_hmm = bw_p(hmm, gammas)
%baum welch estimate a's 
new_hmm=hmm;

for i=1:hmm.N,
    new_hmm.p(i)=gamma(1,i);
end
