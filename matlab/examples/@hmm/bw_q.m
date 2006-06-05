function new_hmm = bw_q(hmm, gammas, obs)
%baum welch estimate a's 
new_hmm=hmm;

for i=1:hmm.N,
    new_hmm.p(i)=gamma(size(obs,2),i);
end
