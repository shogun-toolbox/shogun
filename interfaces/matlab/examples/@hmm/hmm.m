function h=hmm(N,M,p,q,a,b)
%constructor for hmm class
if nargin<6,
    disp('@hmm/hmm.m wrong number of args');
else
    h.N=N;
    h.M=M;
    h.p=p;
    h.q=q;
    h.a=a;
    h.b=b;
    h.fwd=[];
    h.bwd=[];
    h.probability=0;
    h=class(h, 'hmm');
end
