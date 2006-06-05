clear classes;
N=2;
M=4;
ORDER=1;

p=[-7.461482e-01,-6.428144e-01];

q=[-5.971810e-01,-7.993101e-01];

a=[ 
    [-4.308731e-01,-1.049655e+00];
    [-1.281511e+00,-3.252005e-01];
];

b=[ 
   [-9.080200e-01,-3.580255e+00,-8.321478e-01,-2.012189e+00];
   [-6.587098e-01,-9.259533e-01,-4.049287e+00,-2.675200e+00];
];

obs=double(['AGAA';'AGAT']);

p=exp(p);
q=exp(q);
a=exp(a);
b=exp(b);

obs(find(obs==double('A')))=0+1;
obs(find(obs==double('C')))=1+1;
obs(find(obs==double('G')))=2+1;
obs(find(obs==double('T')))=3+1;

hmm=hmm(N,M,p,q,a,b);
[ fwd1, fm1 ] =forward(hmm,obs(1,:))
[ bwd1, bm1 ] =backward(hmm,obs(1,:))
[ fwd2, fm2 ] =forward(hmm,obs(2,:))
[ bwd2, bm2 ] =backward(hmm,obs(2,:))
