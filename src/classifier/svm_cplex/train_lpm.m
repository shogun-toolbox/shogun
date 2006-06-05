if lpenv==0,
  lpenv=cplex_init(1) ;
end ;
Cs = logspace(log10(0.0001),log10(100),10);
val_trerr=[]; val_teerr=[] ;


for citer = 1:length(Cs)
  C=Cs(citer) ;

  ell=get_train_size(dataset) ;
  dim=get_idim(dataset) ;
  [XT,LT]=get_train(dataset) ;

  INF=1e20 ;
  %      b        w^+,           q^-                xi
  LB= [ -INF;  zeros(dim,1);    zeros(dim,1);      zeros(ell,1)] ;
  UB= [ INF;   INF*ones(dim,1); INF*ones(dim,1); INF*ones(ell,1)] ;
  f=[0; ones(dim*2,1)/dim; C*ones(ell,1)/ell] ;
  A=sparse([-LT' -spdiag(LT)*XT' spdiag(LT)*XT' -eye(ell)]) ;
  b=-ones(ell,1) ;
  [res,lambda]=lp_solve(lpenv, f, A, b, LB, UB, 0, 1,'dual') ;

  b=res(1) ;
  w=res(2:dim+1)'-res(2+dim:2*dim+1)' ;
  xis=res(2*dim+2:end)' ;

  tr_out=w*XT+b ;
  val_trerr(citer)=mean(sign(tr_out)~=LT)

  [XTE,LTE]=get_test(dataset) ;
  te_out=w*XTE+b ;
  val_teerr(citer)=mean(sign(te_out)~=LTE)
  clear XTE LTE

end ;

