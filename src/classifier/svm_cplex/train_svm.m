function [w,b]=train_svm(XT, LT, C)
% [w,b]=train_svm(XT, LT, C)

global lpenv ;

[dim,ell]=size(XT) ;

if isempty(lpenv),
  lpenv=0 ;
end ;
if lpenv==0,
  lpenv=cplex_init(1) ;
end ;

INF=1e20 ;
%      b        w,                xi
LB= [ -INF;  -INF*ones(dim,1); zeros(ell,1)] ;
UB= [ INF;    INF*ones(dim,1); INF*ones(ell,1)] ;
Q=sparse(0) ;
Q(dim+ell+1,dim+ell+1)=0 ;
for i=1:dim, Q(1+i,1+i)=1; end;
%Q(2:1+dim,2:1+dim)=speye(dim) ;

f=[zeros(dim+1,1); C*ones(ell,1)] ;
A=sparse([-LT' -spdiag(LT)*XT' -eye(ell)]) ;
b=-ones(ell,1) ;
[res,lambda]=qp_solve(lpenv, Q, f, A, b, LB, UB, 0, 1) ;

b=res(1) ;
w=res(2:dim+1)' ;

%out=w*XT+b

%cplex_quit(lpenv,1)
