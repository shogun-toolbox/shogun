f=fopen('rawdat.dat', 'r');
[d cnt]=fread(f, inf, 'double');
dat=reshape(d, [ 120 1000]);
fclose(f);

f=fopen('rawdat.tst', 'r');
[d cnt]=fread(f, inf, 'double');
testdat=reshape(d, [ 120 1000]);
fclose(f);

f=fopen('procd.dat', 'r');
[d cnt]=fread(f, inf, 'double');
prcddat=reshape(d, [ 93 1000]);
fclose(f);

f=fopen('tprocd.dat', 'r');
[d cnt]=fread(f, inf, 'double');
tprcddat=reshape(d, [ 93 1000]);
fclose(f);

dat=log(1+dat);
tdat=log(1+testdat);

for i=1:size(dat,1),
	m=mean(dat(i,:));
	dat(i,:)=dat(i,:)-m;
	tdat(i,:)=tdat(i,:)-m;
end

v=zeros(1,size(dat,1));
for i=1:size(dat,1),
	v(i)=mean((dat(i,:)).^2);
end

idx=find(v>1e-6);

findat=dat(idx,:);
tfindat=tdat(idx,:);

max(abs(findat(:)-prcddat(:)))
max(abs(tfindat(:)-tprcddat(:)))

sum=0;
len=size(findat,2);
for i=1:size(findat,2),
	sum=sum+dot(findat(:,i),findat(:,i))^5;
end

fac=sum/size(findat,2)

K=zeros(len);
KE=zeros(len);

XT=findat;
XTE=tfindat;

K   = (XT'*XT).^5 ;
KE   = (XT'*XTE).^5 ;

K=K/fac;
KE=KE/fac;

f=fopen('rawlab.dat', 'r');
[LT cnt]=fread(f, inf, 'int32');
fclose(f);

f=fopen('rawlab.tst', 'r');
[LTE cnt]=fread(f, inf, 'int32');
fclose(f);

LT=LT';
LTE=LTE';

f=fopen('kernel', 'r');
[d cnt]=fread(f, inf, 'double');
kern=reshape(d, [ 1000 1000]);
fclose(f);

max(abs((K(:)-kern(:))))

f=fopen('testkernel', 'r');
[d cnt]=fread(f, inf, 'double');
testkern=reshape(d, [ 1000 1000]);
fclose(f);

max(abs((KE(:)-testkern(:))))

C_qp=10;

KY = (LT'*LT).*K+1e-17*eye(size(K)) ;

A  = sparse(LT) ;
b  = 0 ;
INF= 1e20 ;
LB = zeros(1000,1) ;
UB = C_qp*ones(1000,1);
c  = -ones(1000,1) ;
Q  = sparse(KY) ;

addpath /home/schnarch/sonne/CANDY/matlab/cplex
global lpenv
if isempty(lpenv)
  lpenv=cplex_init(0) ;
end ;  

[sol,lambda,how]=qp_solve(lpenv,Q,c,A,b,LB,UB,1,1) ;

alpha = sol.*LT' ;
b = -lambda ;
tr_out = alpha'*K + b ;
te_out = alpha'*KE + b;
ERR_tr=mean(sign(tr_out)~=LT)
ERR_te=mean(sign(te_out)~=LTE)

addpath /home/schnarch/sonne/CANDY/matlab/data
addpath /home/schnarch/sonne/CANDY/matlab/svm
addpath /home/schnarch/sonne/CANDY/matlab/rbf
addpath /home/schnarch/sonne/CANDY/matlab/svm/kernels

dataset=data(XT,LT) ;
dot='bla';
sv=svm(C_qp, dot, 'lightK2', 1) ;
options.e=1e-6 ;
options.q=50 ;
sv=do_learn(sv, dataset, K, options) ;

alphas_lm=get_alpha(sv, get_train(dataset,1)) ;
b_lm=get_b(sv) ;
tr_out_lm=alphas_lm'*K - b_lm;
