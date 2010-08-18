c=ones(10,1);
H=eye(10);
A=zeros(10);
A=ones(1,10);
b=1;
l=zeros(10,1);
u=ones(10,1);

%tic;
%[x,y] = pr_loqo2(c, H, A, b, l, u);
%toc

tic;
[x2,y2] = sg('pr_loqo',c', H, A, b, l', u');
toc
