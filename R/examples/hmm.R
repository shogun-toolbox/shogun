library(graph)
library(Rgraphviz)
library(sg)

cube=list(NULL,NULL,NULL)
num=vector(mode='numeric',length=18)+100
num[1]=0;
num[2]=0;
num[3]=0;
num[10]=0;
num[11]=0;
num[12]=0;

for (c in 1:3)
{
	for (i in 1:6)
		cube[[c]] <- c(cube[[c]], vector(mode='numeric',length=num[(c-1)*6+i])+i)
	cube[[c]] <- sample(cube[[c]],300,replace=TRUE);
}

x <- c(cube[[1]],cube[[2]],cube[[3]])
x <- paste(x,sep="",collapse="")

sg('set_features','TRAIN',x)
sg('send_command','convert TRAIN STRING CHAR STRING WORD CUBE 1')

#train 10 HMM models 
liks=vector(mode='numeric', length=10)-Inf
models=vector(mode='pairlist', length=10)
for (i in 1:10)
{
	sg('send_command','new_hmm 3 6')
	sg('send_command','bw')
	liks[[i]] <- sg('hmm_likelihood')
	models[[i]] <- sg('get_hmm')
}

#choose the most likely model and compute viterbi path
h=models[[which(liks==max(liks))]]
sg('set_hmm',h)
sg('set_features','TEST',x)
sg('send_command','convert TEST STRING CHAR STRING WORD CUBE 1')
path=sg('get_viterbi_path',0)

p=exp(h$p)
q=exp(h$q)
a=exp(h$a)
b=exp(h$b)

y=c(vector(mode='numeric', length(cube[[1]])),vector(mode='numeric', length(cube[[2]]))+1, vector(mode='numeric', length(cube[[2]]))+2)
matplot(1:length(y), y-0.01,type='l',col='red')
matplot(1:length(path$path), path$path,type='l',col='blue',add=T)
g=new("graphAM",a>1e-6,edgemode = "directed")
plot(g, "neato")
