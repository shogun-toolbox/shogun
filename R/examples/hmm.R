#library(graph)
#library(Rgraphviz)

library(sg)
sg('send_command','loglevel ALL')

cube=list(NULL,NULL,NULL)
num=vector(mode='numeric',length=12)+100
num[1]=10;
num[2]=10;
num[3]=10;

for (c in 1:2)
	for (i in 1:6)
		cube[[c]] <- c(cube[[c]], vector(mode='numeric',length=num[(c-1)*6+i])+i)
	cube[[c]] <- sample(cube[[c]],300,replace=TRUE);

x <- c(cube[[1]],cube[[2]])
x <- paste(x,sep="",collapse="")

sg('set_features','TRAIN',x)
sg('send_command','convert TRAIN STRING CHAR STRING WORD CUBE 1')
sg('send_command','new_hmm 2 6')
sg('send_command','bw')
h=sg('get_hmm')
sg('set_features','TEST',x)
sg('send_command','convert TEST STRING CHAR STRING WORD CUBE 1')
sg('send_command','best_path 0 1000')

p=exp(h$p)
q=exp(h$q)
a=exp(h$a)
b=exp(h$b)

path=sg('get_viterbi_path',0)
y=c(vector(mode='numeric', length(cube[[1]])),vector(mode='numeric', length(cube[[2]]))+1)
matplot(1:length(y), y-0.01,type='l',col='red')
matplot(1:length(path$path), path$path,type='l',col='blue',add=T)


g=new("graphAM",a>0.1,edgemode = "directed")
plot(g, "neato")
