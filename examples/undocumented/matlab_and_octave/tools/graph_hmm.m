function hmm_graph(srcname, dstname, treshold, treshout, startendstates, outmatrix, startcolor, endcolor, ratio, nodesep, shape)
%function hmm_graph(srcname, dstname, treshold, treshout, startendstates, outmatrix, startcolor, endcolor, ratio, nodesep, shape)
%
% generate a state graph of a hmm
%
% srcname	- filename of some hmm.mod
% dstname	- destination postscript HMM file name (default: hmm_graph.ps)
% treshold	- treshold down to which bows are plotted (default: log(0.001))
% treshout	- treshout down to which outputs are included (default: log(0.001))
% outmatrix	- depending on the alphabet the M output symbols (default: [ 'A'; 'C'; 'G'; 'T' ])
% startcolor	- color of start states (default 'green')
% endcolor	- color of start states (default 'red')
% ratio	- default compress but can be 'fill' 'compress' 'none' 'auto'
% nodesep	- default=0.5
% shape		- default 'record' but can be 'circle' 'ellipse' 'record' 'doublecircle' 'diamond'
% startendstates- whether to include virtual start/stop states default 0 (set to 1 to include them)

if nargin<2,dstname='hmm_graph.dot'; end;
if nargin<3,treshold=log(0.001);end;
if nargin<4,treshout=log(0.001);end;
if nargin<5,startendstates=0;end;
if nargin<6,outmatrix=[ 'A'; 'C'; 'G'; 'T' ];end;
if nargin<7,startcolor='green';end;
if nargin<8,endcolor='red';end;
if nargin<9,ratio='compress';end;
if nargin<10,nodesep=0.5;end;
if nargin<11,shape='record';end;

unix([ 'ln -sf ' srcname ' a_particularly_nice_hmm.m' ]);
a_particularly_nice_hmm;
unix('rm -f a_particularly_nice_hmm.m');

psname=dstname;
tmpname=[ '/tmp/bla.dot' ];
fid=fopen(tmpname, 'w');
fprintf(fid,'digraph hmm {\npage="8,10.5";\nsize="7,9.5";\ncenter=true;\nnodesep=%f;\nratio=%s;\nnode [shape = record];\nedge [style=bold];\n',nodesep,ratio);

if startendstates==1,
    fprintf(fid,'start [style =filled, color=%s, label="start"];\n',startcolor);
    fprintf(fid,'end [style =filled, color=%s, label="end"];\n',endcolor);
end

for i=1:N,
    out='';
    for j=1:M,
	if b(i,j)>treshout,
	    out = sprintf('%s%c:%1.2f\\n',out,outmatrix(j),b(i,j));
	end
    end

    if startendstates==0,
	if q(i)>treshold,
	   fprintf(fid,'%d [ style = filled, color=%s, label = "n%d|%s" ];\n',i,endcolor,i,out);
	elseif p(i)>treshold,
	   fprintf(fid,'%d [ style = filled, color=%s, label = "n%d|%s" ];\n',i,startcolor,i,out);
	else
	   fprintf(fid,'%d [ label = "n%d|%s" ];\n',i,i,out);
	end
    else
	   fprintf(fid,'%d [ label = "n%d|%s" ];\n',i,i,out);
    end
end

for i=1:N,

    if startendstates==1
	if p(i)>treshold,
	    fprintf(fid, 'start -> %d [ label = "%1.2f"];\n',i,p(i));
	end
	if q(i)>treshold,
	    fprintf(fid, '%d -> end [ label = "%1.2f"];\n',i,q(i));
	end
    end

    for j=1:N,
        if a(i,j) > treshold,
	    fprintf(fid,'%d -> %d [ label = "%1.2f" ];\n',i,j,a(i,j));
	end
    end
end
fprintf(fid,'}\n');
fclose(fid);

unix([ 'dot ' ' -Tps ' tmpname ' -o' psname ]);
