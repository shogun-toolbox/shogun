srcpath= './';
dstpath= './';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files=dir( [ srcpath '*.mod' ]);
for i=1:length(files),
    graph_hmm([ srcpath files(i).name] , [dstpath files(i).name '.ps' ] );
end
