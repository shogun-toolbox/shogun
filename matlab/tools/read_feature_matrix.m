fname='/short/x46/sonne/svm/topfeatures.orig'
fid=fopen(fname, 'r');
intlen=fread(fid, 1, 'char')
doublelen=fread(fid, 1, 'char')
intstring=[ 'int' int2str(8*intlen) ]
doublestring=[ 'float' int2str(8*doublelen) ]
endian=fread(fid, 1, intstring)
fourcc=fread(fid, 1, intstring)
num_vec=fread(fid, 1, intstring)
num_feat=fread(fid, 1, intstring)
preprocd=fread(fid, 1, intstring)

feature_matrix=fread(fid, num_vec*num_feat, doublestring);
feature_matrix=reshape(feature_matrix, num_vec, num_feat);
labels=fread(fid, num_vec, intstring);
