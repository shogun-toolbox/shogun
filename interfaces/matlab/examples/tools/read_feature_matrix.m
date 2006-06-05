function [matrix, labels] = read_feature_matrix(fname)
%function [matrix, labels] = read_feature_matrix(fname)
%
% load feature matrix and labels
% from genefinder save_feature_matrix output
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

matrix=fread(fid, [ num_feat num_vec ], doublestring);
labels=fread(fid, num_vec, intstring);
