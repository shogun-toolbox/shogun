function [fp, tp]=readroc(rocfilename)
%function plotroc(rocfilename)
%
% plot roc from genefinder rocfile
%
% rocfilename - filename to genefinder rocfile
% plotstring  - e.g. 'r-' for red line, default 'b-'

if nargin>0,
    fid=fopen(rocfilename, 'r','a');
    if fid ~= -1,
	[ id count ] =fread(fid, 4, 'char');
	if count == 4 & id(1) == 'R' & id(2) == 'O' & id(3) == 'C',
	    fpos_cur=ftell(fid);
	    fseek(fid, 0, 'eof');
	    fpos_end=ftell(fid);
	    fseek(fid, fpos_cur, 'bof');
	    length=(fpos_end-fpos_cur)/8/2;
	    fp=fread(fid, length, 'double');
	    tp=fread(fid, length, 'double');
	end;
	fclose(fid);
    end;
end;
