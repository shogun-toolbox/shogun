function [matrix] = load_matrix(fname)
%function [matrix] = load_matrix(fname)
%
% load a matrix from file
%
% this implementation is prone to become rather slow with large data,
% but our expected toy data here is not large. :)
	fid=fopen(fname, 'r');
	matrix=[];
	while 1
		line=fgetl(fid);
		if ~ischar(line),   break,   end

		converted=str2num(line);
		if isempty(converted)
			matrix=[matrix, line'];
		else
			matrix=[matrix, converted'];
		end
	end
	fclose(fid);
