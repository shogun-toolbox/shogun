function y = fix_kernel_name_inconsistency (kname)
	kname=upper(kname);
	if findstr('SIMPLELOCALITYIMPROVEDSTRING', kname)
		y='SLIK';
	elseif findstr('LOCALITYIMPROVEDSTRING', kname)
		y='LIK';
	elseif findstr('SPARSEGAUSSIAN', kname)
		y='GAUSSIAN';
	elseif findstr('SPARSEPOLY', kname)
		y='POLY';
	elseif findstr('SPARSELINEAR', kname)
		y='LINEAR';
	elseif findstr('WEIGHTEDDEGREEPOSITIONSTRING', kname)
		y='WEIGHTEDDEGREEPOS';
	elseif strcmp(kname, 'WEIGHTEDCOMMWORDSTRING')==1
		y='WEIGHTEDCOMMSTRING';
	elseif findstr('COMMULONGSTRING', kname)
		y='COMMSTRING';
	elseif findstr('COMMWORDSTRING', kname)
		y='COMMSTRING';
	elseif findstr('WORDSTRING', kname)
		pos=findstr('WORDSTRING', kname);
		y=kname(1:pos-1);
	elseif findstr('STRING', kname)
		pos=findstr('STRING', kname);
		y=kname(1:pos-1);
	elseif findstr('WORD', kname)
		pos=findstr('WORD', kname);
		y=kname(1:pos-1);
	else
		y=kname;
	end
