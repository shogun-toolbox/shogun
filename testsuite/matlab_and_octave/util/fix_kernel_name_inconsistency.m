function y = fix_kernel_name_inconsistency (kname)
	kname=toupper(kname);
	if findstr('SIMPLELOCALITYIMPROVEDSTRING', kname)
		y='SLIK';
	elseif findstr('LOCALITYIMPROVEDSTRING', kname)
		y='LIK';
	elseif findstr('WORDMATCH', kname)
		y='MATCH';
	elseif findstr('WEIGHTEDDEGREEPOSITIONSTRING', kname)
		y='WEIGHTEDDEGREEPOS';
	elseif strcmp(kname, 'WEIGHTEDCOMMWORDSTRING')==1
		y='WEIGHTEDCOMMSTRING';
	elseif findstr('COMMULONGSTRING', kname)
		y='COMMSTRING';
	elseif findstr('COMMWORDSTRING', kname)
		y='COMMSTRING';
	elseif findstr('STRING', kname)
		pos=findstr('STRING', kname);
		y=kname(1:pos-1);
	elseif findstr('WORD', kname)
		pos=findstr('WORD', kname);
		y=kname(1:pos-1);
	else
		y=kname;
	end

