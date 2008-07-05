function y = fix_kernel_name_inconsistency (kname)
	kname=toupper(kname);
	if findstr('LOCALITYIMPROVEDSTRING', kname)
		y='LIK';
	elseif findstr('SIMPLELOCALITYIMPROVEDSTRING', kname)
		y='SLIK';
	elseif findstr('WORDMATCH', kname)
		y='MATCH';
	elseif findstr('WEIGHTEDDEGREEPOSITIONSTRING', kname)
		y='WEIGHTEDDEGREEPOS';
	elseif findstr('COMMULONGSTRING', kname)
		y='COMMSTRING';
	elseif findstr('COMMWORDSTRING', kname)
		y='COMMSTRING';
	elseif findstr('WEIGHTEDCOMMWORDSTRING', kname)
		y='WEIGHTEDCOMMSTRING';
	elseif findstr('STRING', kname)
		pos=findstr('STRING', kname);
		y=kname(1:pos);
	elseif findstr('WORD', kname)
		pos=findstr('WORD', kname);
		y=kname(1:pos);
	else
		y=kname;
	end

