function y = fix_normalization_inconsistency (normalization)
	if (normalization==1)
		y='SQRT';
	elseif (normalization==2)
		y='FULL';
	elseif (normalization==3)
		y='SQRTLEN';
	elseif (normalization==4)
		y='LEN';
	elseif (normalization==5)
		y='SQLEN';
	else
		y='NO';
	end
