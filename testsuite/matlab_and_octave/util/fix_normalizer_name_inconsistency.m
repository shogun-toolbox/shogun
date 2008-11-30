function y=fix_normalizer_name_inconsistency(name)
	if strcmp(name, 'IdentityKernelNormalizer')==1
		y='IDENTITY';
	elseif strcmp(name, 'AvgDiagKernelNormalizer')==1
		y='AVGDIAG';
	elseif strcmp(name, 'SqrtDiagKernelNormalizer')==1
		y='SQRTDIAG';
	elseif strcmp(name, 'FirstElementKernelNormalizer')==1
		y='FIRSTELEMENT';
	end
