CFeatures::CFeatures()
{
    preproc=NULL;
    num_vectors=0;
    num_features=0;
    feature_matrix=NULL;
}

CFeatures::~CFeatures()
{
	delete preproc;
	delete feature_matrix;
}

/// get feature vector for sample num
const REAL* CFeatures::get_feature_vector(int num)
{
#ifdef DEBUG
	if (num<num_vectors && feature_matrix)
#endif
		return feature_matrix[num*num_features];
	else
	{
		if (preproc)
			apply_to_feature_vector(get_feature_vector_comp(int num), len);
		else
			return get_feature_vector_comp(int num);
	}
#ifdef DEBUG
	else
		return CMath::INFTY;
#endif
}

/// set feature vector for sample num
bool CFeatures::set_feature_vector(int num)
{
#ifdef DEBUG
	if (num<num_vectors)
	{
#endif
		if (feature_matrix)
		{
			for (int i=num; i<num+features; i++)
				feature_matrix[num*num_features];
		}
#ifdef DEBUG
	}
	else
		return CMath::INFTY;
#endif
}

/// get the pointer to the feature matrix
const REAL* CFeatures::get_feature_matrix(int &num_feat, int &num_vectors)
{
	num_feat=num_features;
	num_vec=num_vectors;
	return feature_matrix;
}

/// set preprocessor
void CFeatures::set_preproc(CPreProc* p)
{ 
	delete preproc; 
	preproc=p 
}

/// set current preprocessor
CPreProc* CFeatures::get_preproc()
{ 
	return preproc;
}

/// preproc feature_matrix
bool CFeatures::preproc_feature_matrix()
{
	if (preproc)
		preproc->do_preproc(this);
}
#endif
