#ifndef _CPREPROC__H__
#define _CPREPROC__H__

#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CFeatures;

class CPreProc
{
public:
	CPreProc();
	virtual ~CPreProc();

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f)=0;
	/// initialize preprocessor from file
	virtual bool load(FILE* f)=0;
	/// save preprocessor init-data to file
	virtual bool save(FILE* f)=0;

	/// apply preproc on feature matrix
	/// result in feature matrix
	/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
	virtual REAL* apply_to_feature_matrix(CFeatures* f)=0;
	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual REAL* apply_to_feature_vector(REAL* f, int len)=0;
};
#endif
