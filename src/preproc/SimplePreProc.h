#ifndef _CSIMPLEPREPROC__H__
#define _CSIMPLEPREPROC__H__

#include "features/Features.h"
#include "lib/common.h"
#include "preproc/PreProc.h"

#include <stdio.h>


template <class ST> class CSimplePreProc : public CPreProc
{
public:
	CSimplePreProc(const char *name, const char* id) : CPreProc(name,id)
	{
	}

	/// apply preproc on feature matrix
	/// result in feature matrix
	/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
	virtual ST* apply_to_feature_matrix(CFeatures* f)=0;

	/// apply preproc on single feature vector
	/// result in feature matrix

	virtual ST* apply_to_feature_vector(ST* f, int &len)=0;
};
#endif
