#ifndef _CNORM_DERIVATIVE_LEM3__H__
#define _CNORM_DERIVATIVE_LEM3__H__

#include "preproc/RealPreProc.h"
#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CNormDerivativeLem3 : public CRealPreProc
{
public:
	CNormDerivativeLem3();
	virtual ~CNormDerivativeLem3();

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f);
	/// cleanup
	virtual void cleanup();
	/// initialize preprocessor from file
	virtual bool load(FILE* f);
	/// save preprocessor init-data to file
	virtual bool save(FILE* f);

	/// apply preproc on feature matrix
	/// result in feature matrix
	/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
	virtual REAL* apply_to_feature_matrix(CFeatures* f);

	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual REAL* apply_to_feature_vector(REAL* f, INT len);
};
#endif
