#ifndef _CREALPREPROC__H__
#define _CREALPREPROC__H__

#include "features/Features.h"
#include "lib/common.h"
#include "preproc/PreProc.h"

#include <stdio.h>


class CRealPreProc : public CPreProc
{
public:
	CRealPreProc(const char *name);
	virtual ~CRealPreProc();
	
	/// initialize preprocessor from file
	virtual bool load_init_data(FILE* src)=0;
	/// save init-data (like transforamtion matrices etc) to file
	virtual bool save_init_data(FILE* dst)=0;

	/// apply preproc on feature matrix
	/// result in feature matrix
	/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
	virtual REAL* apply_to_feature_matrix(CFeatures* f)=0;

	/// apply preproc on single feature vector
	/// result in feature matrix

	virtual REAL* apply_to_feature_vector(REAL* f, int &len)=0;
	
	/** return feature type with which objects derived 
	*         from CPreProc can deal
	*             */
	virtual EType get_feature_type() { return F_REAL; }
};
#endif
