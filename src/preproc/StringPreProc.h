#ifndef _CSTRINGPREPROC__H__
#define _CSTRINGPREPROC__H__

#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CFeatures;

class CStringPreProc : CPreProc
{
public:
	CStringPreProc();
	virtual ~CStringPreProc();

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f)=0;
	/// cleanup
	virtual void cleanup()=0;
	/// initialize preprocessor from file
	virtual bool load(FILE* f)=0;
	/// save preprocessor init-data to file
	virtual bool save(FILE* f)=0;

	/// apply preproc on feature matrix
	/// result in feature matrix
	/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
	virtual char* apply_to_feature_matrix(CFeatures* f)=0;
	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual char* apply_to_feature_vector(char* f, int len)=0;
	
	/** return feature type with which objects derived 
	*         from CPreProc can deal
	*             */
	virtual CFeatures::EType get_feature_type() { return CFeatures::F_STRING; }
};
#endif
