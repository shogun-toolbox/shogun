#ifndef _CSORTWORDSTRING__H__
#define _CSORTWORDSTRING__H__

#include "features/StringFeatures.h"
#include "StringPreProc.h"
#include "lib/common.h"

#include <stdio.h>

class CSortWordString : public CStringPreProc<WORD>
{
public:
	CSortWordString();
	virtual ~CSortWordString();

	virtual EFeatureType get_feature_type() { return F_WORD; }

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f);

	/// initialize preprocessor from file
	virtual bool load_init_data(FILE* src);
	/// save init-data (like transforamtion matrices etc) to file
	virtual bool save_init_data(FILE* dst);
	/// cleanup
	virtual void cleanup();
	/// initialize preprocessor from file
	virtual bool load(FILE* f);
	/// save preprocessor init-data to file
	virtual bool save(FILE* f);

	/// apply preproc on feature matrix
	/// result in feature matrix
	/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
	virtual bool apply_to_feature_strings(CFeatures* f);

	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual WORD* apply_to_feature_string(WORD* f, INT &len);
};
#endif
