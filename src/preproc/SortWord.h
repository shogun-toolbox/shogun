#ifndef _CSORTWORD__H__
#define _CSORTWORD__H__

#include "WordPreProc.h"
#include "features/Features.h"
#include "lib/common.h"

#include <stdio.h>

class CSortWord : public CWordPreProc
{
public:
	CSortWord();
	virtual ~CSortWord();

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
	virtual WORD* apply_to_feature_matrix(CFeatures* f);

	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual WORD* apply_to_feature_vector(WORD* f, INT &len);
};
#endif
