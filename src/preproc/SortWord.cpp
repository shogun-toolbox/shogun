#include "SortWord.h"
#include "WordPreProc.h"
#include "features/Features.h"
#include "features/WordFeatures.h"
#include <math.h>

CSortWord::CSortWord() : CWordPreProc("SortWord", "SRTW") 
{
}

CSortWord::~CSortWord()
{
}

/// initialize preprocessor from features
bool CSortWord::init(CFeatures* f)
{
	assert(f->get_feature_class() == C_SIMPLE);
	assert(f->get_feature_type() == F_WORD);
	return true;
}

/// clean up allocated memory
void CSortWord::cleanup()
{
}

/// initialize preprocessor from file
bool CSortWord::load(FILE* f)
{
	return false;
}

/// save preprocessor init-data to file
bool CSortWord::save(FILE* f)
{
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
WORD* CSortWord::apply_to_feature_matrix(CFeatures* f)
{
	INT i;
	INT num_vec;
	INT num_feat;
	WORD* matrix=((CWordFeatures*) f)->get_feature_matrix(num_feat, num_vec);

	for (i=0; i<num_vec; i++)
	{
		WORD* vec=&matrix[i*num_feat];
		math.qsort(vec, num_feat);
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
WORD* CSortWord::apply_to_feature_vector(WORD* f, INT& len)
{
	WORD* vec=new WORD[len];
	INT i=0;

	for (i=0; i<len; i++)
		vec[i]=f[i];

	math.qsort(vec, len);

	return vec;
}

/// initialize preprocessor from file
bool CSortWord::load_init_data(FILE* src)
{
	return true;
}

/// save init-data (like transforamtion matrices etc) to file
bool CSortWord::save_init_data(FILE* dst)
{
	return true;
}
