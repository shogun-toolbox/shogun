#include "preproc/SortUlongString.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/Mathmatics.h"

CSortUlongString::CSortUlongString() : CStringPreProc<ULONG>("SortUlongString", "STUS") 
{
}

CSortUlongString::~CSortUlongString()
{
}

/// initialize preprocessor from features
bool CSortUlongString::init(CFeatures* f)
{
	assert(f->get_feature_class() == C_STRING);
	assert(f->get_feature_type() == F_ULONG);
	return true;
}

/// clean up allocated memory
void CSortUlongString::cleanup()
{
}

/// initialize preprocessor from file
bool CSortUlongString::load(FILE* f)
{
	return false;
}

/// save preprocessor init-data to file
bool CSortUlongString::save(FILE* f)
{
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
bool CSortUlongString::apply_to_feature_strings(CFeatures* f)
{
	INT i;
	INT num_vec=((CStringFeatures<ULONG>*)f)->get_num_vectors() ;
	
	for (i=0; i<num_vec; i++)
	{
		INT len = 0 ;
		ULONG* vec = ((CStringFeatures<ULONG>*)f)->get_feature_vector(i, len) ;
		//CIO::message(M_DEBUG, "sorting string of length %i\n", len) ;
		
		CMath::qsort(vec, len);
	}
	return true ;
}

/// apply preproc on single feature vector
ULONG* CSortUlongString::apply_to_feature_string(ULONG* f, INT& len)
{
	ULONG* vec=new ULONG[len];
	INT i=0;

	for (i=0; i<len; i++)
		vec[i]=f[i];

	CMath::qsort(vec, len);

	return vec;
}

/// initialize preprocessor from file
bool CSortUlongString::load_init_data(FILE* src)
{
	return true;
}

/// save init-data (like transforamtion matrices etc) to file
bool CSortUlongString::save_init_data(FILE* dst)
{
	return true;
}
