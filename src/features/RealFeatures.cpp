#include "features/RealFeatures.h"
#include "lib/File.h"

CFeatures* CRealFeatures::duplicate() const
{
	return new CRealFeatures(*this);
}

bool CRealFeatures::load(char* fname)
{
	bool status=false;
	num_vectors=1;
    num_features=0;
	CFile f(fname, 'r', F_REAL);
	feature_matrix=f.load_real_data(NULL, num_features);

    if (!f.is_ok())
		CIO::message("loading file \"%s\" failed", fname);
	else
		status=true;

	return status;
}

bool CRealFeatures::save(char* fname)
{
	long len;
	bool free;
	REAL* fv;

	CFile f(fname, 'w', F_REAL);

    for (int i=0; i< (long) num_vectors && f.is_ok(); i++)
	{
		if (!(i % (num_vectors/10+1)))
			CIO::message("%02d%%.", (int) (100.0*i/num_vectors));
		else if (!(i % (num_vectors/200+1)))
			CIO::message(".");

		fv=get_feature_vector(i, len, free);
		f.save_real_data(fv, len);
		free_feature_vector(fv, i, free) ;
	}

	if (f.is_ok())
		CIO::message("%d vectors with %d features each successfully written (filesize: %ld)", num_vectors, num_features, num_vectors*num_features*sizeof(REAL));

    return true;
}
