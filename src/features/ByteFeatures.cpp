#include "features/ByteFeatures.h"
#include "lib/common.h"
#include "lib/File.h"

CByteFeatures::CByteFeatures(long size) : CSimpleFeatures<BYTE>(size)
{
}

CByteFeatures::CByteFeatures(const CByteFeatures & orig) : CSimpleFeatures<BYTE>(orig)
{
}

CByteFeatures::CByteFeatures(char* fname) : CSimpleFeatures<BYTE>(fname)
{
	load(fname);
}

CFeatures* CByteFeatures::duplicate() const
{
	return new CByteFeatures(*this);
}


bool CByteFeatures::load(char* fname)
{
	bool status=false;
	num_vectors=1;
    num_features=0;
	CFile f(fname, 'r', F_BYTE);
	feature_matrix=f.load_byte_data(NULL, num_features);

    if (!f.is_ok())
		CIO::message("loading file \"%s\" failed", fname);
	else
		status=true;

	return status;
}

bool CByteFeatures::save(char* fname)
{
	return false;
}
