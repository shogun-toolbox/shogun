#include "features/SparseRealFeatures.h"
#include "lib/File.h"

CFeatures* CSparseRealFeatures::duplicate() const
{
	return new CSparseRealFeatures(*this);
}

bool CSparseRealFeatures::load(char* fname)
{
	return false;
}

bool CSparseRealFeatures::save(char* fname)
{
	return false;
}

