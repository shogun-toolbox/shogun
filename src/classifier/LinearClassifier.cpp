#include "classifier/LinearClassifier.h"

CLinearClassifier::CLinearClassifier() : CClassifier(), w(NULL), bias(0), features(NULL)
{
}

CLinearClassifier::~CLinearClassifier()
{
}


bool CLinearClassifier::load(FILE* srcfile)
{
	return false;
}

bool CLinearClassifier::save(FILE* dstfile)
{
	return false;
}
