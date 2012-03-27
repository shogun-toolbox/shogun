#include <shogun/classifier/svm/LatentLinearMachine.h>

using namespace shogun;

CLatentLinearMachine::CLatentLinearMachine (minLatent usrFunc)
	: handleLatent (usrFunc)
{
	ASSERT (handleLatent != NULL);
}


CLatentLinearMachine::~CLatentLinearMachine ()
{
	
}

CLatentLabels* CLatentLinearMachine::apply ()
{
	
}

CLatentLabels* CLatentLinearMachine::apply (CFeatures* data)
{
	
}
