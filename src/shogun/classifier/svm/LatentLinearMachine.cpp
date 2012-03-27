#include <shogun/classifier/svm/LatentLinearMachine.h>

using namespace shogun;

CLatentLinearMachine::CLatentLinearMachine ()
{
	
}

CLatentLinearMachine::CLatentLinearMachine (minimizeLatent usrFunc)
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

void CLatentLinearMachine::setLatentHandlerFunc (minimizeLatent usrFunc)
{
	ASSERT (handleLatent != NULL);
	handleLatent = usrFunc;
}

