#include <shogun/classifier/svm/LatentLinearMachine.h>

using namespace shogun;

CLatentLinearMachine::CLatentLinearMachine ()
{
}

CLatentLinearMachine::CLatentLinearMachine (argMaxLatent usrArgMaxFunc)
{
	setArgmax (usrArgMaxFunc);
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

void CLatentLinearMachine::setArgmax (argMaxLatent usrArgMaxFunc)
{
	ASSERT (usrArgMaxFunc != NULL);
	argMaxH = usrArgMaxFunc;
}

CFeatures* CLatentLinearMachine::defaultArgMaxH (CLatentLinearMachine& llm, void* userData)
{
	SGVector<float64_t> w = llm.get_w ();
	CDotFeatures* features = llm.get_features ();
	
	
	return 0;
}