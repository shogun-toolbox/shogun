#include "classifier/KernelPerceptron.h"
#include "features/Labels.h"
#include "lib/Mathmatics.h"

CKernelPerceptron::CKernelPerceptron()
{
}


CKernelPerceptron::~CKernelPerceptron()
{
}

bool CKernelPerceptron::train()
{
	ASSERT(CKernelMachine::get_labels());
	//CLabels* train_labels=CKernelMachine::get_labels()->get_int_labels(num_train_labels);

//
//# compute output activation y = f(w x)
//# If y = t, don't change weights
//# If y != t, update the weights:
//
//w(new) = w(old) + 2 m t x
	return false;

}

REAL* CKernelPerceptron::test()
{
	return NULL;
}

bool CKernelPerceptron::load(FILE* srcfile)
{
	return false;
}

bool CKernelPerceptron::save(FILE* dstfile)
{
	return false;
}
