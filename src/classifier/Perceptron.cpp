#include "classifier/Perceptron.h"
#include "features/Labels.h"
#include "lib/Mathmatics.h"

CPerceptron::CPerceptron()
{
}


CPerceptron::~CPerceptron()
{
}

bool CPerceptron::train()
{
	assert(CKernelMachine::get_labels());
	//INT num_train_labels;
	//CLabels* train_labels=CKernelMachine::get_labels()->get_int_labels(num_train_labels);

//# compute output activation y = f(w x)
//# If y = t, don't change weights
//# If y != t, update the weights:
//
//w(new) = w(old) + 2 m t x
	return false;

}

REAL* CPerceptron::test()
{
	return NULL;
}
