#include "classifier/svm/MKLOneClass.h"

CMKLOneClass::CMKLOneClass(CSVM* s) : CMKL(s)
{
}

CMKLOneClass::~CMKLOneClass()
{
}

float64_t CMKLOneClass::compute_sum_alpha()
{
	return 0.0;
}

void CMKLOneClass::init_training()
{
}
