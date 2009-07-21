#include "classifier/svm/MKLOneClass.h"
#include "classifier/svm/LibSVMOneClass.h"

CMKLOneClass::CMKLOneClass(CSVM* s) : CMKL(s)
{
	if (!s)
		set_svm(new CLibSVMOneClass());
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
	ASSERT(svm);
	ASSERT(svm->get_classifier_type() == CT_LIBSVMONECLASS);
}
