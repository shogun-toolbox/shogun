#include <shogun/classifier/mkl/MKLOneClass.h>
#include <shogun/classifier/svm/LibSVMOneClass.h>

using namespace shogun;

MKLOneClass::MKLOneClass(std::shared_ptr<SVM> s) : MKL(s)
{
	if (!s)
		set_svm(std::make_shared<LibSVMOneClass>());
}

MKLOneClass::~MKLOneClass()
{
}

float64_t MKLOneClass::compute_sum_alpha()
{
	return 0.0;
}

void MKLOneClass::init_training()
{
	ASSERT(svm)
	ASSERT(svm->get_classifier_type() == CT_LIBSVMONECLASS)
}
