#include <classifier/mkl/MKLClassification.h>
#ifdef USE_SVMLIGHT
#include <classifier/svm/SVMLight.h>
#endif //USE_SVMLIGHT
#include <classifier/svm/LibSVM.h>

using namespace shogun;

CMKLClassification::CMKLClassification(CSVM* s) : CMKL(s)
{
	if (!s)
	{
#ifdef USE_SVMLIGHT
		s=new CSVMLight();
#endif //USE_SVMLIGHT
		if (!s)
			s=new CLibSVM();
		set_svm(s);
	}
}

CMKLClassification::~CMKLClassification()
{
}
float64_t CMKLClassification::compute_sum_alpha()
{
	float64_t suma=0;
	int32_t nsv=svm->get_num_support_vectors();
	for (int32_t i=0; i<nsv; i++)
		suma+=CMath::abs(svm->get_alpha(i));

	return suma;
}

void CMKLClassification::init_training()
{
	ASSERT(m_labels && m_labels->get_num_labels() && m_labels->get_label_type() == LT_BINARY)
}
