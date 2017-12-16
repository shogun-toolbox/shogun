#include <shogun/classifier/mkl/MKLClassification.h>
#ifdef USE_SVMLIGHT
#include <shogun/classifier/svm/SVMLight.h>
#endif //USE_SVMLIGHT
#include <shogun/classifier/svm/LibSVM.h>

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
	REQUIRE(m_labels, "Labels not set.\n");
	REQUIRE(m_labels->get_num_labels(), "Number of labels is zero.\n");
	REQUIRE(m_labels->get_label_type() == LT_BINARY, "Labels must be binary.\n");
}

CMKLClassification* CMKLClassification::obtain_from_generic(CMachine* machine)
{
	if (machine == NULL)
		return NULL;

	if (machine->get_classifier_type() != CT_MKLCLASSIFICATION)
		SG_SERROR("Provided machine is not of type CMKLClassification!")

	CMKLClassification* casted = dynamic_cast<CMKLClassification*>(machine);
	SG_REF(casted)
	return casted;
}