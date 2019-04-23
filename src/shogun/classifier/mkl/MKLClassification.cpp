#include <shogun/classifier/mkl/MKLClassification.h>
#ifdef USE_SVMLIGHT
#include <shogun/classifier/svm/SVMLight.h>
#endif //USE_SVMLIGHT
#include <shogun/classifier/svm/LibSVM.h>

using namespace shogun;

MKLClassification::MKLClassification(std::shared_ptr<SVM> s) : MKL(s)
{
	if (!s)
	{
#ifdef USE_SVMLIGHT
		s=std::make_shared<SVMLight>();
#endif //USE_SVMLIGHT
		if (!s)
			s=std::make_shared<LibSVM>();
		set_svm(s);
	}
}

MKLClassification::~MKLClassification()
{
}
float64_t MKLClassification::compute_sum_alpha()
{
	float64_t suma=0;
	int32_t nsv=svm->get_num_support_vectors();
	for (int32_t i=0; i<nsv; i++)
		suma+=Math::abs(svm->get_alpha(i));

	return suma;
}

void MKLClassification::init_training()
{
	require(m_labels, "Labels not set.");
	require(m_labels->get_num_labels(), "Number of labels is zero.");
}

std::shared_ptr<MKLClassification> MKLClassification::obtain_from_generic(std::shared_ptr<Machine> machine)
{
	if (machine == NULL)
		return NULL;

	if (machine->get_classifier_type() != CT_MKLCLASSIFICATION)
		error("Provided machine is not of type CMKLClassification!");

	auto casted = std::dynamic_pointer_cast<MKLClassification>(machine);
	
	return casted;
}
