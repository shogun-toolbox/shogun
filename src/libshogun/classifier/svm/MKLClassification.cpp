#include "classifier/svm/MKLClassification.h"

CMKLClassification::CMKLClassification(CSVM* s) : CMKL(s)
{
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
	ASSERT(labels && labels->get_num_labels() && labels->is_two_class_labeling());
}
