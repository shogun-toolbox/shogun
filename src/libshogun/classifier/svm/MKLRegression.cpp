#include "classifier/svm/MKLRegression.h"

CMKLRegression::CMKLRegression(CSVM* s) : CMKL(s)
{
}

CMKLRegression::~CMKLRegression()
{
}

float64_t CMKLRegression::compute_sum_alpha()
{
	SG_NOTIMPLEMENTED;
	return 0;

	// not correct needs explicit access to alpha and alpha* 
	//float64_t suma=0;
	//int32_t nsv=svm->get_num_support_vectors();
	//for (int32_t i=0; i<nsv; i++)
	//	suma+=CMath::abs(svm->get_alpha(i))*tube_epsilon-svm->get_alpha(i);
	//return suma;
}
float64_t CMKLRegression::compute_mkl_dual_objective()
{
	SG_NOTIMPLEMENTED;
	return 0;
}

void CMKLRegression::init_training()
{
	ASSERT(labels && labels->get_num_labels());
}
