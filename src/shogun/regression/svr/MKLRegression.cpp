#include <regression/svr/MKLRegression.h>
#include <regression/svr/LibSVR.h>
#ifdef USE_SVMLIGHT
#include <regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT

using namespace shogun;

CMKLRegression::CMKLRegression(CSVM* s) : CMKL(s)
{
	if (!s)
	{
#ifdef USE_SVMLIGHT
		s=new CSVRLight();
#endif //USE_SVMLIGHT
		if (!s)
			s=new CLibSVR();
		set_svm(s);
	}
}

CMKLRegression::~CMKLRegression()
{
}

float64_t CMKLRegression::compute_sum_alpha()
{
	SG_NOTIMPLEMENTED
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
	SG_NOTIMPLEMENTED
	return 0;
}

void CMKLRegression::init_training()
{
	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(svm)
	ASSERT(svm->get_classifier_type() == CT_SVRLIGHT)
	ASSERT(interleaved_optimization)
}
