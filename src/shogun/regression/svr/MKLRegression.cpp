#include <shogun/regression/svr/MKLRegression.h>
#include <shogun/regression/svr/LibSVR.h>
#ifdef USE_SVMLIGHT
#include <shogun/regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT

using namespace shogun;

MKLRegression::MKLRegression(std::shared_ptr<SVM> s) : MKL(s)
{
	if (!s)
	{
#ifdef USE_SVMLIGHT
		s=std::make_shared<SVRLight>();
#endif //USE_SVMLIGHT
		if (!s)
			s=std::make_shared<LibSVR>();
		set_svm(s);
	}
}

MKLRegression::~MKLRegression()
{
}

float64_t MKLRegression::compute_sum_alpha()
{
	SG_NOTIMPLEMENTED
	return 0;

	// not correct needs explicit access to alpha and alpha*
	//float64_t suma=0;
	//int32_t nsv=svm->get_num_support_vectors();
	//for (int32_t i=0; i<nsv; i++)
	//	suma+=Math::abs(svm->get_alpha(i))*tube_epsilon-svm->get_alpha(i);
	//return suma;
}

float64_t MKLRegression::compute_mkl_dual_objective()
{
	SG_NOTIMPLEMENTED
	return 0;
}

void MKLRegression::init_training()
{
	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(svm)
	ASSERT(svm->get_classifier_type() == CT_SVRLIGHT)
	ASSERT(interleaved_optimization)
}
