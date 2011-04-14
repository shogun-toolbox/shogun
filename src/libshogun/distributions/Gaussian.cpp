#include "distributions/Gaussian.h"

using namespace shogun;

CGaussian::CGaussian() : CDistribution()
{
	float64_t* mean = new float64_t[1];
	float64_t* cov = new float64_t[1];
	mean[0] = 0;
	cov[0] = 1;
	int32_t dim = 1;
	init(mean, cov, dim);
}

CGaussian::CGaussian(float64_t* mean, float64_t* cov, int32_t dim) : CDistribution()
{
	init(mean, cov, dim);
}

void CGaussian::init(float64_t* mean, float64_t* cov, int32_t dim)
{
	m_mean = new float64_t[dim];
	memcpy(m_mean, mean, sizeof(float64_t)*dim);
	m_cov = new float64_t[dim*dim];
	memcpy(m_cov, cov, sizeof(float64_t)*dim*dim);
	m_cov_inverse = new float64_t[dim*dim];
	memcpy(m_cov_inverse, cov, sizeof(float64_t)*dim*dim);
	int32_t result = clapack_dpotrf(CblasRowMajor, CblasLower, dim, m_cov_inverse, dim);
	m_constant = 1;

	for (int i = 0; i < dim; i++)
		m_constant *= m_cov_inverse[i*dim+i];

	m_constant = 1/m_constant;
	m_constant *= pow(2*M_PI, (float64_t) -dim/2);

	result = clapack_dpotri(CblasRowMajor, CblasLower, dim, m_cov_inverse, dim);
	m_dim = dim;
}

CGaussian::~CGaussian()
{	
	delete[] m_cov_inverse;
	delete[] m_cov;
	delete[] m_mean;
}

bool CGaussian::train(CFeatures* data)
{
	return true;
}

int32_t CGaussian::get_num_model_parameters()
{
	return m_dim*(m_dim+1);
}

float64_t CGaussian::get_log_model_parameter(int32_t num_param)
{
	return 0;
}

float64_t CGaussian::get_log_derivative(int32_t num_param, int32_t num_example)
{
	return 0;
}

float64_t CGaussian::get_log_likelihood_example(int32_t num_example)
{
	return 0;
}
