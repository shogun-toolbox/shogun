/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Evgeniy Andreev, Viktor Gal, 
 *          Sergey Lisitsyn, Bjoern Esser, Sanuj Sharma, Saurabh Goyal
 */

#include <shogun/preprocessor/RandomFourierGaussPreproc.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <vector>
#include <algorithm>

using namespace shogun;

void CRandomFourierGaussPreproc::copy(const CRandomFourierGaussPreproc & feats) {

	dim_input_space = feats.dim_input_space;
	cur_dim_input_space = feats.cur_dim_input_space;

	dim_feature_space = feats.dim_feature_space;
	cur_dim_feature_space=feats.cur_dim_feature_space;

	kernelwidth=feats.kernelwidth;
	cur_kernelwidth=feats.cur_kernelwidth;

	if(cur_dim_feature_space>0)
	{
		if(feats.randomcoeff_additive==NULL)
		{
			throw ShogunException(
							"void CRandomFourierGaussPreproc::copy(...): feats.randomcoeff_additive==NULL && cur_dim_feature_space>0 \n");
		}

		randomcoeff_additive = SG_MALLOC(float64_t, cur_dim_feature_space);
		std::copy(feats.randomcoeff_additive,feats.randomcoeff_additive+cur_dim_feature_space,randomcoeff_additive);
	}
	else
	{
		randomcoeff_additive = NULL;
	}

	if((cur_dim_feature_space>0)&&(cur_dim_input_space>0))
	{
		if(feats.randomcoeff_multiplicative==NULL)
		{
			throw ShogunException(
							"void CRandomFourierGaussPreproc::copy(...): feats.randomcoeff_multiplicative==NULL && cur_dim_feature_space>0 &&(cur_dim_input_space>0)  \n");
		}

		randomcoeff_multiplicative=SG_MALLOC(float64_t, cur_dim_feature_space*cur_dim_input_space);
		std::copy(feats.randomcoeff_multiplicative,feats.randomcoeff_multiplicative+cur_dim_feature_space*cur_dim_input_space,randomcoeff_multiplicative);
	}
	else
	{
		randomcoeff_multiplicative = NULL;
	}

}

CRandomFourierGaussPreproc::CRandomFourierGaussPreproc() :
	CDensePreprocessor<float64_t> () {
	dim_feature_space = 1000;
	dim_input_space = 0;
	cur_dim_input_space = 0;
	cur_dim_feature_space=0;

	randomcoeff_multiplicative=NULL;
	randomcoeff_additive=NULL;

	kernelwidth=1;
	cur_kernelwidth=kernelwidth;

	//m_parameter is inherited from CSGObject,
	//serialization initialization
	if(m_parameters)
	{
		SG_ADD(&dim_input_space, "dim_input_space",
		    "Dimensionality of the input space.", MS_NOT_AVAILABLE);
		SG_ADD(&cur_dim_input_space, "cur_dim_input_space",
		    "Dimensionality of the input space.", MS_NOT_AVAILABLE);
		SG_ADD(&dim_feature_space, "dim_feature_space",
		    "Dimensionality of the feature space.", MS_NOT_AVAILABLE);
		SG_ADD(&cur_dim_feature_space, "cur_dim_feature_space",
		    "Dimensionality of the feature space.", MS_NOT_AVAILABLE);

		SG_ADD(&kernelwidth, "kernelwidth", "Kernel width.", MS_AVAILABLE);
		SG_ADD(&cur_kernelwidth, "cur_kernelwidth", "Kernel width.", MS_AVAILABLE);

		m_parameters->add_vector(&randomcoeff_additive,&cur_dim_feature_space,"randomcoeff_additive");
		watch_param(
		    "randomcoeff_additive", &randomcoeff_additive,
		    &cur_dim_feature_space);

		m_parameters->add_matrix(&randomcoeff_multiplicative,&cur_dim_feature_space,&cur_dim_input_space,"randomcoeff_multiplicative");
		watch_param(
		    "randomcoeff_multiplicative", &randomcoeff_multiplicative,
		    &cur_dim_feature_space, &cur_dim_input_space);
	}

}

CRandomFourierGaussPreproc::CRandomFourierGaussPreproc(
		const CRandomFourierGaussPreproc & feats) :
	CDensePreprocessor<float64_t> () {

	randomcoeff_multiplicative=NULL;
	randomcoeff_additive=NULL;

	//m_parameter is inherited from CSGObject,
	//serialization initialization
	if(m_parameters)
	{
		SG_ADD(&dim_input_space, "dim_input_space",
		    "Dimensionality of the input space.", MS_NOT_AVAILABLE);
		SG_ADD(&cur_dim_input_space, "cur_dim_input_space",
		    "Dimensionality of the input space.", MS_NOT_AVAILABLE);
		SG_ADD(&dim_feature_space, "dim_feature_space",
		    "Dimensionality of the feature space.", MS_NOT_AVAILABLE);
		SG_ADD(&cur_dim_feature_space, "cur_dim_feature_space",
		    "Dimensionality of the feature space.", MS_NOT_AVAILABLE);

		SG_ADD(&kernelwidth, "kernelwidth", "Kernel width.", MS_AVAILABLE);
		SG_ADD(&cur_kernelwidth, "cur_kernelwidth", "Kernel width.", MS_AVAILABLE);

		m_parameters->add_vector(&randomcoeff_additive,&cur_dim_feature_space,"randomcoeff_additive");
		watch_param(
		    "randomcoeff_additive", &randomcoeff_additive,
		    &cur_dim_feature_space);

		m_parameters->add_matrix(&randomcoeff_multiplicative,&cur_dim_feature_space,&cur_dim_input_space,"randomcoeff_multiplicative");
		watch_param(
		    "randomcoeff_multiplicative", &randomcoeff_multiplicative,
		    &cur_dim_feature_space, &cur_dim_input_space);
	}

	copy(feats);
}

CRandomFourierGaussPreproc::~CRandomFourierGaussPreproc() {

	SG_FREE(randomcoeff_multiplicative);
	SG_FREE(randomcoeff_additive);

}

EFeatureClass CRandomFourierGaussPreproc::get_feature_class() {
	return C_DENSE;
}

EFeatureType CRandomFourierGaussPreproc::get_feature_type() {
	return F_DREAL;
}

int32_t CRandomFourierGaussPreproc::get_dim_feature_space() const {
	return ((int32_t) dim_feature_space);
}

void CRandomFourierGaussPreproc::set_dim_feature_space(const int32_t dim) {
	if (dim <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_dim_feature_space(const int32 dim): dim<=0 is not allowed");
	}

	dim_feature_space = dim;

}

int32_t CRandomFourierGaussPreproc::get_dim_input_space() const {
	return ((int32_t) dim_input_space);
}

void CRandomFourierGaussPreproc::set_kernelwidth(const float64_t kernelwidth2 ) {
	if (kernelwidth2 <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_kernelwidth(const float64_t kernelwidth2 ): kernelwidth2 <= 0 is not allowed");
	}
	kernelwidth=kernelwidth2;
}

float64_t CRandomFourierGaussPreproc::get_kernelwidth( ) const {
	return (kernelwidth);
}

void CRandomFourierGaussPreproc::set_dim_input_space(const int32_t dim) {
	if (dim <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_dim_input_space(const int32 dim): dim<=0 is not allowed");
	}

	dim_input_space = dim;

}

bool CRandomFourierGaussPreproc::test_rfinited() const {

	if ((dim_feature_space ==  cur_dim_feature_space)
			&& (dim_input_space > 0) && (dim_feature_space > 0)) {
		if ((dim_input_space == cur_dim_input_space)&&(CMath::abs(kernelwidth-cur_kernelwidth)<1e-5)) {

			// already inited
			return true;
		} else {
			return false;
		}
	}

	return false;
}

bool CRandomFourierGaussPreproc::init_randomcoefficients() {
	if (dim_feature_space <= 0) {
		throw ShogunException(
				"bool CRandomFourierGaussPreproc::init_randomcoefficients(): dim_feature_space<=0 is not allowed\n");
	}
	if (dim_input_space <= 0) {
		throw ShogunException(
				"bool CRandomFourierGaussPreproc::init_randomcoefficients(): dim_input_space<=0 is not allowed\n");
	}

	if (test_rfinited()) {
		return false;
	}


	SG_INFO("initializing randomcoefficients \n")

	float64_t pi = 3.14159265;


	SG_FREE(randomcoeff_multiplicative);
	randomcoeff_multiplicative=NULL;
	SG_FREE(randomcoeff_additive);
	randomcoeff_additive=NULL;


	cur_dim_feature_space=dim_feature_space;
	randomcoeff_additive=SG_MALLOC(float64_t, cur_dim_feature_space);
	cur_dim_input_space = dim_input_space;
	randomcoeff_multiplicative=SG_MALLOC(float64_t, cur_dim_feature_space*cur_dim_input_space);

	cur_kernelwidth=kernelwidth;

	for (int32_t  i = 0; i < cur_dim_feature_space; ++i) {
		randomcoeff_additive[i] = CMath::random((float64_t) 0.0, 2 * pi);
	}

	for (int32_t  i = 0; i < cur_dim_feature_space; ++i) {
		for (int32_t k = 0; k < cur_dim_input_space; ++k) {
			float64_t x1,x2;
			float64_t s = 2;
			while ((s >= 1) ) {
				// Marsaglia polar for gaussian
				x1 = CMath::random((float64_t) -1.0, (float64_t) 1.0);
				x2 = CMath::random((float64_t) -1.0, (float64_t) 1.0);
				s=x1*x1+x2*x2;
			}

			// =  x1/std::sqrt(val)* std::sqrt(-2*std::log(val));
			randomcoeff_multiplicative[i * cur_dim_input_space + k] =
			    x1 * std::sqrt(-2 * std::log(s) / s) / kernelwidth;
		}
	}

	SG_INFO("finished: initializing randomcoefficients \n")

	return true;
}

void CRandomFourierGaussPreproc::get_randomcoefficients(
		float64_t ** randomcoeff_additive2,
		float64_t ** randomcoeff_multiplicative2, int32_t *dim_feature_space2,
		int32_t *dim_input_space2, float64_t* kernelwidth2) const {

	ASSERT(randomcoeff_additive2)
	ASSERT(randomcoeff_multiplicative2)

	if (!test_rfinited()) {
		*dim_feature_space2 = 0;
		*dim_input_space2 = 0;
		*kernelwidth2=1;
		*randomcoeff_additive2 = NULL;
		*randomcoeff_multiplicative2 = NULL;
		return;
	}

	*dim_feature_space2 = cur_dim_feature_space;
	*dim_input_space2 = cur_dim_input_space;
	*kernelwidth2=cur_kernelwidth;

	*randomcoeff_additive2 = SG_MALLOC(float64_t, cur_dim_feature_space);
	*randomcoeff_multiplicative2 = SG_MALLOC(float64_t, cur_dim_feature_space*cur_dim_input_space);

	std::copy(randomcoeff_additive, randomcoeff_additive+cur_dim_feature_space,
			*randomcoeff_additive2);
	std::copy(randomcoeff_multiplicative, randomcoeff_multiplicative+cur_dim_feature_space*cur_dim_input_space,
			*randomcoeff_multiplicative2);


}

void CRandomFourierGaussPreproc::set_randomcoefficients(
		float64_t *randomcoeff_additive2,
		float64_t * randomcoeff_multiplicative2,
		const int32_t dim_feature_space2, const int32_t dim_input_space2, const float64_t kernelwidth2) {
	dim_feature_space = dim_feature_space2;
	dim_input_space = dim_input_space2;
	kernelwidth=kernelwidth2;

	SG_FREE(randomcoeff_multiplicative);
	randomcoeff_multiplicative=NULL;
	SG_FREE(randomcoeff_additive);
	randomcoeff_additive=NULL;

	cur_dim_feature_space=dim_feature_space;
	cur_dim_input_space = dim_input_space;
	cur_kernelwidth=kernelwidth;

	if( (dim_feature_space>0) && (dim_input_space>0) )
	{
	randomcoeff_additive=SG_MALLOC(float64_t, cur_dim_feature_space);
	randomcoeff_multiplicative=SG_MALLOC(float64_t, cur_dim_feature_space*cur_dim_input_space);

	std::copy(randomcoeff_additive2, randomcoeff_additive2
			+ dim_feature_space, randomcoeff_additive);
	std::copy(randomcoeff_multiplicative2, randomcoeff_multiplicative2
			+ cur_dim_feature_space*cur_dim_input_space, randomcoeff_multiplicative);
	}

}

void CRandomFourierGaussPreproc::fit(CFeatures* f)
{
	if (dim_feature_space <= 0) {
		throw ShogunException(
				"CRandomFourierGaussPreproc::init (CFeatures *f): dim_feature_space<=0 is not allowed, use void set_dim_feature_space(const int32 dim) before!\n");
	}

	SG_INFO("calling CRandomFourierGaussPreproc::init(...)\n")
	int32_t num_features =
	    f->as<CDenseFeatures<float64_t>>()->get_num_features();

	if (!test_rfinited()) {
		dim_input_space = num_features;
		init_randomcoefficients();
		ASSERT( test_rfinited())
	} else {
		dim_input_space = num_features;
		// does not reinit if dimension is the same to avoid overriding a previous call of set_randomcoefficients(...)
		init_randomcoefficients();
	}
}

SGVector<float64_t> CRandomFourierGaussPreproc::apply_to_feature_vector(SGVector<float64_t> vector)
{
	if (!test_rfinited()) {
		throw ShogunException(
				"float64_t * CRandomFourierGaussPreproc::apply_to_feature_vector(...): test_rfinited()==false: you need to call before CRandomFourierGaussPreproc::init (CFeatures *f) OR	1. set_dim_feature_space(const int32 dim), 2. set_dim_input_space(const int32 dim), 3. init_randomcoefficients() or set_randomcoefficients(...) \n");
	}

	float64_t val = std::sqrt(2.0 / cur_dim_feature_space);
	SGVector<float64_t> res(cur_dim_feature_space);

	for (int32_t od = 0; od < cur_dim_feature_space; ++od) {
		SGVector<float64_t> wrapper(randomcoeff_multiplicative+od*cur_dim_input_space, cur_dim_input_space, false);
		res[od] = val * cos(randomcoeff_additive[od] + linalg::dot(vector, wrapper));
	}

	return res;
}

SGMatrix<float64_t> CRandomFourierGaussPreproc::apply_to_feature_matrix(CFeatures* features)
{
	// version for case dim_feature_space < dim_input space with direct transformation on feature matrix ??

	int32_t num_vectors = 0;
	int32_t num_features = 0;
	float64_t* m = ((CDenseFeatures<float64_t>*) features)->get_feature_matrix(
			num_features, num_vectors);
	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features)

	if (num_features!=cur_dim_input_space)
	{
		throw ShogunException(
						"float64_t * CRandomFourierGaussPreproc::apply_to_feature_matrix(CFeatures *f): num_features!=cur_dim_input_space is not allowed\n");
	}

	if (m)
	{
		SGMatrix<float64_t> res(cur_dim_feature_space,num_vectors);

		float64_t val = std::sqrt(2.0 / cur_dim_feature_space);

		for (int32_t vec = 0; vec < num_vectors; vec++)
		{
			for (int32_t od = 0; od < cur_dim_feature_space; ++od)
			{
				SGVector<float64_t> a(m+vec * num_features, cur_dim_input_space, false);
				SGVector<float64_t> b(randomcoeff_multiplicative+od*cur_dim_input_space, cur_dim_input_space, false);
				res.matrix[od + vec * cur_dim_feature_space] = val * cos(
						randomcoeff_additive[od]
								+ linalg::dot(a, b));
			}
		}
		((CDenseFeatures<float64_t>*) features)->set_feature_matrix(res);

		return res;
	}
	else
		return SGMatrix<float64_t>();
}

void CRandomFourierGaussPreproc::cleanup()
{

}
