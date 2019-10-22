/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Evgeniy Andreev, Viktor Gal,
 *          Sergey Lisitsyn, Bjoern Esser, Sanuj Sharma, Saurabh Goyal
 */

#include <shogun/preprocessor/RandomFourierGaussPreproc.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <vector>
#include <algorithm>

using namespace shogun;

void RandomFourierGaussPreproc::copy(const RandomFourierGaussPreproc & feats) {

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

RandomFourierGaussPreproc::RandomFourierGaussPreproc() :
	RandomMixin<DensePreprocessor<float64_t>> () {
	dim_feature_space = 1000;
	dim_input_space = 0;
	cur_dim_input_space = 0;
	cur_dim_feature_space=0;

	randomcoeff_multiplicative=NULL;
	randomcoeff_additive=NULL;

	kernelwidth=1;
	cur_kernelwidth=kernelwidth;

	SG_ADD(&dim_input_space, "dim_input_space",
			"Dimensionality of the input space.");
	SG_ADD(&cur_dim_input_space, "cur_dim_input_space",
			"Dimensionality of the input space.");
	SG_ADD(&dim_feature_space, "dim_feature_space",
			"Dimensionality of the feature space.");
	SG_ADD(&cur_dim_feature_space, "cur_dim_feature_space",
			"Dimensionality of the feature space.");

	SG_ADD(&kernelwidth, "kernelwidth", "Kernel width.", ParameterProperties::HYPER);
	SG_ADD(&cur_kernelwidth, "cur_kernelwidth", "Kernel width.", ParameterProperties::HYPER);

	watch_param(
		"randomcoeff_additive", &randomcoeff_additive,
		&cur_dim_feature_space);

	watch_param(
		"randomcoeff_multiplicative", &randomcoeff_multiplicative,
		&cur_dim_feature_space, &cur_dim_input_space);

}

RandomFourierGaussPreproc::RandomFourierGaussPreproc(
		const RandomFourierGaussPreproc & feats) :
	RandomMixin<DensePreprocessor<float64_t>> () {

	randomcoeff_multiplicative=NULL;
	randomcoeff_additive=NULL;

	SG_ADD(&dim_input_space, "dim_input_space",
			"Dimensionality of the input space.");
	SG_ADD(&cur_dim_input_space, "cur_dim_input_space",
			"Dimensionality of the input space.");
	SG_ADD(&dim_feature_space, "dim_feature_space",
			"Dimensionality of the feature space.");
	SG_ADD(&cur_dim_feature_space, "cur_dim_feature_space",
			"Dimensionality of the feature space.");

	SG_ADD(&kernelwidth, "kernelwidth", "Kernel width.", ParameterProperties::HYPER);
	SG_ADD(&cur_kernelwidth, "cur_kernelwidth", "Kernel width.", ParameterProperties::HYPER);

	watch_param(
		"randomcoeff_additive", &randomcoeff_additive,
		&cur_dim_feature_space);

	watch_param(
		"randomcoeff_multiplicative", &randomcoeff_multiplicative,
		&cur_dim_feature_space, &cur_dim_input_space);

	copy(feats);
}

RandomFourierGaussPreproc::~RandomFourierGaussPreproc() {

	SG_FREE(randomcoeff_multiplicative);
	SG_FREE(randomcoeff_additive);

}

EFeatureClass RandomFourierGaussPreproc::get_feature_class() {
	return C_DENSE;
}

EFeatureType RandomFourierGaussPreproc::get_feature_type() {
	return F_DREAL;
}

int32_t RandomFourierGaussPreproc::get_dim_feature_space() const {
	return ((int32_t) dim_feature_space);
}

void RandomFourierGaussPreproc::set_dim_feature_space(const int32_t dim) {
	if (dim <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_dim_feature_space(const int32 dim): dim<=0 is not allowed");
	}

	dim_feature_space = dim;

}

int32_t RandomFourierGaussPreproc::get_dim_input_space() const {
	return ((int32_t) dim_input_space);
}

void RandomFourierGaussPreproc::set_kernelwidth(const float64_t kernelwidth2 ) {
	if (kernelwidth2 <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_kernelwidth(const float64_t kernelwidth2 ): kernelwidth2 <= 0 is not allowed");
	}
	kernelwidth=kernelwidth2;
}

float64_t RandomFourierGaussPreproc::get_kernelwidth( ) const {
	return (kernelwidth);
}

void RandomFourierGaussPreproc::set_dim_input_space(const int32_t dim) {
	if (dim <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_dim_input_space(const int32 dim): dim<=0 is not allowed");
	}

	dim_input_space = dim;

}

bool RandomFourierGaussPreproc::test_rfinited() const {

	if ((dim_feature_space ==  cur_dim_feature_space)
			&& (dim_input_space > 0) && (dim_feature_space > 0)) {
		if ((dim_input_space == cur_dim_input_space)&&(Math::abs(kernelwidth-cur_kernelwidth)<1e-5)) {

			// already inited
			return true;
		} else {
			return false;
		}
	}

	return false;
}

bool RandomFourierGaussPreproc::init_randomcoefficients() {
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


	io::info("initializing randomcoefficients ");

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

	random::fill_array(
	    randomcoeff_additive, randomcoeff_additive + cur_dim_feature_space, 0.0,
	    2 * pi, m_prng);
	
	UniformRealDistribution<float64_t> uniform_real_dist(-1.0, 1.0);
	for (int32_t  i = 0; i < cur_dim_feature_space; ++i) {
		for (int32_t k = 0; k < cur_dim_input_space; ++k) {
			float64_t x1,x2;
			float64_t s = 2;
			while ((s >= 1) ) {
				// Marsaglia polar for gaussian
				x1 = uniform_real_dist(m_prng);
				x2 = uniform_real_dist(m_prng);
				s=x1*x1+x2*x2;
			}

			// =  x1/std::sqrt(val)* std::sqrt(-2*std::log(val));
			randomcoeff_multiplicative[i * cur_dim_input_space + k] =
			    x1 * std::sqrt(-2 * std::log(s) / s) / kernelwidth;
		}
	}

	io::info("finished: initializing randomcoefficients ");

	return true;
}

void RandomFourierGaussPreproc::get_randomcoefficients(
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

void RandomFourierGaussPreproc::set_randomcoefficients(
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

void RandomFourierGaussPreproc::fit(std::shared_ptr<Features> f)
{
	if (dim_feature_space <= 0) {
		throw ShogunException(
				"CRandomFourierGaussPreproc::init (Features *f): dim_feature_space<=0 is not allowed, use void set_dim_feature_space(const int32 dim) before!\n");
	}

	io::info("calling CRandomFourierGaussPreproc::init(...)");
	int32_t num_features =
	    f->as<DenseFeatures<float64_t>>()->get_num_features();

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

SGVector<float64_t> RandomFourierGaussPreproc::apply_to_feature_vector(SGVector<float64_t> vector)
{
	if (!test_rfinited()) {
		throw ShogunException(
				"float64_t * CRandomFourierGaussPreproc::apply_to_feature_vector(...): test_rfinited()==false: you need to call before CRandomFourierGaussPreproc::init (Features *f) OR	1. set_dim_feature_space(const int32 dim), 2. set_dim_input_space(const int32 dim), 3. init_randomcoefficients() or set_randomcoefficients(...) \n");
	}

	float64_t val = std::sqrt(2.0 / cur_dim_feature_space);
	SGVector<float64_t> res(cur_dim_feature_space);

	for (int32_t od = 0; od < cur_dim_feature_space; ++od) {
		SGVector<float64_t> wrapper(randomcoeff_multiplicative+od*cur_dim_input_space, cur_dim_input_space, false);
		res[od] = val * cos(randomcoeff_additive[od] + linalg::dot(vector, wrapper));
	}

	return res;
}

SGMatrix<float64_t>
RandomFourierGaussPreproc::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	// version for case dim_feature_space < dim_input space with direct transformation on feature matrix ??

	auto num_vectors = matrix.num_cols;
	auto num_features = matrix.num_rows;

	io::info("get Feature matrix: {}x{}", num_vectors, num_features);

	if (num_features != cur_dim_input_space)
	{
		throw ShogunException(
		    "float64_t * "
		    "CRandomFourierGaussPreproc::apply_to_matrix("
		    "SGMatrix<float64_t> matrix): matrix.num_rows != "
		    "cur_dim_input_space is not allowed\n");
	}

	SGMatrix<float64_t> res(cur_dim_feature_space, num_vectors);

	auto val = std::sqrt(2.0 / cur_dim_feature_space);

	for (auto vec : range(num_vectors))
	{
		for (auto od : range(cur_dim_feature_space))
		{
			SGVector<float64_t> a(
			    matrix.matrix + vec * num_features, cur_dim_input_space, false);
			SGVector<float64_t> b(
			    randomcoeff_multiplicative + od * cur_dim_input_space,
			    cur_dim_input_space, false);
			res(od, vec) =
			    val * cos(randomcoeff_additive[od] + linalg::dot(a, b));
		}
	}

	return res;
}

void RandomFourierGaussPreproc::cleanup()
{

}
