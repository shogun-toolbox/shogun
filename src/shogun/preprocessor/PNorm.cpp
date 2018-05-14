/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Evgeniy Andreev, Bjoern Esser, 
 *          Sergey Lisitsyn
 */

#include <shogun/preprocessor/PNorm.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/Features.h>

#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#endif

using namespace shogun;

CPNorm::CPNorm ()
: CDensePreprocessor<float64_t>(),
 m_p (2.0)
{
	register_param ();
}

CPNorm::CPNorm (double p)
: CDensePreprocessor<float64_t>(),
 m_p (p)
{
	ASSERT (m_p >= 1.0)
	register_param ();
}

CPNorm::~CPNorm ()
{
}

/// clean up allocated memory
void CPNorm::cleanup ()
{
}

/// initialize preprocessor from file
bool CPNorm::load (FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CPNorm::save (FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
SGMatrix<float64_t> CPNorm::apply_to_feature_matrix (CFeatures* features)
{
	auto feature_matrix =
	    features->as<CDenseFeatures<float64_t>>()->get_feature_matrix();

	for (int32_t i=0; i<feature_matrix.num_cols; i++)
	{
		float64_t* vec= &(feature_matrix.matrix[i*feature_matrix.num_rows]);
		float64_t norm = get_pnorm (vec, feature_matrix.num_rows);
		SGVector<float64_t>::scale_vector(1.0/norm, vec, feature_matrix.num_rows);
	}
	return feature_matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CPNorm::apply_to_feature_vector (SGVector<float64_t> vector)
{
	float64_t* normed_vec = SG_MALLOC(float64_t, vector.vlen);
	float64_t norm = get_pnorm (vector.vector, vector.vlen);

	for (int32_t i=0; i<vector.vlen; i++)
		normed_vec[i]=vector.vector[i]/norm;

	return SGVector<float64_t>(normed_vec,vector.vlen);
}

void CPNorm::set_pnorm (double pnorm)
{
	ASSERT (pnorm >= 1.0)
	m_p = pnorm;
	register_param ();
}

double CPNorm::get_pnorm () const
{
	return m_p;
}

void CPNorm::register_param ()
{
	SG_ADD(&m_p, "norm", "P-norm parameter", MS_AVAILABLE);
}

inline float64_t CPNorm::get_pnorm (float64_t* vec, int32_t vec_len) const
{
	float64_t norm = 0.0;
	if (m_p == 1.0)
	{
		for (int i = 0; i < vec_len; ++i)
			norm += fabs (vec[i]);
	}
	else if (m_p == 2.0)
	{
		norm = SGVector<float64_t>::twonorm(vec, vec_len);
	}
	else
	{
		norm = SGVector<float64_t>::qnorm(vec, vec_len, m_p);
	}

	return norm;
}
