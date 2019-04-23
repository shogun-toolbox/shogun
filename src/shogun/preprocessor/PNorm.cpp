/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Evgeniy Andreev, Bjoern Esser, 
 *          Sergey Lisitsyn
 */

#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/PNorm.h>

#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#endif

using namespace shogun;

PNorm::PNorm ()
: DensePreprocessor<float64_t>(),
 m_p (2.0)
{
	register_param ();
}

PNorm::PNorm (double p)
: DensePreprocessor<float64_t>(),
 m_p (p)
{
	ASSERT (m_p >= 1.0)
	register_param ();
}

PNorm::~PNorm ()
{
}

/// clean up allocated memory
void PNorm::cleanup ()
{
}

/// initialize preprocessor from file
bool PNorm::load (FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool PNorm::save (FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

SGMatrix<float64_t> PNorm::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	for (auto i : range(matrix.num_cols))
	{
		auto vec = matrix.get_column(i);
		auto norm = get_pnorm(vec.vector, vec.vlen);
		linalg::scale(vec, vec, 1.0 / norm);
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> PNorm::apply_to_feature_vector (SGVector<float64_t> vector)
{
	float64_t* normed_vec = SG_MALLOC(float64_t, vector.vlen);
	float64_t norm = get_pnorm (vector.vector, vector.vlen);

	for (int32_t i=0; i<vector.vlen; i++)
		normed_vec[i]=vector.vector[i]/norm;

	return SGVector<float64_t>(normed_vec,vector.vlen);
}

void PNorm::set_pnorm (double pnorm)
{
	ASSERT (pnorm >= 1.0)
	m_p = pnorm;
	register_param ();
}

double PNorm::get_pnorm () const
{
	return m_p;
}

void PNorm::register_param ()
{
	SG_ADD(&m_p, "norm", "P-norm parameter", ParameterProperties::HYPER);
}

inline float64_t PNorm::get_pnorm (float64_t* vec, int32_t vec_len) const
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
