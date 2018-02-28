/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann, 
 *          Evgeniy Andreev, Evan Shelhamer, Björn Esser
 */

#include <shogun/io/SGIO.h>

#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/HomogeneousKernelMap.h>

using namespace shogun;

CHomogeneousKernelMap::CHomogeneousKernelMap()
	: CDensePreprocessor<float64_t>(),
	m_kernel(HomogeneousKernelIntersection),
	m_window(HomogeneousKernelMapWindowRectangular),
	m_gamma(1.0),
	m_period(-1),
	m_order(1)
{
	init ();
	register_params ();
}

CHomogeneousKernelMap::CHomogeneousKernelMap(HomogeneousKernelType kernel,
	HomogeneousKernelMapWindowType wType, float64_t gamma,
	uint64_t order, float64_t period)
	: CDensePreprocessor<float64_t>(),
	m_kernel(kernel),
	m_window(wType),
	m_gamma(gamma),
	m_period(period),
	m_order(order)

{
	init ();
	register_params ();
}

CHomogeneousKernelMap::~CHomogeneousKernelMap()
{
}

bool CHomogeneousKernelMap::init(CFeatures* features)
{
	ASSERT(features->get_feature_class()==C_DENSE)
	ASSERT(features->get_feature_type()==F_DREAL)

	return true;
}

void CHomogeneousKernelMap::cleanup()
{
	m_table=SGVector<float64_t>();
}


void CHomogeneousKernelMap::init()
{
	SG_DEBUG ("Initialising homogeneous kernel map...\n")
	ASSERT (m_gamma > 0)

	ASSERT (m_kernel == HomogeneousKernelIntersection ||
			m_kernel == HomogeneousKernelChi2 ||
			m_kernel == HomogeneousKernelJS);

	ASSERT (m_window == HomogeneousKernelMapWindowUniform ||
			m_window == HomogeneousKernelMapWindowRectangular);

	if (m_period < 0) {
		switch (m_window) {
			case HomogeneousKernelMapWindowUniform:
				switch (m_kernel) {
					case HomogeneousKernelChi2:
						m_period = 5.86 * CMath::sqrt (static_cast<float64_t> (m_order))  + 3.65;
						break;
					case HomogeneousKernelJS:
						m_period = 6.64 * CMath::sqrt (static_cast<float64_t> (m_order))  + 7.24;
						break;
					case HomogeneousKernelIntersection:
				        m_period = 2.38 * std::log(m_order + 0.8) + 5.6;
				        break;
			        }
			        break;
		    case HomogeneousKernelMapWindowRectangular:
			    switch (m_kernel) {
					case HomogeneousKernelChi2:
						m_period = 8.80 * CMath::sqrt (m_order + 4.44) - 12.6;
						break;
					case HomogeneousKernelJS:
						m_period = 9.63 * CMath::sqrt (m_order + 1.00) - 2.93;
						break;
					case HomogeneousKernelIntersection:
				        m_period = 2.00 * std::log(m_order + 0.99) + 3.52;
				        break;
			        }
			        break;
		}
		m_period = CMath::max (m_period, 1.0) ;
	}

	m_numSubdivisions = 8 + 8*m_order;
	m_subdivision = 1.0 / m_numSubdivisions;
	m_minExponent = -20;
	m_maxExponent = 8;

	int tableHeight = 2*m_order + 1 ;
	int tableWidth = m_numSubdivisions * (m_maxExponent - m_minExponent + 1);
	size_t numElements = (tableHeight * tableWidth + 2*(1+m_order));
	if (unsigned(m_table.vlen) != numElements) {
		SG_DEBUG ("reallocating... %d -> %d\n", m_table.vlen, numElements)
		m_table.vector = SG_REALLOC (float64_t, m_table.vector, m_table.vlen, numElements);
		m_table.vlen = numElements;
	}

	int exponent;
	uint64_t i = 0, j = 0;
	float64_t* tablep = m_table.vector;
	float64_t* kappa = m_table.vector + tableHeight * tableWidth;
	float64_t* freq = kappa + (1+m_order);
	float64_t L = 2.0 * CMath::PI / m_period;

	/* precompute the sampled periodicized spectrum */
	while (i <= m_order) {
		freq[i] = j;
		kappa[i] = get_smooth_spectrum (j * L);
		++ j;
		if (kappa[i] > 0 || j >= 3*i) ++ i;
	}

	/* fill table */
	for (exponent  = m_minExponent ;
			exponent <= m_maxExponent ; ++ exponent) {

		float64_t x, Lxgamma, Llogx, xgamma;
		float64_t sqrt2kappaLxgamma;
		float64_t mantissa = 1.0;

		for (i = 0 ; i < m_numSubdivisions;
				++i, mantissa += m_subdivision) {
			x = std::ldexp (mantissa, exponent);
			xgamma = CMath::pow (x, m_gamma);
			Lxgamma = L * xgamma;
			Llogx = L * std::log(x);

			*tablep++ = CMath::sqrt (Lxgamma * kappa[0]);
			for (j = 1 ; j <= m_order; ++j) {
				sqrt2kappaLxgamma = CMath::sqrt (2.0 * Lxgamma * kappa[j]);
				*tablep++ = sqrt2kappaLxgamma * CMath::cos (freq[j] * Llogx);
				*tablep++ = sqrt2kappaLxgamma * CMath::sin (freq[j] * Llogx);
			}
		} /* next mantissa */
	} /* next exponent */

}

SGMatrix<float64_t> CHomogeneousKernelMap::apply_to_feature_matrix (CFeatures* features)
{
	CDenseFeatures<float64_t>* simple_features = (CDenseFeatures<float64_t>*)features;
	int32_t num_vectors = simple_features->get_num_vectors ();
	int32_t num_features = simple_features->get_num_features ();

	SGMatrix<float64_t> feature_matrix(num_features*(2*m_order+1),num_vectors);
	for (int i = 0; i < num_vectors; ++i)
	{
		SGVector<float64_t> transformed = apply_to_vector(simple_features->get_feature_vector(i));
		for (int j=0; j<transformed.vlen; j++)
			feature_matrix(j,i) = transformed[j];
	}

	simple_features->set_feature_matrix(feature_matrix);

	return feature_matrix;
}

/// apply preproc on single feature vector
SGVector<float64_t> CHomogeneousKernelMap::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SGVector<float64_t> result = apply_to_vector(vector);
	return result;
}

void CHomogeneousKernelMap::set_kernel_type(HomogeneousKernelType k)
{
	m_kernel = k;
	init ();
}

HomogeneousKernelType CHomogeneousKernelMap::get_kernel_type() const
{
	return m_kernel;
}

void CHomogeneousKernelMap::set_window_type(HomogeneousKernelMapWindowType w)
{
	m_window = w;
	init ();
}

HomogeneousKernelMapWindowType CHomogeneousKernelMap::get_window_type() const
{
	return m_window;
}

void CHomogeneousKernelMap::set_gamma(float64_t g)
{
	m_gamma = g;
	init ();
}

float64_t CHomogeneousKernelMap::get_gamma(float64_t g) const
{
	return m_gamma;
}

void CHomogeneousKernelMap::set_order(uint64_t o)
{
	m_order = o;
	init ();
}

uint64_t CHomogeneousKernelMap::get_order() const
{
	return m_order;
}

void CHomogeneousKernelMap::set_period(float64_t p)
{
	m_period = p;
	init ();
}

float64_t CHomogeneousKernelMap::get_period() const
{
	return m_period;
}

inline float64_t
CHomogeneousKernelMap::get_spectrum(float64_t omega) const
{
	switch (m_kernel) {
		case HomogeneousKernelIntersection:
			return (2.0 / CMath::PI) / (1 + 4 * omega*omega);
		case HomogeneousKernelChi2:
			return 2.0 / (CMath::exp (CMath::PI * omega) + CMath::exp (-CMath::PI * omega)) ;
		case HomogeneousKernelJS:
		    return (2.0 / std::log(4.0)) * 2.0 /
		           (CMath::exp(CMath::PI * omega) +
		            CMath::exp(-CMath::PI * omega)) /
		           (1 + 4 * omega * omega);
	    default:
		    /* throw exception */
		    throw ShogunException(
		        "CHomogeneousKernelMap::get_spectrum: no "
		        "valid kernel has been set!");
	}
}

inline float64_t
CHomogeneousKernelMap::sinc(float64_t x) const
{
	if (x == 0.0) return 1.0 ;
	return CMath::sin (x) / x;
}

inline float64_t
CHomogeneousKernelMap::get_smooth_spectrum(float64_t omega) const
{
	float64_t kappa_hat = 0;
	float64_t omegap ;
	float64_t epsilon = 1e-2;
	float64_t const omegaRange = 2.0 / (m_period * epsilon);
	float64_t const domega = 2 * omegaRange / (2 * 1024.0 + 1);
	switch (m_window) {
		case HomogeneousKernelMapWindowUniform:
			kappa_hat = get_spectrum (omega);
			break;
		case HomogeneousKernelMapWindowRectangular:
			for (omegap = - omegaRange ; omegap <= omegaRange ; omegap += domega) {
				float64_t win = sinc ((m_period/2.0) * omegap);
				win *= (m_period/(2.0*CMath::PI));
				kappa_hat += win * get_spectrum (omegap + omega);
			}
			kappa_hat *= domega;
			/* project on the postivie orthant (see PAMI) */
			kappa_hat = CMath::max (kappa_hat, 0.0);
			break;
		default:
			/* throw exception */
			throw ShogunException ("CHomogeneousKernelMap::get_smooth_spectrum: no valid kernel has been set!");
	}
	return kappa_hat;
}

SGVector<float64_t> CHomogeneousKernelMap::apply_to_vector(const SGVector<float64_t>& in_v) const
{
	/* assert for in vector */
	ASSERT (in_v.vlen > 0)
	ASSERT (in_v.vector != NULL)

	uint64_t featureDimension = 2*m_order+1;

	SGVector<float64_t> out_v(featureDimension*in_v.vlen);

	for (int k = 0; k < in_v.vlen; ++k) {
		/* break value into exponent and mantissa */
		int exponent;
		int unsigned j;
		float64_t mantissa = std::frexp (in_v[k], &exponent);
		float64_t sign = (mantissa >= 0.0) ? +1.0 : -1.0;
		mantissa *= 2*sign;
		exponent -- ;

		if (mantissa == 0 ||
				exponent <= m_minExponent ||
				exponent >= m_maxExponent)
		{
			for (j = 0 ; j <= m_order ; ++j) {
				out_v[k*featureDimension+j] = 0.0;
			}
			continue;
		}

		//uint64_t featureDimension = 2*m_order+1;
		float64_t const * v1 = m_table.vector +
			(exponent - m_minExponent) * m_numSubdivisions * featureDimension;
		float64_t const * v2;
		float64_t f1, f2;

		mantissa -= 1.0;
		while (mantissa >= m_subdivision) {
			mantissa -= m_subdivision;
			v1 += featureDimension;
		}

		v2 = v1 + featureDimension;
		for (j = 0 ; j < featureDimension ; ++j) {
			f1 = *v1++;
			f2 = *v2++;

			out_v[k*featureDimension+j] = sign * ((f2 - f1) * (m_numSubdivisions * mantissa) + f1);
		}
	}
	return out_v;
}

void CHomogeneousKernelMap::register_params()
{
	/* register variables */
	SG_ADD((machine_int_t*) &m_kernel, "kernel", "Kernel type to use.",MS_AVAILABLE);
	SG_ADD((machine_int_t*) &m_window, "window", "Window type to use.",MS_AVAILABLE);
	SG_ADD(&m_gamma, "gamma", "Homogeneity order.",MS_AVAILABLE);
	SG_ADD(&m_period, "period", "Approximation order",MS_NOT_AVAILABLE);
	SG_ADD(&m_numSubdivisions, "num_subdivisions", "The number of sublevels",MS_NOT_AVAILABLE);
	SG_ADD(&m_subdivision, "subdivision", "subdivision.",MS_NOT_AVAILABLE);
	SG_ADD(&m_order, "order", "The order",MS_AVAILABLE);
	SG_ADD(&m_minExponent, "min_exponent", "Minimum exponent",MS_NOT_AVAILABLE);
	SG_ADD(&m_maxExponent, "max_exponent", "Maximum exponent",MS_NOT_AVAILABLE);
	SG_ADD(&m_table, "table", "Lookup-table",MS_NOT_AVAILABLE);
}
