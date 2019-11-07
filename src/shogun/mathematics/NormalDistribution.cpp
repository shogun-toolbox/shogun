#include <shogun/io/SGIO.h>
#include <shogun/mathematics/NormalDistribution.h>

#include <cmath>
#include <limits>

using namespace shogun;

NormalDistribution<float64_t>::NormalDistribution(param_type param)
    : NormalDistribution(param.mean, param.stddev)
{
}

NormalDistribution<float64_t>::NormalDistribution(
    float64_t mean, float64_t stddev)
    : m_uniform_real_dist(0.0, 1.0),
    m_mean(mean), m_stddev(stddev)
{
	// Initialise rectangle position data.
	// m_x[i] and m_y[i] describe the top-right position ox Box i.

	// Determine top right position of the base rectangle/box (the rectangle
	// with the Gaussian tale attached). We call this Box 0 or B0 for short.
	// Note. x[0] also describes the right-hand edge of B1. (See diagram).
	m_x[0] = m_R;
	m_y[0] = GaussianPdfDenorm(m_R);

	// The next box (B1) has a right hand X edge the same as B0.
	// Note. B1's height is the box area divided by its width, hence B1 has a
	// smaller height than B0 because B0's total area includes the attached
	// distribution tail.
	m_x[1] = m_R;
	m_y[1] = m_y[0] + (m_A / m_x[1]);

	// Calc positions of all remaining rectangles.
	for (int i = 2; i < m_blockCount; i++)
	{
		m_x[i] = GaussianPdfDenormInv(m_y[i - 1]);
		m_y[i] = m_y[i - 1] + (m_A / m_x[i]);
	}

	// For completeness we define the right-hand edge of a notional box 6 as
	// being zero (a box with no area).
	m_x[m_blockCount] = 0.0;

	// Useful precomputed values.
	m_A_div_y0 = m_A / m_y[0];

	// Special case for base box. m_xComp[0] stores the area of B0 as a
	// proportion of R (recalling that all segments have area A, but that the
	// base segment is the combination of B0 and the distribution tail). Thus
	// -m_xComp[0] is the probability that a sample point is within the box part
	// of the segment.
	m_xComp[0] = (uint32_t)(
	    ((m_R * m_y[0]) / m_A) *
	    (float64_t)std::numeric_limits<uint32_t>::max());

	for (int32_t i = 1; i < m_blockCount - 1; i++)
	{
		m_xComp[i] = (uint32_t)(
		    (m_x[i + 1] / m_x[i]) *
		    (float64_t)std::numeric_limits<uint32_t>::max());
	}
	m_xComp[m_blockCount - 1] = 0; // Shown for completeness.

	// Sanity check. Test that the top edge of the topmost rectangle is at
	// y=1.0. Note. We expect there to be a tiny drift away from 1.0 due to the
	// inexactness of floating point arithmetic.
	ASSERT(std::abs(1.0 - m_y[m_blockCount - 1]) < 1e-10);
}

float64_t NormalDistribution<float64_t>::GaussianPdfDenorm(float64_t x)
{
	return std::exp(-(x * x * 0.5));
}

float64_t NormalDistribution<float64_t>::GaussianPdfDenormInv(float64_t y)
{
	// Operates over the y range (0,1], which happens to be the y range of the
	// pdf, with the exception that it does not include y=0, but we would never
	// call with y=0 so it doesn't matter. Remember that a Gaussian effectively
	// has a tail going off into x == infinity, hence asking what is x when y=0
	// is an invalid question in the context of this class.
	return std::sqrt(-2.0 * std::log(y));
}
