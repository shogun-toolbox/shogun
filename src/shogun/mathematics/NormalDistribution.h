#ifndef __NORMALDISTRIBUTION_H__
#define __NORMALDISTRIBUTION_H__

#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/RandomNamespace.h>

#include <array>
#include <type_traits>

namespace shogun
{
	template <typename T>
	class NormalDistribution
	{
		static_assert(
		    std::is_same<T, float64_t>::value,
		    "shogun::NormalDistribution is specialized only for float64_t");
	};

	template <>
	class NormalDistribution<float64_t>
	{
	public:
		using result_type = float64_t;
		struct param_type
		{
			float64_t mean;
			float64_t stddev;
			using distribution_type = NormalDistribution<float64_t>;
		};

	public:
		NormalDistribution(param_type param);

		NormalDistribution(float64_t mean = 0, float64_t stddev = 1);

		template <typename PRNG>
		float64_t operator()(PRNG& prng) const
		{
			return m_mean + (std_normal_distrib(prng) * m_stddev);
		}

		template <typename PRNG>
		float64_t operator()(PRNG& prng, param_type param)
		{
			return param.mean + (std_normal_distrib(prng) * param.stddev);
		}

		param_type param() const
		{
			return param_type{m_mean, m_stddev};
		}

		void param(param_type param)
		{
			m_mean = param.mean;
			m_stddev = param.stddev;
		}

		void reset()
		{
		}

	protected:
		/**
		 * Sample a standard normal distribution,
		 * i.e. mean = 0, var = 1.0
		 *
		 * @return sample from the std normal distrib
		 */
		template <typename PRNG>
		float64_t std_normal_distrib(PRNG& prng) const
		{
			for (;;)
			{
				// Select box at random.
				uint8_t u = prng();
				int32_t i = (int32_t)(u & 0x7F);
				float64_t sign = ((u & 0x80) == 0) ? -1.0 : 1.0;

				// Generate uniform random value with range [0,0xffffffff].
				uint32_t u2 = prng();

				// Special case for the base segment.
				if (0 == i)
				{
					if (u2 < m_xComp[0])
					{
						// Generated x is within R0.
						return u2 * m_uint32ToU * m_A_div_y0 * sign;
					}
					// Generated x is in the tail of the distribution.
					return sample_tail(prng) * sign;
				}

				// All other segments.
				if (u2 < m_xComp[i])
				{ // Generated x is within the rectangle.
					return u2 * m_uint32ToU * m_x[i] * sign;
				}

				// Generated x is outside of the rectangle.
				// Generate a random y coordinate and test if our (x,y) is
				// within the distribution curve. This execution path is
				// relatively slow/expensive (makes a call to Math.Exp()) but
				// relatively rarely executed, although more often than the
				// 'tail' path (above).
				float64_t x = u2 * m_uint32ToU * m_x[i];
				if (m_y[i - 1] +
				        ((m_y[i] - m_y[i - 1]) * random::random_half_open(prng)) <
				    GaussianPdfDenorm(x))
				{
					return x * sign;
				}
			}
		}

		/**
		 * Sample from the distribution tail (defined as having x >= R).
		 *
		 * @return
		 */
		template <typename PRNG>
		float64_t sample_tail(PRNG& prng) const
		{
			float64_t x, y;
			float64_t m_R_reciprocal = 1.0 / m_R;
			do
			{
				x = -std::log(random::random_half_open(prng)) * m_R_reciprocal;
				y = -std::log(random::random_half_open(prng));
			} while (y + y < x * x);
			return m_R + x;
		}

		/**
		 * Gaussian probability density function, denormailised, that is, y =
		 * e^-(x^2/2).
		 */
		static float64_t GaussianPdfDenorm(float64_t x);

		/**
		 * Inverse function of GaussianPdfDenorm(x)
		 */
		static float64_t GaussianPdfDenormInv(float64_t y);

	private:
		/** Number of blocks */
		static constexpr int32_t m_blockCount = 128;

		/** Right hand x coord of the base rectangle, thus also the left hand x
		 * coord of the tail */
		static constexpr float64_t m_R = 3.442619855899;

		/** Area of each rectangle (pre-determined/computed for 128 blocks). */
		static constexpr float64_t m_A = 9.91256303526217e-3;

		/** Scale factor for converting a UInt with range [0,0xffffffff] to a
		 * double with range [0,1]. */
		static constexpr float64_t m_uint32ToU = 1.0 / (float64_t)UINT32_MAX;

		/** top-right position ox rectangle i */
		std::array<float64_t, m_blockCount + 1> m_x;
		std::array<float64_t, m_blockCount> m_y;

		/** Area A divided by the height of B0 */
		float64_t m_A_div_y0;

		/** The proprtion of each segment that is entirely within the
		distribution, expressed as uint where a value of 0 indicates 0% and
		uint.MaxValue 100%. Expressing this as an integer allows some floating
		points operations to be replaced with integer ones.
		*/
		std::array<uint32_t, m_blockCount> m_xComp;

		float64_t m_mean;
		float64_t m_stddev;
	};
} // namespace shogun

#endif // __NORMALDISTRIBUTION_H__