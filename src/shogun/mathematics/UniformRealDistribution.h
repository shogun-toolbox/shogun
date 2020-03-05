#ifndef __UNIFORMREALDISTRIBUTION_H__
#define __UNIFORMREALDISTRIBUTION_H__

#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

#include <limits>
#include <type_traits>

namespace shogun
{
	template <typename T = float64_t>
	class UniformRealDistribution
	{
		static_assert(
		    std::is_floating_point<T>::value,
		    "shogun::UniformRealDistribution is "
		    "specialized only for floating point types");

	public:
		using result_type = T;
		struct param_type
		{
			T min;
			T max;
			using distribution_type = UniformRealDistribution<T>;
		};

	public:
		UniformRealDistribution(
		    T min = 0, T max = std::numeric_limits<T>::max())
		    : m_min(min), m_max(max)
		{
			require(min < max, "The minimum value ({}) should be always less than the maximum value ({})", min, max);
		}

		template <typename PRNG>
		T operator()(PRNG& prng) const
		{
			return generate(prng, m_min, m_max);
		}

		template <typename PRNG>
		T operator()(PRNG& prng, param_type param) const
		{
			return generate(prng, param.min, param.max);
		}

		param_type param() const
		{
			return param_type{m_min, m_max};
		}

		void param(param_type param)
		{
			m_min = param.min;
			m_max = param.max;
		}

		T min() const
		{
			return m_min;
		}

		T max() const
		{
			return m_max;
		}

		void reset()
		{
		}

	private:
		template <typename PRNG>
		T generate(PRNG& prng, const T& min, const T& max) const
		{
			auto required_range = max - min;
			T result;
			do
			{
				auto fraction =
				    T(prng() - PRNG::min()) / (PRNG::max() - PRNG::min());
				result = fraction * required_range;
			} while (result >= required_range);

			return result + min;
		}

		T m_min;
		T m_max;
	};
} // namespace shogun

#endif // __UNIFORMREALDISTRIBUTION_H__
