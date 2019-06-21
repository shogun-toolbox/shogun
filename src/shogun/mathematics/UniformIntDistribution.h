#ifndef __UNIFORMINTDISTRIBUTION_H__
#define __UNIFORMINTDISTRIBUTION_H__

#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

#include <limits>
#include <type_traits>

namespace shogun
{
	template <typename T = int32_t>
	class UniformIntDistribution
	{
		static_assert(
		    std::is_integral<T>::value, "shogun::UniformIntDistribution is "
		                                "specialized only for integral types");

	public:
		using result_type = T;
		struct param_type
		{
			T min;
			T max;
			using distribution_type = UniformIntDistribution<T>;
		};

	public:
		UniformIntDistribution(T min = 0, T max = std::numeric_limits<T>::max())
		    : m_min(min), m_max(max)
		{
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
			constexpr auto prng_range = PRNG::max() - PRNG::min();
			const uint64_t required_range = max - min + 1;
			const auto max_val = prng_range - (prng_range % required_range);

			typename PRNG::result_type result;
			do
			{
				result = prng() - PRNG::min();
			} while (result >= max_val);

			return (result % required_range) + m_min;
		}

		T m_min;
		T m_max;
	};
} // namespace shogun

#endif // __UNIFORMINTDISTRIBUTION_H__