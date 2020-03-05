#ifndef __UNIFORMINTDISTRIBUTION_H__
#define __UNIFORMINTDISTRIBUTION_H__

#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

#include <limits>
#include <type_traits>

namespace shogun
{
	// FIXME: this could be implemented way more efficiently
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
		UniformIntDistribution(T min = 0, T max = std::numeric_limits<T>::max() - 1):
		      m_min(min),
		      m_max(max),
		      m_range(static_cast<std::make_unsigned_t<T>>(max) - static_cast<std::make_unsigned_t<T>>(min) + 1)
		{
			require(min < max, "The minimum value ({}) should be always less than the maximum value ({})", min, max);
		}

		template <typename PRNG>
		T operator()(PRNG& prng) const
		{
			static_assert(PRNG::max() >= std::numeric_limits<T>::max(),
					"Provide a PRNG::result_type is too small!");
			return generate(prng);
		}

		template <typename PRNG>
		T operator()(PRNG& prng, param_type param) const
		{
			static_assert(PRNG::max() >= std::numeric_limits<T>::max(),
					"Provide a PRNG::result_type is too small!");
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
			m_range = static_cast<std::make_unsigned_t<T>>(m_max) - static_cast<std::make_unsigned_t<T>>(m_min) + 1;
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
			const auto required_range = static_cast<decltype(m_range)>(max) - static_cast<decltype(m_range)>(min) + 1;
			const auto max_val = prng_range - (prng_range % required_range);

			typename PRNG::result_type result;
			do
			{
				result = prng() - PRNG::min();
			} while (result >= max_val);

			return (result % required_range) + min;
		}

		template <typename PRNG>
		T generate(PRNG& prng) const
		{
			constexpr auto prng_range = PRNG::max() - PRNG::min();
			const auto max_val = prng_range - (prng_range % m_range);

			typename PRNG::result_type result;
			do
			{
				result = prng() - PRNG::min();
			} while (result >= max_val);

			return (result % m_range) + m_min;
		}

		T m_min;
		T m_max;
		std::make_unsigned_t<T> m_range;

	};
} // namespace shogun

#endif // __UNIFORMINTDISTRIBUTION_H__
