#ifndef __RANDOM_NAMESPACE_H__
#define __RANDOM_NAMESPACE_H__

#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <algorithm>
#include <random>

namespace shogun
{
	namespace random
	{
		/** Reorders the elements in [first,  last) randomly
		 *
		 * @param first an iterator to the first element in the range
		 * @param last an iterator to the last element in the range
		 */
		template <typename RandomIt, typename PRNG>
		static inline void shuffle(RandomIt first, RandomIt last, PRNG& prng)
		{
			using diff_t =
			    typename std::iterator_traits<RandomIt>::difference_type;
			UniformIntDistribution<diff_t> dist;
			diff_t n = last - first;
			for (diff_t i = n - 1; i > 0; --i)
			{
				std::swap(first[i], first[dist(prng, {0, i})]);
			}
		}

		/** Reorders a container of elements randomly
		 *
		 * @param container the container holding the elements
		 */
		template <typename Container, typename PRNG>
		static inline void shuffle(Container& container, PRNG& prng)
		{
			shuffle(container.begin(), container.end(), prng);
		}

		/**
		 * Get random
		 * @return a float64_t random from [0,1] interval
		 */
		template <typename PRNG>
		static inline float64_t random_close(PRNG& prng)
		{
			UniformRealDistribution<float64_t> dist(
			    0, std::nextafter(1.0, std::numeric_limits<float64_t>::max()));
			return dist(prng);
		}

		/**
		 * Get random
		 *
		 * @return a float64_t random from [0,1) interval
		 */
		template <typename PRNG>
		static inline float64_t random_half_open(PRNG& prng)
		{
			UniformRealDistribution<float64_t> dist(0, 1);
			return dist(prng);
		}

		/** generate a signed value in the range
		 * [min_value, max_value] (closed interval!)
		 *
		 * @param min_value minimum value
		 * @param max_value maximum value
		 * @return random number
		 */
		template <
		    typename PRNG, typename I,
		    typename std::enable_if_t<std::is_integral<I>::value>* = nullptr>
		static inline I random(I min_value, I max_value, PRNG& prng)
		{
			UniformIntDistribution<I> dist(min_value, max_value);
			return dist(prng);
		}

		/** generate a floating point value in the range
		 * [min_value, max_value] (closed interval!)
		 *
		 * @param min_value minimum value
		 * @param max_value maximum value
		 * @return random number
		 */
		template <typename PRNG>
		static inline float64_t
		random(float64_t min_value, float64_t max_value, PRNG& prng)
		{
			UniformRealDistribution<float64_t> dist(min_value, max_value);
			return dist(prng);
		}

		/** Fills a container with random values in the range [min, max]
		 */
		template <typename Container, typename T, typename PRNG>
		static inline void
		fill_random(Container& container, T min, T max, PRNG& prng)
		{
			for (auto& val : container)
				val = random(min, max, prng);
		}

	} // namespace random
} // namespace shogun

#endif // __RANDOM_NAMESPACE_H__
