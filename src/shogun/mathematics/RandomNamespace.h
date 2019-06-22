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

		/** Fills a container with random values in the range [min, max]
		 */
		template <typename Iterator, typename T, typename PRNG, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
		static inline void
		fill_array(Iterator first, Iterator last, T min, T max, PRNG& prng)
		{
			UniformIntDistribution<T> uniform_int_dist(min, max);
			for (auto it = first; it != last; ++it)
				*it = uniform_int_dist(prng);
		}

		/** Fills a container with random values in the range [min, max]
		 */
		template <typename Iterator, typename T, typename PRNG, typename std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
		static inline void
		fill_array(Iterator first, Iterator last, T min, T max, PRNG& prng)
		{
			UniformRealDistribution<T> uniform_real_dist(min, max);
			for (auto it = first; it != last; ++it)
				*it = uniform_real_dist(prng);
		}

		/** Fills a container with random values in the range [min, max]
		 */
		template <typename Container, typename T, typename PRNG>
		static inline void
		fill_array(Container& container, T min, T max, PRNG& prng)
		{
			fill_array(container.begin(), container.end(), min, max, prng);
		}
	} // namespace random
} // namespace shogun

#endif // __RANDOM_NAMESPACE_H__
