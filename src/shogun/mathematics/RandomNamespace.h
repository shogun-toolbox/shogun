#ifndef __RANDOM_NAMESPACE_H__
#define __RANDOM_NAMESPACE_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <algorithm>
#include <iterator>
#include <random>
#include <utility>

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
		static inline void shuffle(RandomIt first, RandomIt last, PRNG&& prng)
		{
			using diff_t =
			    typename std::iterator_traits<RandomIt>::difference_type;
			UniformIntDistribution<diff_t> dist;
			diff_t n = last - first;
			for (diff_t i = n - 1; i > 0; --i)
				std::swap(first[i], first[dist(prng, {0, i})]);
		}

		/** Reorders a container of elements randomly
		 *
		 * @param container the container holding the elements
		 * @param prng pseudo number generator object
		 */
		template <typename Container, typename PRNG>
		static inline void shuffle(Container& container, PRNG&& prng)
		{
			random::shuffle(
			    std::begin(container), std::end(container),
			    std::forward<PRNG>(prng));
		}

		/** Fills an array with random numbers generated from a given
		 * distribution
		 *
		 * @param first an iterator to the first element in the range
		 * @param last an iterator to the last element in the range
		 * @param dist random number distribution
		 * @param prng pseudo number generator object
		 */
		template <typename Iterator, typename Distribution, typename PRNG>
		static inline void fill_array(
		    Iterator first, Iterator last, Distribution&& dist, PRNG&& prng)
		{
			while (first != last)
				*first++ = dist(prng);
		}

		/** Fills an array with random numbers generated from a given
		 * distribution
		 *
		 * @param container the container to be filled
		 * @param dist random number distribution
		 * @param prng pseudo number generator object
		 */
		template <typename Container, typename Distribution, typename PRNG>
		static inline void
		fill_array(Container& container, Distribution&& dist, PRNG&& prng)
		{
			fill_array(
			    std::begin(container), std::end(container),
			    std::forward<Distribution>(dist), std::forward<PRNG>(prng));
		}

		/** Fills a container with random values in the range [min, max]
		 */
		template <typename Iterator, typename T, typename PRNG>
		static inline void
		fill_array(Iterator first, Iterator last, T min, T max, PRNG&& prng)
		{
			static_assert(
			    std::is_arithmetic<T>::value,
			    "random::fill_array: range [min, max] must be numerical");

			if constexpr (std::is_integral<T>::value)
				fill_array(
				    first, last, UniformIntDistribution<T>(min, max),
				    std::forward<PRNG>(prng));
			else
				fill_array(
				    first, last, UniformRealDistribution<T>(min, max),
				    std::forward<PRNG>(prng));
		}

		/** Fills a container with random values in the range [min, max]
		 */
		template <typename Container, typename T, typename PRNG>
		static inline void
		fill_array(Container& container, T min, T max, PRNG&& prng)
		{
			fill_array(container.begin(), container.end(), min, max, prng);
		}

		/** Returns a vector of randomly generated numbers using the 
		 * the provided pseudo random number generator.
		 * @param prng pseudo number generator object
		 * @param num_elements number of elements to generate
		 */
		template <typename PRNG, typename T>
		std::vector<typename std::remove_reference_t<PRNG>::result_type> 
		generate_array(PRNG&& prng, const T& num_elements) 
		{
			std::vector<typename std::remove_reference_t<PRNG>::result_type> result;
			result.reserve(num_elements);
			std::generate_n(std::back_inserter(result), num_elements, 
				[prng = std::forward<PRNG>(prng)]() mutable { 
					return prng(); 
				});

			return result;
		}
	} // namespace random
} // namespace shogun

#endif // __RANDOM_NAMESPACE_H__
