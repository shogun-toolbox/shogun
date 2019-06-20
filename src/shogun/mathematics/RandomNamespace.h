#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

#include <algorithm>

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
			std::shuffle(first, last, prng);
		}

		/** Reorders a container of elements randomly
		 *
		 * @param container the container holding the elements
		 */
		template <
		    template <typename> class Container, typename T, typename PRNG>
		static inline void shuffle(Container<T>& container, PRNG& prng)
		{
			shuffle(container.begin(), container.end(), prng);
		}

		/**
		 * Generate a positive signed random integer in the
		 * range [0, PRNG::max() - PRNG::min()] if
		 * signed(PRNG::max() - PRNG::min()) > 0, otherwise
		 * the range is [0, std::limits<I>::max()]
		 *
		 * @return the random positive signed integer
		 */
		template <
		    typename PRNG, typename I = typename std::make_signed<
		                       typename PRNG::result_type>::type>
		static inline I random_pos(PRNG& prng)
		{
			return (prng() - PRNG::min()) & ((PRNG::result_type(-1) << 1) >> 1);
		}

		/**
		 * Get random
		 * @return a float64_t random from [0,1] interval
		 */
		template <typename PRNG>
		static inline float64_t random_close(PRNG& prng)
		{
			return (prng() - PRNG::min()) /
			       float64_t(PRNG::max() - PRNG::min());
		}

		/**
		 * Get random
		 *
		 * @return a float64_t random from [0,1) interval
		 */
		template <typename PRNG>
		static inline float64_t random_half_open(PRNG& prng)
		{
			return (prng() - PRNG::min()) /
			       (PRNG::max() - PRNG::min() + float64_t(1.0));
		}

		/** generate an unsigned value in the range
		 * [min_value, max_value] (closed interval!)
		 *
		 * @param min_value minimum value
		 * @param max_value maximum value
		 * @return random number
		 */
		template <
		    typename PRNG, typename U,
		    typename std::enable_if_t<
		        std::is_same<U, typename PRNG::result_type>::value>* = nullptr>
		static inline U random(U min_value, U max_value, PRNG& prng)
		{
			return min_value +
			       (prng() - PRNG::min()) % (max_value - min_value + 1);
		}

		/** generate a signed value in the range
		 * [min_value, max_value] (closed interval!)
		 *
		 * @param min_value minimum value
		 * @param max_value maximum value
		 * @return random number
		 */
		template <
		    typename PRNG, typename S,
		    typename std::enable_if_t<std::is_same<
		        S, typename std::make_signed<
		               typename PRNG::result_type>::type>::value>* = nullptr>
		static inline S random(S min_value, S max_value, PRNG& prng)
		{
			return min_value + random_pos(prng) % (max_value - min_value + 1);
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
			return min_value + ((max_value - min_value) * random_close(prng));
		}
	} // namespace random
} // namespace shogun