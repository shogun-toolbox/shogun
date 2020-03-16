/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_GRAPH_UTILS_H_
#define SHOGUN_GRAPH_UTILS_H_

namespace shogun
{
	namespace graph {

		template <typename T>
		size_t hash(const std::vector<T>& vec);

		template <typename T>
		inline auto hash_combine(std::size_t seed, const T& v) -> decltype(std::hash<T>{}(v))
		{
			std::hash<T> hasher;
		    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
		    return seed;
		}

		template <typename T>
		inline auto hash_combine(std::size_t seed, const T& v) -> decltype(hash(v))
		{
		    seed ^= hash(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
		    return seed;
		}

		// template <typename T>
		// inline auto hash_combine(std::size_t seed, const T& v) -> decltype(v->hash())
		// {
		//     seed ^= v->hash() + 0x9e3779b9 + (seed<<6) + (seed>>2);
		//     return seed;
		// }

		template <typename T>
		inline auto hash_combine(std::size_t seed, const T& v) -> decltype(v.hash())
		{
		    seed ^= v.hash() + 0x9e3779b9 + (seed<<6) + (seed>>2);
		    return seed;
		}

		template <typename T>
		size_t hash(const std::vector<T>& vec)
		{
			size_t seed = 0;
			for (const auto& el: vec)
			{
				seed = hash_combine(seed, el);
			}
			return seed;
		}
	}
}

#endif