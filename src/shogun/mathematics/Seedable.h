#ifndef __SEEDABLE_H__
#define __SEEDABLE_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>

#include <functional>

namespace shogun
{
	namespace random
	{
		/** Seeds an SGObject using a specific seed
		 */
		template <
		    typename T, typename PRNG,
		    std::enable_if_t<std::is_base_of<
		        CSGObject, typename std::remove_pointer<T>::type>::value>* =
		        nullptr>
		static inline void seed(T* object, int32_t seed)
		{
			if (object->has("seed"))
				object->put("seed", seed);
		}
	} // namespace random

	static inline void seed_callback(CSGObject*, int32_t);

	template <typename Parent>
	class Seedable : public Parent
	{
	public:
		template <typename... T>
		Seedable(T... args) : Parent(args...)
		{
			init();
		}

	private:
		void init()
		{
			Parent::watch_param("seed", &m_seed);
			Parent::add_callback_function(
			    "seed", std::bind(seed_callback, this, std::ref(m_seed)));
		}

	protected:
		/** Seeds an SGObject using the current object seed
		 * This is intended to seed non-parameter SGObjects created inside
		 * methods
		 */
		template <
		    typename T,
		    std::enable_if_t<std::is_base_of<
		        CSGObject, typename std::remove_pointer<T>::type>::value>* =
		        nullptr>
		inline void seed(T* object) const
		{
			random::seed(object, m_seed);
		}

		int32_t m_seed;
	};

	static inline void seed_callback(CSGObject* obj, int32_t seed)
	{
		obj->for_each_param_of_type<CSGObject*>(
		    [&](const std::string& name, CSGObject** param) {
			    if ((*param)->has("seed"))
				    (*param)->put("seed", seed);
			    else
				    seed_callback(*param, seed);
		    });
	}
} // namespace shogun

#endif // __SEEDABLE_H__
