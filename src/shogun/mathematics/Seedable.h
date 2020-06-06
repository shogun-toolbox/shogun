#ifndef __SEEDABLE_H__
#define __SEEDABLE_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>

#include <functional>

namespace shogun
{
	namespace random
	{
#ifndef SWIG
		static constexpr std::string_view kSetRandomSeed = "set_random_seed";
		static constexpr std::string_view kSeed = "seed";
#endif // SWIG		
		/** Seeds an SGObject using a specific seed
		 */
		template <
		    typename T, typename PRNG,
		    std::enable_if_t<std::is_base_of<
		        SGObject, typename std::remove_pointer<T>::type>::value>* =
		        nullptr>
		static inline void seed(T* object, int32_t seed)
		{
			if (object->has(kSeed))
				object->put(kSeed, seed);
		}
	} // namespace random

	static inline void seed_callback(SGObject*, int32_t);

	template <typename Parent>
	class Seedable : public Parent
	{
	public:
		template <typename... T>
		Seedable(T... args) : Parent(args...)
		{
			init();
		}

		virtual const char* get_name() const override
		{
			return "Seedable";
		}

	private:
		void init()
		{
			Parent::watch_param(random::kSeed, &m_seed);
			Parent::add_callback_function(
			    random::kSeed, std::bind(seed_callback, this, std::ref(m_seed)));
		}

	protected:
		/** Seeds an SGObject using the current object seed
		 * This is intended to seed non-parameter SGObjects created inside
		 * methods
		 */
		template <
		    typename T,
		    std::enable_if_t<std::is_base_of<
		        SGObject, typename std::remove_pointer<T>::type>::value>* =
		        nullptr>
		inline void seed(T* object) const
		{
			random::seed(object, m_seed);
		}

		int32_t m_seed;
	};

	static inline void seed_callback(SGObject* obj, int32_t seed)
	{
		obj->for_each_param_of_type<SGObject*>(
		    [&](const std::string& name, SGObject** param) {
			    if ((*param)->has(random::kSeed))
				    (*param)->put(random::kSeed, seed);
			    else
				    seed_callback(*param, seed);
		    });
	}
} // namespace shogun

#endif // __SEEDABLE_H__
