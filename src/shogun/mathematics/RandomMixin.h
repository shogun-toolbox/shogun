#ifndef __RANDOMMIXIN_H__
#define __RANDOMMIXIN_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/config.h>

#include <iterator>
#include <random>
#include <type_traits>

namespace shogun
{
	namespace random
	{
		/** Seeds an SGObject using a random number generator as a seed source
		 */
		template <
		    typename T, typename PRNG,
		    std::enable_if_t<std::is_base_of<
		        CSGObject, typename std::remove_pointer<T>::type>::value>* =
		        nullptr>
		static inline void seed(T* object, PRNG&& prng)
		{
			if (object->has("seed"))
				object->put("seed", (int32_t)prng());
		}
	} // namespace random

	static inline void seed_callback(CSGObject*, int32_t);
	static inline void random_seed_callback(CSGObject*);

	template <typename Parent, typename PRNG = std::mt19937_64>
	class RandomMixin : public Parent
	{
	private:
		using this_t = RandomMixin<Parent, PRNG>;

	public:
		using prng_type = PRNG;

		template <typename... T>
		RandomMixin(T... args) : Parent(args...)
		{
			init();
		}

		virtual CSGObject* clone() const override
		{
			auto clone = dynamic_cast<this_t*>(Parent::clone());
			clone->m_prng = m_prng;
			return clone;
		}

	private:
		void init_random_seed()
		{
			typename PRNG::result_type random_data[PRNG::state_size];
			std::random_device source;
			std::generate(
			    std::begin(random_data), std::end(random_data),
			    std::ref(source));
			std::seed_seq seeds(std::begin(random_data), std::end(random_data));
			m_prng = PRNG(seeds);
		}

		bool set_random_seed()
		{
			init_random_seed();
			random_seed_callback(this);
			return true;
		}

		void init()
		{
			init_random_seed();

			Parent::watch_param("seed", &m_seed);
			Parent::add_callback_function("seed", [&]() {
				m_prng = PRNG(m_seed);
				seed_callback(this, m_seed);
			});

			Parent::watch_method(
			    "set_random_seed", &this_t::set_random_seed);
		}

	protected:
		/** Seeds an SGObject using a random number generator as a seed source
		 * This is intended to seed non-parameter SGObjects created inside methods
		 */
		template <
		    typename T,
		    std::enable_if_t<std::is_base_of<
		        CSGObject, typename std::remove_pointer<T>::type>::value>* =
		        nullptr>
		inline void seed(T* object) const
		{
			random::seed(object, m_prng);
		}

		int32_t m_seed;
		mutable PRNG m_prng;
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

	static inline void random_seed_callback(CSGObject* obj)
	{
		obj->for_each_param_of_type<CSGObject*>(
		    [&](const std::string& name, CSGObject** param) {
			    if ((*param)->has("seed"))
				    (*param)->run("set_random_seed");
			    else
				    random_seed_callback(*param);
		    });
	}
} // namespace shogun

#endif // __RANDOMMIXIN_H__
