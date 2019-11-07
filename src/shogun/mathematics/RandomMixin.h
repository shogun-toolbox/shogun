#ifndef __RANDOMMIXIN_H__
#define __RANDOMMIXIN_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Seedable.h>

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
		    std::enable_if_t<std::is_base_of_v<
		        SGObject, typename std::remove_pointer_t<T>>>* =
		        nullptr>
		static inline void seed(T* object, PRNG&& prng)
		{
			if (object->has(kSeed))
				object->put(kSeed, (int32_t)prng());
		}

		template <
		    typename T, typename PRNG,
		    std::enable_if_t<std::is_base_of_v<
		        SGObject, typename std::remove_pointer_t<T>>>* =
		        nullptr>
		static inline void seed(std::shared_ptr<T> object, PRNG&& prng)
		{
			seed(object.get(), prng);
		}
	} // namespace random

	static inline void random_seed_callback(SGObject*);

	template <typename Parent, typename PRNG = std::mt19937_64>
	class RandomMixin : public Seedable<Parent>
	{
	private:
		using this_t = RandomMixin<Parent, PRNG>;

	public:
		using prng_type = PRNG;

		template <typename... T>
		RandomMixin(T... args) : Seedable<Parent>(args...)
		{
			init();
		}

		virtual std::shared_ptr<SGObject> clone(ParameterProperties pp = ParameterProperties::ALL) const override
		{
			auto clone = std::dynamic_pointer_cast<this_t>(Parent::clone());
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

			Parent::add_callback_function(random::kSeed, [&]() {
				m_prng = PRNG(Seedable<Parent>::m_seed);
			});

			Parent::watch_method(
			    random::kSetRandomSeed, &this_t::set_random_seed);
		}

	protected:
		/** Seeds an SGObject using a random number generator as a seed source
		 * This is intended to seed non-parameter SGObjects created inside methods
		 */
		template <
		    typename T,
		    std::enable_if_t<std::is_base_of_v<
		        SGObject, typename std::remove_pointer<T>::type>>* =
		        nullptr>
		inline void seed(std::shared_ptr<T> object) const
		{
			random::seed(object, m_prng);
		}

		mutable PRNG m_prng;
	};

	static inline void random_seed_callback(SGObject* obj)
	{
		obj->for_each_param_of_type<SGObject*>(
		    [&](const std::string& name, SGObject** param) {
			    if ((*param)->has(random::kSeed))
				    (*param)->run(random::kSetRandomSeed);
			    else
				    random_seed_callback(*param);
		    });
	}
} // namespace shogun

#endif // __RANDOMMIXIN_H__
