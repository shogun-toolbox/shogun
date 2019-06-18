#ifndef __RANDOMMIXIN_H__
#define __RANDOMMIXIN_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/config.h>

#include <random>

namespace shogun
{
	static inline void seed_callback(CSGObject*, int32_t);

	template <
	    typename Parent, typename PRNG = std::mt19937_64,
	    typename RandomDevice = std::random_device>
	class RandomMixin : public Parent
	{
	public:
		template <typename... T>
		RandomMixin(T... args) : Parent(args...)
		{
			init();
		}

		void set_random_seed()
		{
			RandomDevice random_device;
			set_seed(random_device());
		}

		void set_seed(const int32_t seed)
		{
			m_seed = seed;
			reinit_prng();
			seed_callback(this, seed);
		}

		bool reinit_prng()
		{
			m_prng = PRNG(m_seed);
			return true;
		}

		int32_t seed()
		{
			return m_seed;
		}

	private:
		void init()
		{
			set_random_seed();

			using this_t = RandomMixin<Parent, PRNG, RandomDevice>;

			Parent::watch_param("seed", &m_seed);
			Parent::watch_method("reinit_prng", &this_t::reinit_prng);
			Parent::template watch_method<bool>("seed_callback", [&]() {
				reinit_prng();
				seed_callback(this, m_seed);
				return true;
			});
		}

		int32_t m_seed;

	protected:
		PRNG m_prng;
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

#endif // __RANDOMMIXIN_H__
