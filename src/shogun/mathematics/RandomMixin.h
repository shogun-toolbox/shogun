#ifndef __RANDOMMIXIN_H__
#define __RANDOMMIXIN_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/config.h>

#include <random>

namespace shogun
{
	static inline void seed_callback(CSGObject*, int32_t);

	template <typename Parent, typename PRNG = std::mt19937_64>
	class RandomMixin : public Parent
	{
	public:
		template <typename... T>
		RandomMixin(T... args) : Parent(args...)
		{
			init();
		}

		int32_t seed()
		{
			return m_seed;
		}

	private:
		void init()
		{
			std::random_device rd;
			m_seed = rd();

			Parent::watch_param("seed", &m_seed);
			Parent::add_callback_function("seed", [&]() {
				m_prng = PRNG(m_seed);
				seed_callback(this, m_seed);
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
