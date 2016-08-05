#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

/** @brief Used to test the SGObject serialization */
class CCerealObject : public CSGObject
{
public:
	CCerealObject() : CSGObject()
	{
		init_params();
	}

	SGVector<float64_t> data()
	{
		return m_vector;
	}

	const char* get_name() const { return "CerealObject"; }

protected:
	void init_params()
	{
		m_vector = SGVector<float64_t>(5);
		m_vector.set_const(0);
		register_param("test_vector", m_vector);
	}

	SGVector<float64_t> m_vector;
};
}
