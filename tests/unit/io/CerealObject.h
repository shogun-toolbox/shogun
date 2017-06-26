#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

	/** @brief Used to test the SGObject serialization */
	class CCerealObject : public CSGObject
	{
	public:
		// Construct CCerealObject from input SGVector
		CCerealObject(SGVector<float64_t> vec) : CSGObject()
		{
			m_vector = vec;
			init_params();
		}

		// Default constructor
		CCerealObject() : CSGObject()
		{
			m_vector = SGVector<float64_t>(5);
			m_vector.set_const(0);
			init_params();
		}

		const char* get_name() const
		{
			return "CerealObject";
		}

	protected:
		// Register m_vector to parameter list with name(tag) "test_vector"
		void init_params()
		{
			register_param("test_vector", m_vector);
		}

		SGVector<float64_t> m_vector;
	};
}
