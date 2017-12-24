#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

	/** @brief Used to test the tags-parameter framework
	 * Allows testing of registering new member and avoiding
	 * non-registered member variables using tags framework.
	 */
	class CMockObject : public CSGObject
	{
	public:
		CMockObject() : CSGObject()
		{
			init_params();
		}

		const char* get_name() const
		{
			return "MockObject";
		}

		void set_watched(int32_t value)
		{
			m_watched = value;
		}

		int32_t get_watched() const
		{
			return m_watched;
		}

	protected:
		void init_params()
		{
			float64_t decimal = 0.0;
			register_param("vector", SGVector<float64_t>());
			register_param("int", m_integer);
			register_param("float", decimal);

			watch_param("watched_int", &m_watched);
		}

	private:
		int32_t m_integer = 0;
		int32_t m_watched = 0;
	};
}
