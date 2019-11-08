#include <shogun/base/SGObject.h>

namespace shogun
{
	class MockClass : public SGObject
	{
		virtual const char* get_name() const
        {
            return "MockClass";
        }
	};
}