#include <shogun/base/some.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
	class MockClass : public CSGObject
	{
		virtual const char* get_name() const
        {
            return "MockClass";
        }
	};
}
