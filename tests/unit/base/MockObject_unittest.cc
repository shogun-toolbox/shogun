#include <shogun/lib/equals.h>
#include "MockObject.h"

namespace shogun
{
    template <>
    bool equals(CMockObject** lhs, CMockObject** rhs)
    {
        return (*lhs)->equals(*rhs);
    }

}
