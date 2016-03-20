#include <shogun/base/class_list.h>
#include <shogun/base/some.h>

namespace shogun
{
    Some<CSGObject> object(const char* name) {
        CSGObject* obj = new_sgserializable(name, PT_NOT_GENERIC);
        return adopt_pointer<CSGObject>(obj);
    }
}
