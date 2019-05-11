#include <shogun/base/SGObject.h>
#include <shogun/base/class_list.h>

namespace shogun
{
	CSGObject* CSGObject::create_empty() const
	{
		CSGObject* object = create(this->get_name(), this->get_generic());
		SG_REF(object);
		return object;
	};

	// ugly explicit template specialization
	template class HouseKeeper<CSGObject>;
} // namespace shogun
