#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include <shogun/util/mixins.h>
#include <shogun/base/mixins/HouseKeeper.h>
#include <shogun/base/mixins/ParameterHandler.h>
#include <shogun/base/mixins/ParameterWatcher.h>
#include <shogun/base/mixins/SGObjectBase.h>

namespace shogun
{
#ifndef IGNORE_IN_CLASSLIST
#define IGNORE_IN_CLASSLIST
#endif

	IGNORE_IN_CLASSLIST class CSGObject
	    : public composition<
	          CSGObject, CSGObjectBase, HouseKeeper, ParameterHandler,
	          ParameterWatcher>
	{
	public:
		virtual ~CSGObject(){};

		/** Returns an empty instance of own type.
		 *
		 * When inheriting from CSGObject from outside the main source tree
		 * (i.e. customized classes, or in a unit test), then this method has to
		 * be overloaded manually to return an empty instance. Shogun can only
		 * instantiate empty class instances from its source tree.
		 *
		 * @return empty instance of own type
		 */
		virtual CSGObject* create_empty() const override;

		// to resolve naming conflict
		SGIO*& io = mixin_t<shogun::HouseKeeper>::io;
	};

} // namespace shogun

#endif