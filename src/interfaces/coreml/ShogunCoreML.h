#ifndef __SHOGUN_COREML_H__
#define __SHOGUN_COREML_H__

#include <shogun/machine/Machine.h>

#include "CoreMLModel.h"

namespace shogun
{
	namespace coreml
	{
		std::shared_ptr<CoreMLModel> convert(const CMachine* m) noexcept(false);
	}
}

#endif
