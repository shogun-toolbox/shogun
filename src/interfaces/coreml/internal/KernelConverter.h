#ifndef __KERNEL_CONVERTER_H__
#define __KERNEL_CONVERTER_H__

#include "internal/CoreMLConverter.h"

#include <shogun/kernel/Kernel.h>

namespace CoreML
{
	namespace Specification
	{
		class Kernel;
	}
}

namespace shogun
{
	namespace coreml
	{
        struct KernelConverter
        {
            static CoreML::Specification::Kernel* convert(const CKernel* k);
        };
	}
}

#endif
