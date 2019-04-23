/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Viktor Gal
 */

#ifndef __DATA_H__
#define __DATA_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{
	/** @brief dummy data holder
	 *
	 * An abstract class for storing data in any kind of format for
	 * LatentLabels, LatentFeatures and for StructuredLabels.
	 */
	class Data : public SGObject
	{
		public:
			/** default ctor */
			Data();

			virtual ~Data();

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "Data"; }
	};
}

#endif /* __DATA_H__ */

