/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __DATA_H__
#define __DATA_H__

#include <base/SGObject.h>

namespace shogun
{
	/** @brief dummy data holder
	 *
	 * An abstract class for storing data in any kind of format for
	 * LatentLabels, LatentFeatures and for StructuredLabels.
	 */
	class CData : public CSGObject
	{
		public:
			/** default ctor */
			CData();

			virtual ~CData();

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "Data"; }
	};
}

#endif /* __DATA_H__ */

