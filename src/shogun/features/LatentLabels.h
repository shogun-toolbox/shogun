/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTLABELS_H__
#define __LATENTLABELS_H__

#include <shogun/features/Labels.h>

namespace shogun
{
	class CLatentLabels : public CLabels
	{
		public:
			CLatentLabels ();
			
			virtual ~CLatentLabels ();
	};
}

#endif /* __LATENTLABELS_H__ */
