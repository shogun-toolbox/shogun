/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Sebastian Henschel
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _REGRESSION_H__
#define _REGRESSION_H__

namespace shogun
{
	/// type of regressor
	enum ERegressionType
	{
		RT_NONE = 0,
		RT_LIGHT = 10,
		RT_LIBSVM = 20
	};
}

#endif // _REGRESSION_H__

