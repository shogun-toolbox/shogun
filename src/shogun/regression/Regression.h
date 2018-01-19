/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _REGRESSION_H__
#define _REGRESSION_H__

#include <shogun/lib/config.h>

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

