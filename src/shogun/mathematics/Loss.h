/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef __LOSS_H_
#define __LOSS_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
//#include <shogun/mathematics/lapack.h>
//#include <shogun/base/SGObject.h>
//#include <shogun/base/Parallel.h>


// Available losses
#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11

// Select loss
#define LOSS HINGELOSS


namespace shogun
{
/** @brief Class which collects generic mathematical functions
 */
class CLoss
{
	public:
		/**@name Constructor/Destructor.
		*/
		//@{
		///Constructor - initializes log-table
		CLoss();

		///Destructor - frees logtable
		virtual ~CLoss();
		//@}
		/** loss
		 * @param z
		 */
		static inline float64_t loss(float64_t z)
		{
#if LOSS == LOGLOSS
			if (z >= 0)
				return log(1+exp(-z));
			else
				return -z + log(1+exp(z));
#elif LOSS == LOGLOSSMARGIN
			if (z >= 1)
				return log(1+exp(1-z));
			else
				return 1-z + log(1+exp(z-1));
#elif LOSS == SMOOTHHINGELOSS
			if (z < 0)
				return 0.5 - z;
			if (z < 1)
				return 0.5 * (1-z) * (1-z);
			return 0;
#elif LOSS == SQUAREDHINGELOSS
			if (z < 1)
				return 0.5 * (1 - z) * (1 - z);
			return 0;
#elif LOSS == HINGELOSS
			if (z < 1)
				return 1 - z;
			return 0;
#else
# error "Undefined loss"
#endif
		}

		/** dloss
		 * @param z
		 */
		static inline float64_t dloss(float64_t z)
		{
#if LOSS == LOGLOSS
			if (z < 0)
				return 1 / (exp(z) + 1);
			float64_t ez = exp(-z);
			return ez / (ez + 1);
#elif LOSS == LOGLOSSMARGIN
			if (z < 1)
				return 1 / (exp(z-1) + 1);
			float64_t ez = exp(1-z);
			return ez / (ez + 1);
#elif LOSS == SMOOTHHINGELOSS
			if (z < 0)
				return 1;
			if (z < 1)
				return 1-z;
			return 0;
#elif LOSS == SQUAREDHINGELOSS
			if (z < 1)
				return (1 - z);
			return 0;
#else
			if (z < 1)
				return 1;
			return 0;
#endif
		}
};
}
#endif
