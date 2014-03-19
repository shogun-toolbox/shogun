#ifndef TYPE_DEFINE
#define TYPE_DEFINE

#include "shogun/lib/common.h"
#include "shogun/base/init.h"
#include "Eigen/Dense"

namespace shogun
{
	namespace deeplearning
	{
		namespace typedefine
		{
			typedef Eigen::Matrix<float32_t, Eigen::Dynamic, Eigen::Dynamic> EigenDenseMat;
			typedef Eigen::Matrix<float, Eigen::Dynamic, 1> EigenDenseVec;
			typedef Eigen::Matrix<float, 1, Eigen::Dynamic> EigenDenseRowVec;
		}
	}
}

#endif
