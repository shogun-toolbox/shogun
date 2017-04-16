#ifndef MATHS
#define MATHS

#include "TypeDefine.h"
#include "Enums.h"

using namespace shogun::deeplearning::typedefine;
using namespace shogun::deeplearning::enums;

namespace shogun
{
	namespace deeplearning
	{
		namespace maths
		{
			const float32_t eps = (float32_t)1e-8;

			//Activation functions
			float32_t Sigmoid(float32_t e);
			float32_t TangentH(float32_t e);
			float32_t Rectify(float32_t e);
			float32_t Exponential(float32_t e);
			float32_t Logarithm(float32_t e);
			float32_t Absolute(float32_t e);

			//Computational functions in Neural Networks
			void SoftMax(EigenDenseMat& m);
			void ApplyActivationFunc(EigenDenseMat& m, FuncType func);
			void ComputeDerivatives(EigenDenseMat& data, FuncType type);

			float32_t GetSquareLoss(const EigenDenseMat& err);
			float32_t GetLogLoss(const EigenDenseMat& output, const EigenDenseMat& true_labels);
		}
	}
}

#endif
