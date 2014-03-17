#include "Maths.h"
#include <math.h>

using namespace shogun::deeplearning;
using namespace shogun::deeplearning::enums;

float32_t maths::Sigmoid(float32_t x)
{
	return 1 / (1 + exp(-x));
}

float32_t maths::TangentH(float32_t x)
{
	return tanh(x);
}

float32_t maths::Rectify(float32_t x)
{
	return x > 0 ? x : 0;
}

float32_t maths::Exponential(float32_t x)
{
	return exp(x);
}

float32_t maths::Logarithm(float32_t x)
{
	assert(x < 0);
	return log(x);
}

float32_t maths::Absolute(float32_t x)
{
	return x > 0 ? x : -x;
}

void maths::SoftMax(EigenDenseMat& m)
{
	if (m.cols() == 0) return;
	for (int32_t i = 0; i < m.rows(); ++i)
	{
		float32_t sum = 0, max_v = m(i, 0);
		for (int32_t j = 1; j < m.cols(); ++j)
			max_v = std::max(max_v, m(i, j));
		for (int32_t j = 0; j < m.cols(); ++j)
			sum += exp(m(i, j) - max_v);
		for (int32_t j = 0; j < m.cols(); ++j)
			m(i, j) = exp(m(i, j) - max_v) / sum;
	}
}

void maths::ApplyActivationFunc(EigenDenseMat& m, FuncType func)
{
	switch (func)
	{
	case FuncType::SIGM:
		m = m.unaryExpr(std::ptr_fun(maths::Sigmoid));
		break;
	case FuncType::TANH:
		m = m.unaryExpr(std::ptr_fun(maths::TangentH));
		break;
	case FuncType::RECTIFIED:
		m = m.unaryExpr(std::ptr_fun(maths::Rectify));
		break;
	case FuncType::SOFTMAX:
		SoftMax(m);
		break;
	default:
		break;
	}
}

void maths::ComputeDerivatives(EigenDenseMat& data, FuncType type)
{
	for (int32_t j = 0; j < data.cols(); ++j)
	for (int32_t i = 0; i < data.rows(); ++i)
	{
		switch (type)
		{
		case FuncType::RECTIFIED:
			data(i, j) = data(i, j) > 0 ? 1 : 0;
			break;
		case FuncType::SIGM:
			data(i, j) *= (1 - data(i, j));
			break;
		case FuncType::TANH:
			data(i, j) = 1 - data(i, j) * data(i, j);
			break;
		default:
			break;
		}
	}
}

float32_t maths::GetSquareLoss(const EigenDenseMat& err)
{
	float32_t loss = err.norm();
	loss *= loss;
	loss /= 2.0 * err.rows();
	return loss;
}

float32_t maths::GetLogLoss(const EigenDenseMat& output, const EigenDenseMat& true_labels)
{
	EigenDenseMat tmp = output.unaryExpr(std::ptr_fun(maths::Logarithm));

	return 0;
}
