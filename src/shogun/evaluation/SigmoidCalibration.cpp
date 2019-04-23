/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Dhruv Arya
 */

#include <shogun/evaluation/SigmoidCalibration.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

SigmoidCalibration::SigmoidCalibration() : Calibration()
{
	init();
}

SigmoidCalibration::~SigmoidCalibration()
{
}

void SigmoidCalibration::init()
{
	m_maxiter = 100;
	m_minstep = 1E-10;
	m_sigma = 1E-12;
	m_epsilon = 1E-5;

	SG_ADD(
	    &m_sigmoid_as, "m_sigmoid_as",
	    "Vector of paramter A of sigmoid for each class.");
	SG_ADD(
	    &m_sigmoid_bs, "m_sigmoid_bs",
	    "Vector of paramter B of sigmoid for each class.");
	SG_ADD(
	    &m_maxiter, "m_maxiter", "Maximum number of iteration for search.");
	SG_ADD(
	    &m_minstep, "m_minstep", "Minimum step taken in line search.");
	SG_ADD(
	    &m_sigma, "m_sigma",
	    "Positive parameter to ensure positive semi-definite Hessian.");
	SG_ADD(&m_epsilon, "m_epsilon", "Stopping criteria.");
}

void SigmoidCalibration::set_maxiter(index_t maxiter)
{
	m_maxiter = maxiter;
}

index_t SigmoidCalibration::get_maxiter()
{
	return m_maxiter;
}

void SigmoidCalibration::set_minstep(float64_t minstep)
{
	m_minstep = minstep;
}

float64_t SigmoidCalibration::get_minstep()
{
	return m_minstep;
}

void SigmoidCalibration::set_sigma(float64_t sigma)
{
	m_sigma = sigma;
}

float64_t SigmoidCalibration::get_sigma()
{
	return m_sigma;
}

void SigmoidCalibration::set_epsilon(float64_t epsilon)
{
	m_epsilon = epsilon;
}

float64_t SigmoidCalibration::get_epsilon()
{
	return m_epsilon;
}

bool SigmoidCalibration::fit_binary(
    std::shared_ptr<BinaryLabels> predictions, std::shared_ptr<BinaryLabels> targets)
{
	m_sigmoid_as.resize_vector(1);
	m_sigmoid_bs.resize_vector(1);
	auto sigmoid_params = Statistics::fit_sigmoid(
	    predictions->get_values(), targets->get_labels(), m_maxiter, m_minstep,
	    m_sigma, m_epsilon);

	m_sigmoid_as[0] = sigmoid_params.a;
	m_sigmoid_bs[0] = sigmoid_params.b;

	return true;
}

std::shared_ptr<BinaryLabels> SigmoidCalibration::calibrate_binary(std::shared_ptr<BinaryLabels> predictions)
{
	require(
	    m_sigmoid_as.size() == 1,
	    "Parameters not fitted, which need to be fitted before calibrating.");
	Statistics::SigmoidParamters params;
	params.a = m_sigmoid_as[0];
	params.b = m_sigmoid_bs[0];

	auto probabilities = calibrate_values(predictions->get_values(), params);

	return std::make_shared<BinaryLabels>(probabilities);
}

bool SigmoidCalibration::fit_multiclass(
    std::shared_ptr<MulticlassLabels> predictions, std::shared_ptr<MulticlassLabels> targets)
{
	index_t num_classes = predictions->get_num_classes();

	m_sigmoid_as.resize_vector(num_classes);
	m_sigmoid_bs.resize_vector(num_classes);

	for (index_t i = 0; i < num_classes; ++i)
	{
		auto class_targets = targets->get_binary_for_class(i);
		auto pred_values = predictions->get_confidences_for_class(i);
		auto target_labels = class_targets->get_labels();

		auto sigmoid_params = Statistics::fit_sigmoid(
		    pred_values, target_labels, m_maxiter, m_minstep, m_sigma,
		    m_epsilon);
		m_sigmoid_as[i] = sigmoid_params.a;
		m_sigmoid_bs[i] = sigmoid_params.b;


	}

	return true;
}

std::shared_ptr<MulticlassLabels>
SigmoidCalibration::calibrate_multiclass(std::shared_ptr<MulticlassLabels> predictions)
{
	index_t num_classes = predictions->get_num_classes();
	index_t num_samples = predictions->get_num_labels();
	require(
	    m_sigmoid_as.size() == num_classes,
	    "Parameters not fitted, which need to be fitted before calibrating.");

	/** Temporarily stores the probabilities. */
	SGMatrix<float64_t> confidence_values(num_samples, num_classes);

	for (index_t i = 0; i < num_classes; ++i)
	{
		auto class_values = predictions->get_confidences_for_class(i);

		Statistics::SigmoidParamters sigmoid_params;
		sigmoid_params.a = m_sigmoid_as[i];
		sigmoid_params.b = m_sigmoid_bs[i];

		SGVector<float64_t> calibrated_values =
		    calibrate_values(class_values, sigmoid_params);

		confidence_values.set_column(i, calibrated_values);
	}

	auto result_labels = std::make_shared<MulticlassLabels>(num_samples);
	result_labels->allocate_confidences_for(num_classes);

	/** Normalize the probabilities. */
	for (index_t i = 0; i < num_samples; ++i)
	{
		SGVector<float64_t> values = confidence_values.get_row_vector(i);
		float64_t sum = SGVector<float64_t>::sum(values);

		/** All classes have equal probability when sum is zero */
		if (sum == 0)
			linalg::add_scalar(values, 1. / (float64_t)num_classes);
		else
			linalg::scale(values, values, 1 / sum);
		result_labels->set_multiclass_confidences(i, values);
	}
	return result_labels;
}

SGVector<float64_t> SigmoidCalibration::calibrate_values(
    SGVector<float64_t> values, Statistics::SigmoidParamters params)
{
	/** Calibrate values by passing them to a sigmoid function. */
	for (index_t i = 0; i < values.vlen; ++i)
	{
		float64_t fApB = values[i] * params.a + params.b;
		values[i] = fApB >= 0 ? std::exp(-fApB) / (1.0 + std::exp(-fApB))
		                      : 1.0 / (1 + std::exp(fApB));
	}
	return values;
}
