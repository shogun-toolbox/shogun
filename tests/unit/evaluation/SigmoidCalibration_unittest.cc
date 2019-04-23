/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Dhruv Arya
 */
#include <gtest/gtest.h>
#include <shogun/evaluation/SigmoidCalibration.h>

using namespace shogun;

TEST(SigmoidCalibrationTest, binary_calibration)
{
	SGVector<float64_t> preds(10), labs(10);

	preds.vector[0] = 0.6;
	preds.vector[1] = -0.2;
	preds.vector[2] = 0.7;
	preds.vector[3] = 0.9;
	preds.vector[4] = -0.1;
	preds.vector[5] = -0.3;
	preds.vector[6] = 0.9;
	preds.vector[7] = 0.6;
	preds.vector[8] = -0.3;
	preds.vector[9] = 0.7;

	labs.vector[0] = 1;
	labs.vector[1] = -1;
	labs.vector[2] = 1;
	labs.vector[3] = 1;
	labs.vector[4] = -1;
	labs.vector[5] = -1;
	labs.vector[6] = 1;
	labs.vector[7] = 1;
	labs.vector[8] = -1;
	labs.vector[9] = -1;

	auto predictions = std::make_shared<BinaryLabels>(preds);
	auto labels = std::make_shared<BinaryLabels>(labs);

	auto sigmoid_calibration = std::make_shared<SigmoidCalibration>();
	auto calibrated = sigmoid_calibration->fit_binary(predictions, labels);
	EXPECT_EQ(calibrated, true);
	auto calibrated_labels = sigmoid_calibration->calibrate_binary(predictions);
	auto values = calibrated_labels->get_values();

	EXPECT_EQ(values[0], 0.656628663983293337);
	EXPECT_EQ(values[1], 0.159375349583615822);
	EXPECT_EQ(values[2], 0.718534704684106407);
	EXPECT_EQ(values[3], 0.819801347075004516);
	EXPECT_EQ(values[4], 0.201976857736835741);
	EXPECT_EQ(values[5], 0.124359200656053326);
	EXPECT_EQ(values[6], 0.819801347075004516);
	EXPECT_EQ(values[7], 0.656628663983293337);
	EXPECT_EQ(values[8], 0.124359200656053326);
	EXPECT_EQ(values[9], 0.718534704684106407);
}

TEST(SigmoidCalibrationTest, multiclass_calibration)
{
	index_t num_vec = 10;
	index_t num_class = 3;

	double preds[] = {
	    3.58412386,  -1.85426031, -3.01727279, 3.26476717,  -4.29277837,
	    -0.97842984, 3.71045102,  -2.87473617, -2.39465561, 0.50210539,
	    -0.90306753, -0.96823852, -0.23000778, 0.8901302,   -1.60981499,
	    -0.82280503, -6.85573966, 4.49250544,  2.51249731,  -1.9841395,
	    -1.97320905, -2.51812929, 2.75936145,  -0.9259119,  -0.92123693,
	    1.95678501,  -1.76459559, -1.30939982, 1.79817245,  -1.30609521};

	SGVector<float64_t> tgt({0, 0, 0, 0, 1, 2, 0, 1, 1, 1});

	auto predictions = std::make_shared<MulticlassLabels>(tgt);
	auto targets = std::make_shared<MulticlassLabels>(tgt);
	predictions->allocate_confidences_for(num_class);

	for (index_t i = 0; i < num_vec; i++)
	{
		SGVector<float64_t> confs(num_class);

		for (index_t j = 0; j < num_class; j++)
		{
			confs[j] = preds[i * num_class + j];
		}

		predictions->set_multiclass_confidences(i, confs);
	}
	auto calibration_method = std::make_shared<SigmoidCalibration>();

	auto calibrated = calibration_method->fit_multiclass(predictions, targets);

	EXPECT_EQ(calibrated, true);

	auto calib_result = calibration_method->calibrate_multiclass(predictions);

	float64_t expected_probs[] = {
	    0.75674645, 0.20144443, 0.04180912, 0.82689269, 0.05965391, 0.11345341,
	    0.81409222, 0.12715657, 0.05875121, 0.48765218, 0.38500051, 0.12734731,
	    0.31003506, 0.60319518, 0.08676976, 0.26630134, 0.01476002, 0.71893864,
	    0.72470815, 0.20387599, 0.07141586, 0.074445,   0.80897488, 0.11658012,
	    0.20620103, 0.71452701, 0.07927195, 0.16983872, 0.72947987, 0.10068142};

	for (index_t i = 0; i < num_vec; i++)
	{
		auto vals = calib_result->get_multiclass_confidences(i);

		for (index_t j = 0; j < vals.size(); j++)
		{
			EXPECT_NEAR(vals[j], expected_probs[i * num_class + j], 1E-5);
		}
	}
}
