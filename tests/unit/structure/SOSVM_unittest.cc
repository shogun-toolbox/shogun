#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>

#include <shogun/structure/FactorType.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/FactorGraphModel.h>
#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/labels/FactorGraphLabels.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/structure/FWSOSVM.h>
#include <shogun/structure/SOSVMHelper.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SOSVM, sgd_check_w_helper)
{
	int32_t num_samples = 1;

	// define factor type
	SGVector<int32_t> card(1);
	card[0] = 2;
	SGVector<float64_t> w(2);
	w[0] = -1;
	w[1] = 1;
	int32_t tid = 0;
	CTableFactorType* factortype = new CTableFactorType(tid, card, w);
	SG_REF(factortype);

	// create features and labels
	CFactorGraphFeatures* instances = new CFactorGraphFeatures(num_samples);
	SG_REF(instances);
	CFactorGraphLabels* labels = new CFactorGraphLabels(num_samples);
	SG_REF(labels);

	for (int32_t n = 0; n < num_samples; ++n)
	{
		// factor graph
		SGVector<int32_t> vc(1);
		vc[0] = 2;

		CFactorGraph* fg = new CFactorGraph(vc);

		// add factors
		SGVector<float64_t> data1(1);
		data1[0] = -1.0;
		SGVector<int32_t> var_index1(1);
		var_index1[0] = 0;
		CFactor* fac1 = new CFactor(factortype, var_index1, data1);
		fg->add_factor(fac1);

		// add factor graph instance
		instances->add_sample(fg);

		fg->connect_components();
		fg->compute_energies();

		CMAPInference infer_met(fg, TREE_MAX_PROD);
		infer_met.inference();

		CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();

		// add ground truth states
		labels->add_label(fg_observ);
		SG_UNREF(fg_observ);
	}

	CFactorGraphModel* model = new CFactorGraphModel(instances, labels, TREE_MAX_PROD, false);
	SG_REF(model);

	// initialize model parameters
	SGVector<float64_t> w_truth = w.clone();
	w.zero();
	factortype->set_w(w);
	model->add_factor_type(factortype);

	// SGD solver
	CStochasticSOSVM* sgd = new CStochasticSOSVM(model, labels, false, false);
	sgd->set_num_iter(1);
	sgd->set_lambda(1.0);
	sgd->train();
	w = sgd->get_w();

	for (int32_t i = 0; i < w.vlen; i++)
		EXPECT_NEAR(w_truth[i], w[i], 1E-10);

	EXPECT_NEAR(1.0, CSOSVMHelper::primal_objective(w, model, 1.0), 1E-10);
	EXPECT_NEAR(0.0, CSOSVMHelper::average_loss(w, model), 1E-10);

	SG_UNREF(sgd);
	SG_UNREF(model);
	SG_UNREF(labels);
	SG_UNREF(instances);
	SG_UNREF(factortype);
}

TEST(SOSVM, fw_check_w_helper)
{
	int32_t num_samples = 1;

	// define factor type
	SGVector<int32_t> card(1);
	card[0] = 2;
	SGVector<float64_t> w(2);
	w[0] = -sqrt(0.5);
	w[1] = sqrt(0.5);
	int32_t tid = 0;
	CTableFactorType* factortype = new CTableFactorType(tid, card, w);
	SG_REF(factortype);

	// create features and labels
	CFactorGraphFeatures* instances = new CFactorGraphFeatures(num_samples);
	SG_REF(instances);
	CFactorGraphLabels* labels = new CFactorGraphLabels(num_samples);
	SG_REF(labels);

	for (int32_t n = 0; n < num_samples; ++n)
	{
		// factor graph
		SGVector<int32_t> vc(1);
		vc[0] = 2;

		CFactorGraph* fg = new CFactorGraph(vc);

		// add factors
		SGVector<float64_t> data1(1);
		data1[0] = -sqrt(0.5);
		SGVector<int32_t> var_index1(1);
		var_index1[0] = 0;
		CFactor* fac1 = new CFactor(factortype, var_index1, data1);
		fg->add_factor(fac1);

		// add factor graph instance
		instances->add_sample(fg);

		fg->connect_components();
		fg->compute_energies();

		CMAPInference infer_met(fg, TREE_MAX_PROD);
		infer_met.inference();

		CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();

		// add ground truth states
		labels->add_label(fg_observ);
		SG_UNREF(fg_observ);
	}

	CFactorGraphModel* model = new CFactorGraphModel(instances, labels, TREE_MAX_PROD, false);
	SG_REF(model);

	// initialize model parameters
	SGVector<float64_t> w_truth = w.clone();
	w.zero();
	factortype->set_w(w);
	model->add_factor_type(factortype);

	// FW solver
	float64_t aloss_fw = CSOSVMHelper::average_loss(w, model, true);
	CFWSOSVM* fw = new CFWSOSVM(model, labels, false, false);
	fw->set_num_iter(1);
	fw->set_lambda(1.0);
	fw->set_gap_threshold(0.0);
	fw->train();
	w = fw->get_w();

	for (int32_t i = 0; i < w.vlen; i++)
		EXPECT_NEAR(w_truth[i], w[i], 1E-10);

	EXPECT_NEAR(0.5, CSOSVMHelper::primal_objective(w, model, 1.0), 1E-10);
	EXPECT_NEAR(1.0, aloss_fw, 1E-10);
	EXPECT_NEAR(0.5, CSOSVMHelper::dual_objective(w, aloss_fw, 1.0), 1E-10);

	SG_UNREF(fw);
	SG_UNREF(model);
	SG_UNREF(labels);
	SG_UNREF(instances);
	SG_UNREF(factortype);
}
