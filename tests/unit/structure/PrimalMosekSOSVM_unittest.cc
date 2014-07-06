#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Mosek.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/FactorGraphModel.h>
#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/labels/FactorGraphLabels.h>
#include <shogun/structure/PrimalMosekSOSVM.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef USE_MOSEK
TEST(PrimalMosekSOSVM, mosek_init_sosvm_w_bounds)
{
	int32_t num_samples = 10;
	CMath::init_random(17);

	// define factor type
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(8);
	w[0] = 0.3; // 0,0
	w[1] = 0.5; // 0,0
	w[2] = 1.0; // 1,0
	w[3] = 0.2; // 1,0
	w[4] = 0.05; // 0,1
	w[5] = 0.6; // 0,1
	w[6] = -0.2; // 1,1
	w[7] = 0.75; // 1,1
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
		SGVector<int32_t> vc(3);
		SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);

		CFactorGraph* fg = new CFactorGraph(vc);

		// add factors
		SGVector<float64_t> data1(2);
		data1[0] = 2.0 * CMath::random(0.0, 1.0) - 1.0;
		data1[1] = 2.0 * CMath::random(0.0, 1.0) - 1.0;
		SGVector<int32_t> var_index1(2);
		var_index1[0] = 0;
		var_index1[1] = 1;
		CFactor* fac1 = new CFactor(factortype, var_index1, data1);
		fg->add_factor(fac1);

		SGVector<float64_t> data2(2);
		data2[0] = 2.0 * CMath::random(0.0, 1.0) - 1.0;
		data2[1] = 2.0 * CMath::random(0.0, 1.0) - 1.0;
		SGVector<int32_t> var_index2(2);
		var_index2[0] = 1;
		var_index2[1] = 2;
		CFactor* fac2 = new CFactor(factortype, var_index2, data2);
		fg->add_factor(fac2);

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

	// create dummy primal mosek solver and check w's bounds
	CPrimalMosekSOSVM* primcp = new CPrimalMosekSOSVM(model, labels);
	SG_REF(primcp);
	primcp->set_regularization(0.01); // TODO: check 1000

	SGVector< float64_t > lb(w.vlen);
	SGVector< float64_t > ub(w.vlen);

	// ranged case
	SGVector< float64_t >::fill_vector(ub.vector, ub.vlen, 10);
	SGVector< float64_t >::fill_vector(lb.vector, ub.vlen, -10);
	primcp->set_lower_bounds(lb);
	primcp->set_upper_bounds(ub);
	primcp->train();
	w = primcp->get_w();

	//w = dummy_mosek_sosvm(model, lb, ub, 0);
	for (int32_t i = 0; i < w.vlen; i++)
	{
		EXPECT_LE(w[i], ub[i]);
		EXPECT_GE(w[i], lb[i]);
	}

	// fixed case
	lb.zero();
	ub.zero();
	primcp->set_lower_bounds(lb);
	primcp->set_upper_bounds(ub);
	primcp->train();
	w = primcp->get_w();

	//w = dummy_mosek_sosvm(model, lb, ub, 0);
	for (int32_t i = 0; i < w.vlen; i++)
		EXPECT_NEAR(w[i], ub[i], 1e-10);

	// lb case
	lb.zero();
	SGVector< float64_t >::fill_vector(ub.vector, ub.vlen, +CMath::INFTY);
	primcp->set_lower_bounds(lb);
	primcp->set_upper_bounds(ub);
	primcp->train();
	w = primcp->get_w();

	//w = dummy_mosek_sosvm(model, lb, ub, 0);
	for (int32_t i = 0; i < w.vlen; i++)
		EXPECT_GE(w[i], lb[i]);

	// ub case
	ub.zero();
	SGVector< float64_t >::fill_vector(lb.vector, lb.vlen, -CMath::INFTY);
	primcp->set_lower_bounds(lb);
	primcp->set_upper_bounds(ub);
	primcp->train();
	w = primcp->get_w();

	//w = dummy_mosek_sosvm(model, lb, ub, 0);
	for (int32_t i = 0; i < w.vlen; i++)
		EXPECT_LE(w[i], ub[i]);

	SG_UNREF(primcp);
	SG_UNREF(model);
	SG_UNREF(labels);
	SG_UNREF(instances);
	SG_UNREF(factortype);
}
#endif
