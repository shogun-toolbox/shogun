#include <shogun/io/SGIO.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Time.h>

#include <shogun/mathematics/Math.h>
#include <shogun/structure/PrimalMosekSOSVM.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/FactorGraphModel.h>
#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/labels/FactorGraphLabels.h>

using namespace shogun;

void test(int32_t num_samples)
{
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

#ifdef SHOW_DATA
	// show labels
	for (unsigned int n = 0; n < num_samples; ++n) 
	{
		CFactorGraphObservation* fg_observ = CFactorGraphObservation::obtain_from_generic(labels->get_label(n));  
		SG_SPRINT("- sample %d:\n", n);
		SGVector<int32_t> fst = fg_observ->get_data();
		SGVector<int32_t>::display_vector(fst.vector, fst.vlen);
		SG_UNREF(fg_observ);
	}
#endif

	SG_SPRINT("----------------------------------------------------\n");

	CFactorGraphModel* model = new CFactorGraphModel(instances, labels, TREE_MAX_PROD, false);
	SG_REF(model);

	// initialize model parameters
	SGVector<float64_t> w_truth = w.clone();
	w.zero();
	factortype->set_w(w);
	model->add_factor_type(factortype);

#ifdef USE_MOSEK
	// create primal mosek solver
	CPrimalMosekSOSVM* primcp = new CPrimalMosekSOSVM(model, labels);
	SG_REF(primcp);
	primcp->set_regularization(0.01); // TODO: check 1000
#endif

	// create BMRM solver
	CDualLibQPBMSOSVM* bmrm = new CDualLibQPBMSOSVM(model, labels, 0.01);
	bmrm->set_verbose(false);
	SG_REF(bmrm);

	// create SGD solver
	CStochasticSOSVM* sgd = new CStochasticSOSVM(model, labels);
	sgd->set_num_iter(100);
	sgd->set_lambda(0.01);
	SG_REF(sgd);

	// timer
	CTime start;

	// train PrimalMosek
	primcp->train();
	float64_t t1 = start.cur_time_diff(false);

	// train BMRM
	bmrm->train();
	float64_t t2 = start.cur_time_diff(false);

	// train SGD
	sgd->train();
	float64_t t3 = start.cur_time_diff(false);

	SG_SPRINT(">>>> PrimalMosekSOSVM trained in %9.4f\n", t1);
	SG_SPRINT(">>>> BMRM trained in %9.4f\n", t2-t1);
	SG_SPRINT(">>>> SGD trained in %9.4f\n", t3-t2);

	// check w 
	primcp->get_slacks().display_vector("slacks");
	primcp->get_w().display_vector("w_mosek");
	bmrm->get_w().display_vector("w_bmrm");
	sgd->get_w().display_vector("w_sgd");
	w_truth.display_vector("w_truth");

#ifdef USE_MOSEK
	// Evaluation PrimalMosek
	CStructuredLabels* labels_primcp = CLabelsFactory::to_structured(primcp->apply());
	SG_REF(labels_primcp);

	float64_t acc_loss_primcp = 0.0;
	float64_t ave_loss_primcp = 0.0;
	
	for (int32_t i=0; i<num_samples; ++i)
	{
		CStructuredData* y_pred = labels_primcp->get_label(i);
		CStructuredData* y_truth = labels->get_label(i);
		acc_loss_primcp += model->delta_loss(y_truth, y_pred);
		SG_UNREF(y_pred);
		SG_UNREF(y_truth);
	}

	ave_loss_primcp = acc_loss_primcp / static_cast<float64_t>(num_samples);
	SG_SPRINT("primal mosek solver: average training loss = %f\n", ave_loss_primcp);
#endif

	// Evaluation BMRM
	CStructuredLabels* labels_bmrm = CLabelsFactory::to_structured(bmrm->apply());
	SG_REF(labels_bmrm);

	float64_t acc_loss_bmrm = 0.0;
	float64_t ave_loss_bmrm = 0.0;
	
	for (int32_t i=0; i<num_samples; ++i)
	{
		CStructuredData* y_pred = labels_bmrm->get_label(i);
		CStructuredData* y_truth = labels->get_label(i);
		acc_loss_bmrm += model->delta_loss(y_truth, y_pred);
		SG_UNREF(y_pred);
		SG_UNREF(y_truth);
	}

	ave_loss_bmrm = acc_loss_bmrm / static_cast<float64_t>(num_samples);
	SG_SPRINT("bmrm solver: average training loss = %f\n", ave_loss_bmrm);

	// Evaluation SGD
	CStructuredLabels* labels_sgd = CLabelsFactory::to_structured(sgd->apply());
	SG_REF(labels_sgd);

	float64_t acc_loss_sgd = 0.0;
	float64_t ave_loss_sgd = 0.0;
	
	for (int32_t i=0; i<num_samples; ++i)
	{
		CStructuredData* y_pred = labels_sgd->get_label(i);
		CStructuredData* y_truth = labels->get_label(i);
		acc_loss_sgd += model->delta_loss(y_truth, y_pred);
		SG_UNREF(y_pred);
		SG_UNREF(y_truth);
	}

	ave_loss_sgd = acc_loss_sgd / static_cast<float64_t>(num_samples);
	SG_SPRINT("sgd solver: average training loss = %f\n", ave_loss_sgd);

#ifdef USE_MOSEK
	SG_UNREF(labels_primcp);
	SG_UNREF(primcp);
#endif
	SG_UNREF(labels_sgd);
	SG_UNREF(labels_bmrm);
	SG_UNREF(sgd);
	SG_UNREF(bmrm);
	SG_UNREF(model);
	SG_UNREF(labels);
	SG_UNREF(instances);
	SG_UNREF(factortype);
}

int main(int argc, char * argv[])
{
	init_shogun_with_defaults();

	//sg_io->set_loglevel(MSG_DEBUG);
	
	test(100);

	exit_shogun();

	return 0;
}
