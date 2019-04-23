#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Time.h>

#include <shogun/mathematics/Math.h>
#include <shogun/structure/PrimalMosekSOSVM.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/structure/FWSOSVM.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/FactorGraphModel.h>
#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/labels/FactorGraphLabels.h>

using namespace shogun;

void test(int32_t num_samples)
{
	Math::init_random(17);

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
	TableFactorType* factortype = new TableFactorType(tid, card, w);

	// create features and labels
	FactorGraphFeatures* instances = new FactorGraphFeatures(num_samples);
	FactorGraphLabels* labels = new FactorGraphLabels(num_samples);

	for (int32_t n = 0; n < num_samples; ++n)
	{
		// factor graph
		SGVector<int32_t> vc(3);
		SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);

		FactorGraph* fg = new FactorGraph(vc);

		// add factors
		SGVector<float64_t> data1(2);
		data1[0] = 2.0 * Math::random(0.0, 1.0) - 1.0;
		data1[1] = 2.0 * Math::random(0.0, 1.0) - 1.0;
		SGVector<int32_t> var_index1(2);
		var_index1[0] = 0;
		var_index1[1] = 1;
		Factor* fac1 = new Factor(factortype, var_index1, data1);
		fg->add_factor(fac1);

		SGVector<float64_t> data2(2);
		data2[0] = 2.0 * Math::random(0.0, 1.0) - 1.0;
		data2[1] = 2.0 * Math::random(0.0, 1.0) - 1.0;
		SGVector<int32_t> var_index2(2);
		var_index2[0] = 1;
		var_index2[1] = 2;
		Factor* fac2 = new Factor(factortype, var_index2, data2);
		fg->add_factor(fac2);

		// add factor graph instance
		instances->add_sample(fg);

		fg->connect_components();
		fg->compute_energies();

		MAPInference infer_met(fg, TREE_MAX_PROD);
		infer_met.inference();

		FactorGraphObservation* fg_observ = infer_met.get_structured_outputs();

		// add ground truth states
		labels->add_label(fg_observ);
	}

#ifdef SHOW_DATA
	// show labels
	for (unsigned int n = 0; n < num_samples; ++n)
	{
		FactorGraphObservation* fg_observ = FactorGraphObservation::obtain_from_generic(labels->get_label(n));
		SG_SPRINT("- sample %d:\n", n);
		SGVector<int32_t> fst = fg_observ->get_data();
		SGVector<int32_t>::display_vector(fst.vector, fst.vlen);
	}
#endif

	SG_SPRINT("----------------------------------------------------\n");

	FactorGraphModel* model = new FactorGraphModel(instances, labels, TREE_MAX_PROD, false);

	// initialize model parameters
	SGVector<float64_t> w_truth = w.clone();
	w.zero();
	factortype->set_w(w);
	model->add_factor_type(factortype);

#undef USE_MOSEK
#ifdef USE_MOSEK
	// create primal mosek solver
	CPrimalMosekSOSVM* primcp = new CPrimalMosekSOSVM(model, labels);
	primcp->set_regularization(0.01); // TODO: check 1000
#endif

	// create BMRM solver
	CDualLibQPBMSOSVM* bmrm = new CDualLibQPBMSOSVM(model, labels, 0.01);
	bmrm->set_verbose(false);

	// create SGD solver
	CStochasticSOSVM* sgd = new CStochasticSOSVM(model, labels);
	sgd->set_num_iter(100);
	sgd->set_lambda(0.01);

	// create FW solver
	CFWSOSVM* fw = new CFWSOSVM(model, labels);
	fw->set_num_iter(100);
	fw->set_lambda(0.01);
	fw->set_gap_threshold(0.01);

	// timer
	Time start;
	float64_t t1 = start.cur_time_diff(false);

#ifdef USE_MOSEK
	// train PrimalMosek
	primcp->train();
	float64_t t1 = start.cur_time_diff(false);
#endif

	// train BMRM
	bmrm->train();
	float64_t t2 = start.cur_time_diff(false);

	// train SGD
	sgd->train();
	float64_t t3 = start.cur_time_diff(false);

	// train FW
	fw->train();
	float64_t t4 = start.cur_time_diff(false);

	SG_SPRINT(">>>> PrimalMosekSOSVM trained in %9.4f\n", t1);
	SG_SPRINT(">>>> BMRM trained in %9.4f\n", t2-t1);
	SG_SPRINT(">>>> SGD trained in %9.4f\n", t3-t2);
	SG_SPRINT(">>>> FW trained in %9.4f\n", t4-t3);

	// check w
#ifdef USE_MOSEK
	primcp->get_slacks().display_vector("slacks");
	primcp->get_w().display_vector("w_mosek");
#endif
	bmrm->get_w().display_vector("w_bmrm");
	sgd->get_w().display_vector("w_sgd");
	fw->get_w().display_vector("w_fw");
	w_truth.display_vector("w_truth");

#ifdef USE_MOSEK
	// Evaluation PrimalMosek
	StructuredLabels* labels_primcp = primcp->apply()->as<StructuredLabels>();

	float64_t acc_loss_primcp = 0.0;
	float64_t ave_loss_primcp = 0.0;

	for (int32_t i=0; i<num_samples; ++i)
	{
		StructuredData* y_pred = labels_primcp->get_label(i);
		StructuredData* y_truth = labels->get_label(i);
		acc_loss_primcp += model->delta_loss(y_truth, y_pred);
	}

	ave_loss_primcp = acc_loss_primcp / static_cast<float64_t>(num_samples);
	SG_SPRINT("primal mosek solver: average training loss = %f\n", ave_loss_primcp);
#endif

	// Evaluation BMRM
	StructuredLabels* labels_bmrm = bmrm->apply()->as<StructuredLabels>();

	float64_t acc_loss_bmrm = 0.0;
	float64_t ave_loss_bmrm = 0.0;

	for (int32_t i=0; i<num_samples; ++i)
	{
		StructuredData* y_pred = labels_bmrm->get_label(i);
		StructuredData* y_truth = labels->get_label(i);
		acc_loss_bmrm += model->delta_loss(y_truth, y_pred);
	}

	ave_loss_bmrm = acc_loss_bmrm / static_cast<float64_t>(num_samples);
	SG_SPRINT("bmrm solver: average training loss = %f\n", ave_loss_bmrm);

	// Evaluation SGD
	StructuredLabels* labels_sgd = sgd->apply()->as<StructuredLabels>();

	float64_t acc_loss_sgd = 0.0;
	float64_t ave_loss_sgd = 0.0;

	for (int32_t i=0; i<num_samples; ++i)
	{
		StructuredData* y_pred = labels_sgd->get_label(i);
		StructuredData* y_truth = labels->get_label(i);
		acc_loss_sgd += model->delta_loss(y_truth, y_pred);
	}

	ave_loss_sgd = acc_loss_sgd / static_cast<float64_t>(num_samples);
	SG_SPRINT("sgd solver: average training loss = %f\n", ave_loss_sgd);

	// Evaluation FW
	StructuredLabels* labels_fw = fw->apply()->as<StructuredLabels>();

	float64_t acc_loss_fw = 0.0;
	float64_t ave_loss_fw = 0.0;

	for (int32_t i=0; i<num_samples; ++i)
	{
		StructuredData* y_pred = labels_fw->get_label(i);
		StructuredData* y_truth = labels->get_label(i);
		acc_loss_fw += model->delta_loss(y_truth, y_pred);
	}

	ave_loss_fw = acc_loss_fw / static_cast<float64_t>(num_samples);
	SG_SPRINT("fw solver: average training loss = %f\n", ave_loss_fw);

#ifdef USE_MOSEK
#endif
}

int main(int argc, char * argv[])
{
	//env()->io()->set_loglevel(MSG_DEBUG);

	test(100);

	return 0;
}
