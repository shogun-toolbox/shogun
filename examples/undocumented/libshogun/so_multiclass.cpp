#include <shogun/base/init.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/evaluation/StructuredAccuracy.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/common.h>
#include <shogun/loss/HingeLoss.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <shogun/structure/MulticlassModel.h>
#include <shogun/structure/PrimalMosekSOSVM.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/lib/Time.h>
#include <shogun/base/init.h>

#include <stdio.h>

using namespace shogun;

#define	DIMS		2
#define EPSILON	10e-5
#define	NUM_SAMPLES	100
#define NUM_CLASSES	10

char FNAME[] = "data.out";

void gen_rand_data(SGVector< float64_t > labs, SGMatrix< float64_t > feats)
{
	float64_t means[DIMS];
	float64_t  stds[DIMS];

	FILE* pfile = fopen(FNAME, "w");

	for ( int32_t c = 0 ; c < NUM_CLASSES ; ++c )
	{
		for ( int32_t j = 0 ; j < DIMS ; ++j )
		{
			means[j] = Math::random(-100, 100);
			 stds[j] = Math::random(   1,   5);
		}

		for ( int32_t i = 0 ; i < NUM_SAMPLES ; ++i )
		{
			labs[c*NUM_SAMPLES+i] = c;

			fprintf(pfile, "%d", c);

			for ( int32_t j = 0 ; j < DIMS ; ++j )
			{
				feats[(c*NUM_SAMPLES+i)*DIMS + j] =
					Math::normal_random(means[j], stds[j]);

				fprintf(pfile, " %f", feats[(c*NUM_SAMPLES+i)*DIMS + j]);
			}

			fprintf(pfile, "\n");
		}
	}

	fclose(pfile);
}

void read_data(SGVector< float64_t > labs, SGMatrix< float64_t > feats)
{
	FILE* pfile = fopen(FNAME, "r");
	if (pfile == NULL)
		SG_SERROR("Unable to open file: %s\n", FNAME);

	int32_t label, idx;
	float32_t value;

	for ( int32_t i = 0 ; i < NUM_SAMPLES*NUM_CLASSES ; ++i )
	{
		fscanf(pfile, "%d", &label);

		labs[i] = label;

		for ( int32_t j = 0 ; j < DIMS ; ++j )
		{
			fscanf(pfile, "%d:%f", &idx, &value);
			feats[i*DIMS + j] = value;
		}
	}

	fclose(pfile);
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();

	SGVector< float64_t > labs(NUM_CLASSES*NUM_SAMPLES);
	SGMatrix< float64_t > feats(DIMS, NUM_CLASSES*NUM_SAMPLES);

	gen_rand_data(labs, feats);
	//read_data(labs, feats);

	// Create train labels
	CMulticlassSOLabels* labels = new CMulticlassSOLabels(labs);
	MulticlassLabels*  mlabels = new MulticlassLabels(labs);

	// Create train features
	DenseFeatures< float64_t >* features = new DenseFeatures< float64_t >(feats);

	// Create structured model
	CMulticlassModel* model = new CMulticlassModel(features, labels);

	// Create SO-SVM
	CPrimalMosekSOSVM* sosvm = new CPrimalMosekSOSVM(model, labels);
	CDualLibQPBMSOSVM* bundle = new CDualLibQPBMSOSVM(model, labels, 100);
	CStochasticSOSVM* sgd = new CStochasticSOSVM(model, labels);
	bundle->set_verbose(false);

	Time start;
	sosvm->train();
	float64_t t1 = start.cur_time_diff(false);
	bundle->train();
	float64_t t2 = start.cur_time_diff(false);
	sgd->train();
	float64_t t3 = start.cur_time_diff(false);
	SG_SPRINT(">>>> PrimalMosekSOSVM trained in %9.4f\n", t1);
	SG_SPRINT(">>>> BMRM trained in %9.4f\n", t2-t1);
	SG_SPRINT(">>>> SGD trained in %9.4f\n", t3-t2);
	StructuredLabels* out = sosvm->apply()->as<StructuredLabels>();
	StructuredLabels* bout = bundle->apply()->as<StructuredLabels>();
	StructuredLabels* sout = sgd->apply()->as<StructuredLabels>();

	// Create liblinear svm classifier with L2-regularized L2-loss
	LibLinear* svm = new LibLinear(L2R_L2LOSS_SVC);

	// Add some configuration to the svm
	svm->set_epsilon(EPSILON);
	svm->set_bias_enabled(false);

	// Create a multiclass svm classifier that consists of several of the previous one
	LinearMulticlassMachine* mc_svm =
			new LinearMulticlassMachine( new MulticlassOneVsRestStrategy(),
			(DotFeatures*) features, svm, mlabels);

	// Train the multiclass machine using the data passed in the constructor
	mc_svm->train();
	MulticlassLabels* mout = mc_svm->apply()->as<MulticlassLabels>();

	SGVector< float64_t > w = sosvm->get_w();
	for ( int32_t i = 0 ; i < w.vlen ; ++i )
		SG_SPRINT("%10f ", w[i]);
	SG_SPRINT("\n\n");

	for ( int32_t i = 0 ; i < NUM_CLASSES ; ++i )
	{
		LinearMachine* lm = (LinearMachine*) mc_svm->get_machine(i);
		SGVector< float64_t > mw = lm->get_w();
		for ( int32_t j = 0 ; j < mw.vlen ; ++j )
			SG_SPRINT("%10f ", mw[j]);

	}
	SG_SPRINT("\n");

	StructuredAccuracy* structured_evaluator = new StructuredAccuracy();
	MulticlassAccuracy* multiclass_evaluator = new MulticlassAccuracy();

	SG_SPRINT("SO-SVM: %5.2f%\n", 100.0*structured_evaluator->evaluate(out, labels));
	SG_SPRINT("BMRM:   %5.2f%\n", 100.0*structured_evaluator->evaluate(bout, labels));
	SG_SPRINT("SGD:   %5.2f%\n", 100.0*structured_evaluator->evaluate(sout, labels));
	SG_SPRINT("MC:     %5.2f%\n", 100.0*multiclass_evaluator->evaluate(mout, mlabels));

	// Free memory
	exit_shogun();

	return 0;
}
