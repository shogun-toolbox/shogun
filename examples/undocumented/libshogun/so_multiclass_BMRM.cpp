/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#include <shogun/base/init.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/common.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <shogun/structure/BmrmStatistics.h>
#include <shogun/structure/MulticlassModel.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/features/streaming/StreamingSparseFeatures.h>

using namespace shogun;

#define	DIMS		2
#define EPSILON	10e-5
#define	NUM_SAMPLES	100
#define NUM_CLASSES	10

char FNAME[] = "data.svmlight";

/** Reads multiclass trainig data stored in svmlight format (i.e. label nz_idx_1:value1 nz_idx_2:value2 ... nz_idx_N:valueN )
 *
 * @param fname    path to file with training data
 * @param DIM      dimension of features
 * @param N        number of feature vectors
 * @param labs     vector with labels
 * @param feats    matrix with features
 */
void read_data(const char fname[], uint32_t DIM, uint32_t N, SGVector<float64_t> labs, SGMatrix<float64_t> feats)
{
	CStreamingAsciiFile* file=new CStreamingAsciiFile(fname);
	SG_REF(file);

	CStreamingSparseFeatures< float64_t >* stream_features=
		new CStreamingSparseFeatures< float64_t >(file, true, 1024);
	SG_REF(stream_features);

	SGVector<float64_t > vec(DIM);

	stream_features->start_parser();

	uint32_t num_vectors=0;

	while (stream_features->get_next_example())
	{
		vec.zero();
		stream_features->add_to_dense_vec(1.0, vec, DIM);

		labs[num_vectors]=stream_features->get_label();

		for (uint32_t i=0; i<DIM; ++i)
			feats[num_vectors*DIM+i]=vec[i];

		num_vectors++;
		stream_features->release_example();
	}

	stream_features->end_parser();

	SG_UNREF(stream_features);
}

/** Generates random multiclass training data and stores them in svmlight format
 *
 * @param labs      returned vector with labels
 * @param feats     returned matrix with features
 */
void gen_rand_data(SGVector< float64_t > labs, SGMatrix< float64_t > feats)
{
	float64_t means[DIMS];
	float64_t  stds[DIMS];

	FILE* pfile = fopen(FNAME, "w");

	CMath::init_random(17);

	for ( int32_t c = 0 ; c < NUM_CLASSES ; ++c )
	{
		for ( int32_t j = 0 ; j < DIMS ; ++j )
		{
			means[j] = CMath::random(-100, 100);
			stds[j] = CMath::random(   1,   5);
		}

		for ( int32_t i = 0 ; i < NUM_SAMPLES ; ++i )
		{
			labs[c*NUM_SAMPLES+i] = c;

			fprintf(pfile, "%d", c);

			for ( int32_t j = 0 ; j < DIMS ; ++j )
			{
				feats[(c*NUM_SAMPLES+i)*DIMS + j] =
					CMath::normal_random(means[j], stds[j]);

				fprintf(pfile, " %d:%f", j+1, feats[(c*NUM_SAMPLES+i)*DIMS + j]);
			}

			fprintf(pfile, "\n");
		}
	}

	fclose(pfile);
}

int main(int argc, char * argv[])
{
	// initialization
	//-------------------------------------------------------------------------

	float64_t lambda=0.01, eps=0.01;
	bool icp=1;
	uint32_t cp_models=1;
	ESolver solver=BMRM;
	uint32_t feat_dim, num_feat;

	init_shogun_with_defaults();

	if (argc > 1 && argc < 8)
	{
		SG_SERROR("Usage: so_multiclass_BMRM <data.in> <feat_dim> <num_feat> <lambda> <icp> <epsilon> <solver> [<cp_models>]\n");
		return -1;
	}

	if (argc > 1)
	{
		// parse command line arguments for parameters setting

		SG_SPRINT("arg[1] = %s\n", argv[1]);

		feat_dim=::atoi(argv[2]);
		num_feat=::atoi(argv[3]);
		lambda=::atof(argv[4]);
		icp=::atoi(argv[5]);
		eps=::atof(argv[6]);

		if (strcmp("BMRM", argv[7])==0)
			solver=BMRM;

		if (strcmp("PPBMRM", argv[7])==0)
			solver=PPBMRM;

		if (strcmp("P3BMRM", argv[7])==0)
			solver=P3BMRM;

		if (argc > 8)
		{
			cp_models=::atoi(argv[8]);
		}
	}
	else
	{
		// default parameters

		feat_dim=DIMS;
		num_feat=NUM_SAMPLES*NUM_CLASSES;
		lambda=1e3;
		icp=1;
		eps=0.01;
		solver=BMRM;
	}

	SGVector<float64_t> labs(num_feat);

	SGMatrix<float64_t> feats(feat_dim, num_feat);

	if (argc==1)
	{
		gen_rand_data(labs, feats);
	}
	else
	{
		// read data
		read_data(argv[1], feat_dim, num_feat, labs, feats);
	}

	// Create train labels
	CMulticlassSOLabels* labels = new CMulticlassSOLabels(labs);

	// Create train features
	CDenseFeatures< float64_t >* features =
		new CDenseFeatures< float64_t >(feats);

	// Create structured model
	CMulticlassModel* model = new CMulticlassModel(features, labels);

	// Create SO-SVM
	CDualLibQPBMSOSVM* sosvm =
		new CDualLibQPBMSOSVM(
				model,
				labels,
				lambda);
	SG_REF(sosvm);

	sosvm->set_cleanAfter(10);
	sosvm->set_cleanICP(icp);
	sosvm->set_TolRel(eps);
	sosvm->set_cp_models(cp_models);
	sosvm->set_solver(solver);

	// Train
	//-------------------------------------------------------------------------

	SG_SPRINT("Train using lambda = %lf ICP removal = %d \n",
			sosvm->get_lambda(), sosvm->get_cleanICP());

	sosvm->train();

	BmrmStatistics res = sosvm->get_result();

	SG_SPRINT("result = { Fp=%lf, Fd=%lf, nIter=%d, nCP=%d, nzA=%d, exitflag=%d }\n",
			res.Fp, res.Fd, res.nIter, res.nCP, res.nzA, res.exitflag);

	CStructuredLabels* out =
		CLabelsFactory::to_structured(sosvm->apply());
	SG_REF(out);

	SG_SPRINT("\n");

	// Compute error
	//-------------------------------------------------------------------------
	float64_t error=0.0;

	for (uint32_t i=0; i<num_feat; ++i)
	{
		CRealNumber* rn = CRealNumber::obtain_from_generic( out->get_label(i) );
		error+=(rn->value==labs.get_element(i)) ? 0.0 : 1.0;
		SG_UNREF(rn);	// because of out->get_label(i) above
	}

	SG_SPRINT("Error = %lf %% \n", error/num_feat*100);


	// Free memory
	SG_UNREF(sosvm);
	SG_UNREF(out);

	exit_shogun();

	return 0;
}
