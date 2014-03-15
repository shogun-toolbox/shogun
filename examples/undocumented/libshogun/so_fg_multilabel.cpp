/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */
#include <shogun/io/LibSVMFile.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/base/DynArray.h>
#include <shogun/base/init.h>

#include <shogun/mathematics/Math.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/FactorGraphModel.h>
#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/labels/FactorGraphLabels.h>
#include <shogun/structure/SOSVMHelper.h>

using namespace shogun;

#define NUM_STATUS 2

const char FNAME_TRAIN[] = "../../../../data/multilabel/scene_train";
const char FNAME_TEST[]  = "../../../../data/multilabel/scene_test";

void read_data(const char * fname, SGMatrix<int32_t>& labels, SGMatrix<float64_t>& feats)
{
	// sparse data from matrix
	CLibSVMFile * svmfile = new CLibSVMFile(fname);

	SGSparseVector<float64_t>* spv_feats;
	SGVector<float64_t>* pv_labels;
	int32_t dim_feat;
	int32_t num_samples;
	int32_t num_classes;

	svmfile->get_sparse_matrix(spv_feats, dim_feat, num_samples, pv_labels, num_classes);

	SG_SPRINT("Number of the samples: %d\n", num_samples);
	SG_SPRINT("Dimention of the feature: %d\n", dim_feat+1);
	SG_SPRINT("Number of classes: %d\n", num_classes);

	feats  = SGMatrix<float64_t>(dim_feat+1, num_samples);
	labels = SGMatrix<int32_t>(num_classes, num_samples);
	feats.zero();
	labels.zero();

	for (int32_t i = 0; i < num_samples; i++)
	{
		SGVector<float64_t> v_feat = spv_feats[i].get_dense();
		SGVector<float64_t> v_labels = pv_labels[i];

		for (int32_t f = 0; f < v_feat.size(); f++)
			feats(f, i) = v_feat[f];
		
		feats(dim_feat, i) = 1.0; // bias

		for (int32_t l = 0; l < v_labels.size(); l++)
			labels((int32_t)v_labels[l], i) = 1;
	}

	SG_UNREF(svmfile);
	SG_FREE(spv_feats);
	SG_FREE(pv_labels);
}

SGMatrix< int32_t > get_tree_index()
{
	SGMatrix< int32_t > label_tree_index;
	
	// A tree structure is defined by a 2-d matrix where
	// each row stores the indecies of a pair of connect factors
	// Define label tree structure
	label_tree_index = SGMatrix< int32_t > (5, 2);
	label_tree_index[0] = 0;
	label_tree_index[1] = 0;
	label_tree_index[2] = 1;
	label_tree_index[3] = 4;
	label_tree_index[4] = 2;

	label_tree_index[5] = 2;
	label_tree_index[6] = 3;
	label_tree_index[7] = 4;
	label_tree_index[8] = 5;
	label_tree_index[9] = 5;
	
	return label_tree_index;
}

void build_factor_graph(SGMatrix<float64_t> feats, SGMatrix<int32_t> labels,
		CFactorGraphFeatures * fg_feats, CFactorGraphLabels * fg_labels,
		const DynArray<CTableFactorType *>& v_ftp_u,
		const DynArray<CTableFactorType *>& v_ftp_t)
{
	int32_t num_sample        = labels.num_cols;
	int32_t num_classes       = labels.num_rows;
	int32_t dim               = feats.num_rows;

	SGMatrix< int32_t > tree_index = get_tree_index();
	int32_t num_edges = tree_index.num_rows;

	// prepare features and labels in factor graph
	for (int32_t n = 0; n < num_sample; n++)
	{
		SGVector<int32_t> vc(num_classes);
		SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, NUM_STATUS);

		CFactorGraph * fg = new CFactorGraph(vc);

		float64_t * pfeat = feats.get_column_vector(n);
		SGVector<float64_t> feat_i(dim);
		memcpy(feat_i.vector, pfeat, dim * sizeof(float64_t));

		// add unary factors
		for (int32_t u = 0; u < num_classes; u++)
		{
			SGVector<int32_t> var_index_u(1);
			var_index_u[0] = u;
			CFactor * fac_u = new CFactor(v_ftp_u[u], var_index_u, feat_i);
			fg->add_factor(fac_u);
		}

		// add tree-structred factors
		for (int32_t t = 0; t < num_edges; t++)
		{
			SGVector<float64_t> data_t(1);
			data_t[0] = 1.0;
			SGVector<int32_t> var_index_t = tree_index.get_row_vector(t);
			CFactor * fac_t = new CFactor(v_ftp_t[t], var_index_t, data_t);
			fg->add_factor(fac_t);
		}
		
		// add factor graph instance
		fg_feats->add_sample(fg);

		// add label
		int32_t * plabs = labels.get_column_vector(n);
		SGVector<int32_t> states_gt(num_classes);
		memcpy(states_gt.vector, plabs, num_classes * sizeof(int32_t));
		SGVector<float64_t> loss_weights(num_classes);
		SGVector<float64_t>::fill_vector(loss_weights.vector, loss_weights.vlen, 1.0/num_classes);
		CFactorGraphObservation * fg_obs = new CFactorGraphObservation(states_gt, loss_weights);
		fg_labels->add_label(fg_obs);
	}
}

void evaluate(CFactorGraphModel * model, int32_t num_samples, CStructuredLabels * labels_sgd, \
              CFactorGraphLabels * fg_labels, float64_t & ave_error)
{
	float64_t acc_loss_sgd = 0.0;

	for (int32_t i = 0; i < num_samples; ++i)
	{
		CStructuredData * y_pred  = labels_sgd->get_label(i);
		CStructuredData * y_truth = fg_labels->get_label(i);
		acc_loss_sgd += model->delta_loss(y_truth, y_pred);

		SG_UNREF(y_pred);
		SG_UNREF(y_truth);
	}

	ave_error = acc_loss_sgd / static_cast<float64_t>(num_samples);
}

void test(SGMatrix<int32_t> labels_train, SGMatrix<float64_t> feats_train, SGMatrix<int32_t> labels_test, SGMatrix<float64_t> feats_test)
{
	int32_t num_sample_train  = labels_train.num_cols;
	int32_t num_classes       = labels_train.num_rows;
	int32_t dim               = feats_train.num_rows;

	// Build factor graph
	SGMatrix< int32_t > tree_index = get_tree_index();
	int32_t num_edges = tree_index.num_rows;

	int32_t tid;
	// we have l = num_classes different weights: w_1, w_2, ..., w_l
	// so we create num_classes different unary factor types
	DynArray<CTableFactorType *> v_ftp_u;

	for (int32_t u = 0; u < num_classes; u++)
	{
		tid = u;
		SGVector<int32_t> card_u(1);
		card_u[0] = NUM_STATUS;
		SGVector<float64_t> w_u(dim * NUM_STATUS);
		w_u.zero();
		v_ftp_u.append_element(new CTableFactorType(tid, card_u, w_u));
	}
	
	// define factor type: tree edge factor
	// note that each edge is a new type
	DynArray<CTableFactorType *> v_ftp_t;

	for (int32_t t = 0; t < num_edges; t++)
	{
		tid = t + num_classes;
		SGVector<int32_t> card_t(2);
		card_t[0] = NUM_STATUS;
		card_t[1] = NUM_STATUS;
		SGVector<float64_t> w_t(NUM_STATUS * NUM_STATUS);
		w_t.zero();
		v_ftp_t.append_element(new CTableFactorType(tid, card_t, w_t));
	}

	// prepare features and labels in factor graph
	CFactorGraphFeatures * fg_feats_train = new CFactorGraphFeatures(num_sample_train);
	SG_REF(fg_feats_train);
	CFactorGraphLabels * fg_labels_train = new CFactorGraphLabels(num_sample_train);
	SG_REF(fg_labels_train);

	build_factor_graph(feats_train, labels_train, fg_feats_train, fg_labels_train, v_ftp_u, v_ftp_t);

	SG_SPRINT("----------------------------------------------------\n");

	CFactorGraphModel * model = new CFactorGraphModel(fg_feats_train, fg_labels_train, TREE_MAX_PROD, false);
	SG_REF(model);

	// initialize model parameters
	for (int32_t u = 0; u < num_classes; u++)
		model->add_factor_type(v_ftp_u[u]);

	for (int32_t t = 0; t < num_edges; t++)
		model->add_factor_type(v_ftp_t[t]);

	// create SGD solver
	CStochasticSOSVM * sgd = new CStochasticSOSVM(model, fg_labels_train, true);
	sgd->set_num_iter(200);
	sgd->set_lambda(0.0001);
	SG_REF(sgd);

	// timer
	CTime start;
	// train SGD
	sgd->train();
	float64_t t2 = start.cur_time_diff(false);

	SG_SPRINT(">>>> SGD trained in %9.4f\n", t2);

	// Evaluation SGD
	CStructuredLabels * labels_sgd = CLabelsFactory::to_structured(sgd->apply());
	SG_REF(labels_sgd);

	float64_t ave_loss_sgd = 0.0;

	evaluate(model, num_sample_train, labels_sgd, fg_labels_train, ave_loss_sgd);

	SG_SPRINT("sgd solver: average training loss = %f\n", ave_loss_sgd);
	SG_UNREF(labels_sgd);

	if(labels_test.num_cols > 0)
	{
		SG_SPRINT("----------------------------------------------------\n");	

		// prepare features and labels in factor graph
		int32_t num_sample_test  = labels_test.num_cols;
		CFactorGraphFeatures * fg_feats_test = new CFactorGraphFeatures(num_sample_test);
		SG_REF(fg_feats_test);
		CFactorGraphLabels * fg_labels_test = new CFactorGraphLabels(num_sample_test);
		SG_REF(fg_labels_test);
		build_factor_graph(feats_test, labels_test, fg_feats_test, fg_labels_test, v_ftp_u, v_ftp_t);

		sgd->set_features(fg_feats_test);
		sgd->set_labels(fg_labels_test);
		labels_sgd = CLabelsFactory::to_structured(sgd->apply());

		evaluate(model, num_sample_test, labels_sgd, fg_labels_test, ave_loss_sgd);
		SG_REF(labels_sgd);

		SG_SPRINT("sgd solver: average testing error = %f\n", ave_loss_sgd);

		SG_UNREF(fg_feats_test);
		SG_UNREF(fg_labels_test);
	}

	SG_UNREF(labels_sgd);
	SG_UNREF(sgd);
	SG_UNREF(model);
	SG_UNREF(fg_feats_train);
	SG_UNREF(fg_labels_train);
}

int main(int argc, char * argv[])
{
	init_shogun_with_defaults();
		
	// Training data
	SGMatrix<int32_t> labels_train;
	SGMatrix<float64_t> feats_train;
	
	// Testing data
	SGMatrix<int32_t> labels_test;
	SGMatrix<float64_t> feats_test;

	// Train and test with real data
	FILE * pfile = fopen(FNAME_TRAIN, "r");
	
	if (pfile == NULL)
	{
		SG_SPRINT("Unable to open file: %s\n", FNAME_TRAIN);
		return 0;
	}

	fclose(pfile);

	pfile = fopen(FNAME_TEST, "r");
	
	if (pfile == NULL)
	{
		SG_SPRINT("Unable to open file: %s\n", FNAME_TEST);
		return 0;
	}

	fclose(pfile);

	SG_SPRINT("Experiment with real dataset: \n");	
	
	read_data(FNAME_TRAIN, labels_train, feats_train);
	read_data(FNAME_TEST, labels_test, feats_test);
	
	test(labels_train, feats_train, labels_test, feats_test);
	
	exit_shogun();

	return 0;
}
