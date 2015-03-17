/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Jiaolong Xu
 * Copyright (C) 2015 Jiaolong Xu
 */

#include <shogun/structure/FactorGraphDataGenerator.h>

using namespace shogun;

CFactorGraphDataGenerator::CFactorGraphDataGenerator(): CSGObject()
{}

CFactorGraph* CFactorGraphDataGenerator::simple_chain_graph()
{
	// ftype
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 0.0; // 0,0
	w[1] = 0.3; // 1,0
	w[2] = 0.2; // 0,1
	w[3] = 0.0; // 1,1
	int32_t tid = 0;
	CTableFactorType* ft_pairwise = new CTableFactorType(tid, card, w);
	SG_REF(ft_pairwise);

	SGVector<int32_t> card1(1);
	card1[0] = 2;
	SGVector<float64_t> w1(2);
	w1[0] = 0.1;
	w1[1] = 0.7;
	int32_t tid1 = 1;
	CTableFactorType* ft_unary_1 = new CTableFactorType(tid1, card1, w1);
	SG_REF(ft_unary_1);

	SGVector<int32_t> card2(1);
	card2[0] = 2;
	SGVector<float64_t> w2(2);
	w2[0] = 0.3;
	w2[1] = 0.6;
	int32_t tid2 = 2;
	CTableFactorType* ft_unary_2 = new CTableFactorType(tid2, card2, w2);
	SG_REF(ft_unary_2);

	// fg
	SGVector<int32_t> vc(2);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	CFactorGraph* fg = new CFactorGraph(vc);
	SG_REF(fg);

	// add factors
	SGVector<float64_t> data;
	SGVector<int32_t> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	CFactor* fac1 = new CFactor(ft_pairwise, var_index, data);
	fg->add_factor(fac1);

	SGVector<int32_t> var_index1(1);
	var_index1[0] = 0;
	CFactor* fac2 = new CFactor(ft_unary_1, var_index1, data);
	fg->add_factor(fac2);

	SGVector<int32_t> var_index2(1);
	var_index2[0] = 1;
	CFactor* fac3 = new CFactor(ft_unary_2, var_index2, data);
	fg->add_factor(fac3);

	SG_UNREF(ft_pairwise);
	SG_UNREF(ft_unary_1);
	SG_UNREF(ft_unary_2);

	// energy table
	fg->compute_energies();

	fg->connect_components();

	return fg;
}

int32_t CFactorGraphDataGenerator::grid_to_index(int32_t x, int32_t y, int32_t w)
{
	return x + w * y;
}

void CFactorGraphDataGenerator::truncate_energy(float64_t &A, float64_t &B, float64_t &C, float64_t &D)
{
	if (A + D > C + B)
	{
		SG_SDEBUG("\nTruncate initialized data to ensure submodularity.\n");
		float64_t delta = A + D - C - B;
		float64_t subtrA = delta / 3;
		A = A - subtrA;
		C = C + subtrA;
		B = B + (delta - subtrA * 2) + 0.0001; // for numeric issue
	}
}

CFactorGraph* CFactorGraphDataGenerator::random_chain_graph(SGVector<int> &assignment_expect, float64_t &min_energy_expect, int32_t N)
{
	CMath::init_random(17);

	// ftype
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w;
	int32_t tid = 0;
	CTableFactorType* factortype = new CTableFactorType(tid, card, w);
	SG_REF(factortype);

	SGVector<int32_t> card1(1);
	card1[0] = 2;
	SGVector<float64_t> w1;
	int32_t tid1 = 1;
	CTableFactorType* factortype1 = new CTableFactorType(tid1, card1, w1);
	SG_REF(factortype1);

	// fg
	SGVector<int32_t> vc(N * N);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	CFactorGraph* fg = new CFactorGraph(vc);
	SG_REF(fg);

	// Add factors
	for (int32_t y = 0; y < N; ++y)
		for (int32_t x = 0; x < N; ++x)
		{
			SGVector<float64_t> data(2);
			data[0] = CMath::random(0.0, 1.0);
			data[1] = CMath::random(0.0, 1.0);

			SGVector<int32_t> var_index(1);
			var_index[0] = y * N + x;

			CFactor* fac1 = new CFactor(factortype1, var_index, data);
			fg->add_factor(fac1);
		}

	for (int32_t x = 0; x < N; x++)
	{
		for (int32_t y = 0; y < N; y++)
		{
			if (x > 0)
			{
				SGVector<float64_t> data(4);
				float64_t A = CMath::random(0.0, 1.0);//E(0,0)->A
				float64_t C = CMath::random(0.0, 1.0);//E(1,0)->C
				float64_t B = CMath::random(0.0, 1.0);//E(0,1)->B
				float64_t D = CMath::random(0.0, 1.0);//E(1,1)->D

				// Add truncation to ensure submodularity
				truncate_energy(A, B, C, D);

				data[0] = A;
				data[1] = C;
				data[2] = B;
				data[3] = D;

				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x, y, N);
				var_index[1] = grid_to_index(x - 1, y, N);
				CFactor* fac1 = new CFactor(factortype, var_index, data);
				fg->add_factor(fac1);
			}

			if (x == 0 && y > 0)
			{
				SGVector<float64_t> data(4);
				float64_t A = CMath::random(0.0, 1.0);//E(0,0)->A
				float64_t C = CMath::random(0.0, 1.0);//E(1,0)->C
				float64_t B = CMath::random(0.0, 1.0);//E(0,1)->B
				float64_t D = CMath::random(0.0, 1.0);//E(1,1)->D

				// Add truncation to ensure submodularity
				truncate_energy(A, B, C, D);

				data[0] = A;
				data[1] = C;
				data[2] = B;
				data[3] = D;

				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x, y - 1, N);
				var_index[1] = grid_to_index(x, y, N);
				CFactor* fac1 = new CFactor(factortype, var_index, data);
				fg->add_factor(fac1);
			}
		}
	}

	SG_UNREF(factortype);
	SG_UNREF(factortype1);

	// energy table
	fg->compute_energies();

	fg->connect_components();

	// Find minimum energy state by exhaustive search
	SGVector<int> test_var(N * N);
	assignment_expect = SGVector<int>(N * N);
	min_energy_expect = std::numeric_limits<double>::infinity();
	for (int v0 = 0; v0 < 2; ++v0)
	{
		test_var[0] = v0;
		for (int v1 = 0; v1 < 2; ++v1)
		{
			test_var[1] = v1;
			for (int v2 = 0; v2 < 2; ++v2)
			{
				test_var[2] = v2;
				for (int v3 = 0; v3 < 2; ++v3)
				{
					test_var[3] = v3;

					double orig_e = fg->evaluate_energy(test_var);
					if (orig_e < min_energy_expect)
					{
						assignment_expect = test_var.clone();
						min_energy_expect = orig_e;
					}
				}
			}
		}
	}

	return fg;
}

CFactorGraph* CFactorGraphDataGenerator::multi_state_tree_graph()
{
	// ftype
	SGVector<int32_t> card(2);
	card[0] = 3;
	card[1] = 3;
	SGVector<float64_t> w(9);
	w[0] = -0.1; // 0,0
	w[1] = -0.7; // 1,0
	w[2] = -0.9; // 2,0
	w[3] = -0.7; // 0,1
	w[4] = -0.1; // 1,1
	w[5] = -0.0; // 2,1
	w[6] = -0.9; // 0,2
	w[7] = -0.0; // 1,2
	w[8] = -0.1; // 2,2
	int32_t tid = 0;
	CTableFactorType* factortype = new CTableFactorType(tid, card, w);
	SG_REF(factortype);

	SGVector<int32_t> card1(1);
	card1[0] = 3;
	SGVector<float64_t> w1(3);
	w1[0] = -0.1;
	w1[1] = -0.7;
	w1[2] = -0.6;
	int32_t tid1 = 1;
	CTableFactorType* factortype1 = new CTableFactorType(tid1, card1, w1);
	SG_REF(factortype1);

	SGVector<int32_t> card2(1);
	card2[0] = 3;
	SGVector<float64_t> w2(3);
	w2[0] = -0.9;
	w2[1] = -0.1;
	w2[2] = -0.2;
	int32_t tid2 = 2;
	CTableFactorType* factortype2 = new CTableFactorType(tid2, card2, w2);
	SG_REF(factortype2);

	SGVector<int32_t> card3(1);
	card3[0] = 3;
	SGVector<float64_t> w3(3);
	w3[0] = -0.3;
	w3[1] = -0.4;
	w3[2] = -0.5;
	int32_t tid3 = 3;
	CTableFactorType* factortype3 = new CTableFactorType(tid3, card3, w3);
	SG_REF(factortype3);

	// fg
	SGVector<int32_t> vc(3);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 3);
	CFactorGraph* fg = new CFactorGraph(vc);
	SG_REF(fg);

	// add factors
	SGVector<float64_t> data;
	SGVector<int32_t> var_index1(1);
	var_index1[0] = 0;
	CFactor* fac1 = new CFactor(factortype1, var_index1, data);
	fg->add_factor(fac1);

	SGVector<int32_t> var_index2(1);
	var_index2[0] = 1;
	CFactor* fac2 = new CFactor(factortype2, var_index2, data);
	fg->add_factor(fac2);

	SGVector<int32_t> var_index3(1);
	var_index3[0] = 2;
	CFactor* fac3 = new CFactor(factortype3, var_index3, data);
	fg->add_factor(fac3);

	SGVector<int32_t> var_index4(2);
	var_index4[0] = 0;
	var_index4[1] = 1;
	CFactor* fac4 = new CFactor(factortype, var_index4, data);
	fg->add_factor(fac4);

	SGVector<int32_t> var_index5(2);
	var_index5[0] = 1;
	var_index5[1] = 2;
	CFactor* fac5 = new CFactor(factortype, var_index5, data);
	fg->add_factor(fac5);

	// energy table
	fg->compute_energies();
	fg->connect_components();

	SG_UNREF(factortype);
	SG_UNREF(factortype1);
	SG_UNREF(factortype2);
	SG_UNREF(factortype3);

	return fg;
}

/*---------------------------------------------------------------------------------
Test approximate inference algorithms with SOSVM framework
using randomly generated synthetic data
----------------------------------------------------------------------------------*/

void CFactorGraphDataGenerator::generate_data(int32_t len_label, int32_t len_feat, int32_t size_data,
                                SGMatrix<float64_t> &feats, SGMatrix<int32_t> &labels)
{
	ASSERT(size_data > len_label);

	feats = SGMatrix<float64_t>(len_feat, size_data);
	labels = SGMatrix<int32_t>(len_label, size_data);

	for (int32_t k = 0; k < size_data; k++)
	{
		// generate a label vector
		SGVector<int32_t> v_label(len_label);
		v_label.zero();
		int32_t i = k % len_label;
		v_label[i] = 1;

		// generate feature vector
		SGVector<int32_t> random_indices(len_feat);
		random_indices.range_fill();
		CMath::permute(random_indices);
		SGVector<float64_t> v_feat(len_feat);
		v_feat.zero();

		for (int32_t j = 0; j < 3 * (i + 1); j++)
		{
			int32_t r = random_indices[j];
			v_feat[r] = 1;
		}

		for (int32_t f = 0; f < len_feat; f++)
			feats(f, k) = v_feat[f];

		for (int32_t l = 0; l < len_label; l++)
			labels(l, k) = v_label[l];
	}
}

SGMatrix< int32_t > CFactorGraphDataGenerator::get_edges_full(const int32_t num_classes)
{
	// A full-connected graph is defined by a 2-d matrix where
	// each row stores the indecies of a pair of connected nodes
	int32_t num_rows =  num_classes * (num_classes - 1) / 2;
	ASSERT(num_rows > 0);

	SGMatrix< int32_t > mat(num_rows, 2);
	int32_t k = 0;

	for (int32_t i = 0; i < num_classes - 1; i++)
	{
		for (int32_t j = i + 1; j < num_classes; j++)
		{
			mat[num_rows + k] = j;
			mat[k++] = i;
		}
	}

	return mat;
}

void CFactorGraphDataGenerator::build_factor_graph(SGMatrix<float64_t> feats, SGMatrix<int32_t> labels,
                                     SGMatrix< int32_t > edge_list, const DynArray<CTableFactorType*> &v_factor_type,
                                     CFactorGraphFeatures* fg_feats, CFactorGraphLabels* fg_labels)
{
	int32_t num_sample        = labels.num_cols;
	int32_t num_classes       = labels.num_rows;
	int32_t dim               = feats.num_rows;
	int32_t num_edges         = edge_list.num_rows;

	// prepare features and labels in factor graph
	for (int32_t n = 0; n < num_sample; n++)
	{
		SGVector<int32_t> vc(num_classes);
		SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);

		CFactorGraph* fg = new CFactorGraph(vc);

		float64_t* pfeat = feats.get_column_vector(n);
		SGVector<float64_t> feat_i(dim);
		memcpy(feat_i.vector, pfeat, dim * sizeof(float64_t));

		// add unary factors
		for (int32_t u = 0; u < num_classes; u++)
		{
			SGVector<int32_t> var_index_u(1);
			var_index_u[0] = u;
			CFactor* fac_u = new CFactor(v_factor_type[u], var_index_u, feat_i);
			fg->add_factor(fac_u);
		}

		// add pairwised factors
		for (int32_t t = 0; t < num_edges; t++)
		{
			SGVector<float64_t> data_t(1);
			data_t[0] = 1.0;
			SGVector<int32_t> var_index_t = edge_list.get_row_vector(t);
			CFactor* fac_t = new CFactor(v_factor_type[t + num_classes], var_index_t, data_t);
			fg->add_factor(fac_t);
		}

		// add factor graph instance
		fg_feats->add_sample(fg);

		// add label
		int32_t* plabs = labels.get_column_vector(n);
		SGVector<int32_t> states_gt(num_classes);
		memcpy(states_gt.vector, plabs, num_classes * sizeof(int32_t));
		SGVector<float64_t> loss_weights(num_classes);
		SGVector<float64_t>::fill_vector(loss_weights.vector, loss_weights.vlen, 1.0 / num_classes);
		CFactorGraphObservation* fg_obs = new CFactorGraphObservation(states_gt, loss_weights);
		fg_labels->add_label(fg_obs);
	}
}

/** Define factor type
 *
 * @param num_class number of class
 * @param dim dimension of the feature
 * @param num_edges number of edegs
 * @param v_factor_type factor types
 */
void CFactorGraphDataGenerator::define_factor_types(int32_t num_classes, int32_t dim, int32_t num_edges,
                                      DynArray<CTableFactorType*> &v_factor_type)
{
	int32_t tid;
	// we have l = num_classes different weights: w_1, w_2, ..., w_l
	// so we create num_classes different unary factor types
	for (int32_t u = 0; u < num_classes; u++)
	{
		tid = u;
		SGVector<int32_t> card_u(1);
		card_u[0] = 2;
		SGVector<float64_t> w_u(dim * 2);
		w_u.zero();
		CTableFactorType* ft = new CTableFactorType(tid, card_u, w_u);
		v_factor_type.append_element(ft);
	}

	// define factor type: tree edge factor
	// note that each edge is a new type
	for (int32_t t = 0; t < num_edges; t++)
	{
		tid = t + num_classes;
		SGVector<int32_t> card_t(2);
		card_t[0] = 2;
		card_t[1] = 2;
		SGVector<float64_t> w_t(2 * 2);
		w_t.zero();
		CTableFactorType* ft = new CTableFactorType(tid, card_t, w_t);
		v_factor_type.append_element(ft);
	}
}

float64_t CFactorGraphDataGenerator::test_sosvm(EMAPInferType infer_type)
{
	SGMatrix<int32_t> labels_train;
	SGMatrix<float64_t> feats_train;

	// Generate random data
	sg_rand->set_seed(10); // fix the random seed
	generate_data(4, 12, 8, feats_train, labels_train);

	int32_t num_sample_train  = labels_train.num_cols;
	int32_t num_classes       = labels_train.num_rows;
	int32_t dim               = feats_train.num_rows;

	// 1.1 Get edge table
	SGMatrix< int32_t > edge_table = get_edges_full(num_classes);
	int32_t num_edges = edge_table.num_rows;

	// 1.2 Define factor type
	DynArray<CTableFactorType*> v_factor_type;
	define_factor_types(num_classes, dim, num_edges, v_factor_type);

	// 1.3 Prepare features and labels in factor graph
	CFactorGraphFeatures* fg_feats_train = new CFactorGraphFeatures(num_sample_train);
	SG_REF(fg_feats_train);
	CFactorGraphLabels* fg_labels_train = new CFactorGraphLabels(num_sample_train);
	SG_REF(fg_labels_train);

	build_factor_graph(feats_train, labels_train, edge_table, v_factor_type, fg_feats_train, fg_labels_train);

	// 1.4 Create factor graph model
	CFactorGraphModel* model = new CFactorGraphModel(fg_feats_train, fg_labels_train, infer_type, false);
	SG_REF(model);

	// Initialize model parameters
	for (int32_t u = 0; u < num_classes; u++)
		model->add_factor_type(v_factor_type[u]);

	for (int32_t t = 0; t < num_edges; t++)
		model->add_factor_type(v_factor_type[t + num_classes]);

	// 2.1 Create SGD solver
	CStochasticSOSVM* sgd = new CStochasticSOSVM(model, fg_labels_train, true);
	sgd->set_num_iter(150);
	sgd->set_lambda(0.0001);
	SG_REF(sgd);

	// 2.2 Train SGD
	sgd->train();

	// 3.1 Evaluation
	CStructuredLabels* labels_sgd = CLabelsFactory::to_structured(sgd->apply());
	SG_REF(labels_sgd);
	float64_t ave_loss_sgd = 0.0;
	float64_t acc_loss_sgd = 0.0;

	for (int32_t i = 0; i < num_sample_train; ++i)
	{
		CStructuredData* y_pred  = labels_sgd->get_label(i);
		CStructuredData* y_truth = fg_labels_train->get_label(i);
		acc_loss_sgd += model->delta_loss(y_truth, y_pred);

		CFactorGraphObservation* y_t = CFactorGraphObservation::obtain_from_generic(y_truth);
		CFactorGraphObservation* y_p = CFactorGraphObservation::obtain_from_generic(y_pred);

		SGVector<int32_t> s_t = y_t->get_data();
		SGVector<int32_t> s_p = y_p->get_data();

		// training labels are expected to be correcty predicted for this dataset
		//EXPECT_TRUE(s_t.equals(s_p));

		SG_UNREF(y_pred);
		SG_UNREF(y_truth);
	}

	ave_loss_sgd = acc_loss_sgd / static_cast<float64_t>(num_sample_train);

	SG_UNREF(labels_sgd);
	SG_UNREF(sgd);
	SG_UNREF(model);
	SG_UNREF(fg_feats_train);
	SG_UNREF(fg_labels_train);

	return ave_loss_sgd;
}
