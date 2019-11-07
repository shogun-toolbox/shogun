/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Weijie Lin, Jiaolong Xu
 */

#include <shogun/structure/FactorGraphDataGenerator.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;

FactorGraphDataGenerator::FactorGraphDataGenerator(): RandomMixin<SGObject>()
{}

std::shared_ptr<FactorGraph> FactorGraphDataGenerator::simple_chain_graph()
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
	auto ft_pairwise = std::make_shared<TableFactorType>(tid, card, w);


	SGVector<int32_t> card1(1);
	card1[0] = 2;
	SGVector<float64_t> w1(2);
	w1[0] = 0.1;
	w1[1] = 0.7;
	int32_t tid1 = 1;
	auto ft_unary_1 = std::make_shared<TableFactorType>(tid1, card1, w1);


	SGVector<int32_t> card2(1);
	card2[0] = 2;
	SGVector<float64_t> w2(2);
	w2[0] = 0.3;
	w2[1] = 0.6;
	int32_t tid2 = 2;
	auto ft_unary_2 = std::make_shared<TableFactorType>(tid2, card2, w2);


	// fg
	SGVector<int32_t> vc(2);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	auto fg = std::make_shared<FactorGraph>(vc);


	// add factors
	SGVector<float64_t> data;
	SGVector<int32_t> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	auto fac1 = std::make_shared<Factor>(ft_pairwise, var_index, data);
	fg->add_factor(fac1);

	SGVector<int32_t> var_index1(1);
	var_index1[0] = 0;
	auto fac2 = std::make_shared<Factor>(ft_unary_1, var_index1, data);
	fg->add_factor(fac2);

	SGVector<int32_t> var_index2(1);
	var_index2[0] = 1;
	auto fac3 = std::make_shared<Factor>(ft_unary_2, var_index2, data);
	fg->add_factor(fac3);





	// energy table
	fg->compute_energies();

	fg->connect_components();

	return fg;
}

int32_t FactorGraphDataGenerator::grid_to_index(int32_t x, int32_t y, int32_t w)
{
	return x + w * y;
}

void FactorGraphDataGenerator::truncate_energy(float64_t &A, float64_t &B, float64_t &C, float64_t &D)
{
	if (A + D > C + B)
	{
		SG_DEBUG("\nTruncate initialized data to ensure submodularity.");
		float64_t delta = A + D - C - B;
		float64_t subtrA = delta / 3;
		A = A - subtrA;
		C = C + subtrA;
		B = B + (delta - subtrA * 2) + 0.0001; // for numeric issue
	}
}

std::shared_ptr<FactorGraph> FactorGraphDataGenerator::random_chain_graph(SGVector<int> &assignment_expect, float64_t &min_energy_expect, int32_t N)
{
	// ftype
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w;
	int32_t tid = 0;
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	SGVector<int32_t> card1(1);
	card1[0] = 2;
	SGVector<float64_t> w1;
	int32_t tid1 = 1;
	auto factortype1 = std::make_shared<TableFactorType>(tid1, card1, w1);


	// fg
	SGVector<int32_t> vc(N * N);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	auto fg = std::make_shared<FactorGraph>(vc);


	// Add factors
	UniformRealDistribution<float64_t> uniform_real_dist(0.0, 1.0);
	for (int32_t y = 0; y < N; ++y)
		for (int32_t x = 0; x < N; ++x)
		{
			SGVector<float64_t> data(2);
			data[0] = uniform_real_dist(m_prng);
			data[1] = uniform_real_dist(m_prng);

			SGVector<int32_t> var_index(1);
			var_index[0] = y * N + x;

			auto fac1 = std::make_shared<Factor>(factortype1, var_index, data);
			fg->add_factor(fac1);
		}

	for (int32_t x = 0; x < N; x++)
	{
		for (int32_t y = 0; y < N; y++)
		{
			if (x > 0)
			{
				SGVector<float64_t> data(4);
				float64_t A = uniform_real_dist(m_prng);//E(0,0)->A
				float64_t C = uniform_real_dist(m_prng);//E(1,0)->C
				float64_t B = uniform_real_dist(m_prng);//E(0,1)->B
				float64_t D = uniform_real_dist(m_prng);//E(1,1)->D

				// Add truncation to ensure submodularity
				truncate_energy(A, B, C, D);

				data[0] = A;
				data[1] = C;
				data[2] = B;
				data[3] = D;

				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x, y, N);
				var_index[1] = grid_to_index(x - 1, y, N);
				auto fac1 = std::make_shared<Factor>(factortype, var_index, data);
				fg->add_factor(fac1);
			}

			if (x == 0 && y > 0)
			{
				SGVector<float64_t> data(4);
				float64_t A = uniform_real_dist(m_prng);//E(0,0)->A
				float64_t C = uniform_real_dist(m_prng);//E(1,0)->C
				float64_t B = uniform_real_dist(m_prng);//E(0,1)->B
				float64_t D = uniform_real_dist(m_prng);//E(1,1)->D

				// Add truncation to ensure submodularity
				truncate_energy(A, B, C, D);

				data[0] = A;
				data[1] = C;
				data[2] = B;
				data[3] = D;

				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x, y - 1, N);
				var_index[1] = grid_to_index(x, y, N);
				auto fac1 = std::make_shared<Factor>(factortype, var_index, data);
				fg->add_factor(fac1);
			}
		}
	}




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

std::shared_ptr<FactorGraph> FactorGraphDataGenerator::multi_state_tree_graph()
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
	auto factortype = std::make_shared<TableFactorType>(tid, card, w);


	SGVector<int32_t> card1(1);
	card1[0] = 3;
	SGVector<float64_t> w1(3);
	w1[0] = -0.1;
	w1[1] = -0.7;
	w1[2] = -0.6;
	int32_t tid1 = 1;
	auto factortype1 = std::make_shared<TableFactorType>(tid1, card1, w1);


	SGVector<int32_t> card2(1);
	card2[0] = 3;
	SGVector<float64_t> w2(3);
	w2[0] = -0.9;
	w2[1] = -0.1;
	w2[2] = -0.2;
	int32_t tid2 = 2;
	auto factortype2 = std::make_shared<TableFactorType>(tid2, card2, w2);


	SGVector<int32_t> card3(1);
	card3[0] = 3;
	SGVector<float64_t> w3(3);
	w3[0] = -0.3;
	w3[1] = -0.4;
	w3[2] = -0.5;
	int32_t tid3 = 3;
	auto factortype3 = std::make_shared<TableFactorType>(tid3, card3, w3);


	// fg
	SGVector<int32_t> vc(3);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 3);
	auto fg = std::make_shared<FactorGraph>(vc);


	// add factors
	SGVector<float64_t> data;
	SGVector<int32_t> var_index1(1);
	var_index1[0] = 0;
	auto fac1 = std::make_shared<Factor>(factortype1, var_index1, data);
	fg->add_factor(fac1);

	SGVector<int32_t> var_index2(1);
	var_index2[0] = 1;
	auto fac2 = std::make_shared<Factor>(factortype2, var_index2, data);
	fg->add_factor(fac2);

	SGVector<int32_t> var_index3(1);
	var_index3[0] = 2;
	auto fac3 = std::make_shared<Factor>(factortype3, var_index3, data);
	fg->add_factor(fac3);

	SGVector<int32_t> var_index4(2);
	var_index4[0] = 0;
	var_index4[1] = 1;
	auto fac4 = std::make_shared<Factor>(factortype, var_index4, data);
	fg->add_factor(fac4);

	SGVector<int32_t> var_index5(2);
	var_index5[0] = 1;
	var_index5[1] = 2;
	auto fac5 = std::make_shared<Factor>(factortype, var_index5, data);
	fg->add_factor(fac5);

	// energy table
	fg->compute_energies();
	fg->connect_components();






	return fg;
}

/*---------------------------------------------------------------------------------
Test approximate inference algorithms with SOSVM framework
using randomly generated synthetic data
----------------------------------------------------------------------------------*/

void FactorGraphDataGenerator::generate_data(int32_t len_label, int32_t len_feat, int32_t size_data,
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
		random::shuffle(random_indices, m_prng);
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

SGMatrix< int32_t > FactorGraphDataGenerator::get_edges_full(const int32_t num_classes)
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

void FactorGraphDataGenerator::build_factor_graph(SGMatrix<float64_t> feats, SGMatrix<int32_t> labels,
                                     SGMatrix< int32_t > edge_list, const std::vector<std::shared_ptr<TableFactorType>> &v_factor_type,
                                     const std::shared_ptr<FactorGraphFeatures>& fg_feats, const std::shared_ptr<FactorGraphLabels>& fg_labels)
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

		auto fg = std::make_shared<FactorGraph>(vc);

		float64_t* pfeat = feats.get_column_vector(n);
		SGVector<float64_t> feat_i(dim);
		sg_memcpy(feat_i.vector, pfeat, dim * sizeof(float64_t));

		// add unary factors
		for (int32_t u = 0; u < num_classes; u++)
		{
			SGVector<int32_t> var_index_u(1);
			var_index_u[0] = u;
			auto fac_u = std::make_shared<Factor>(v_factor_type[u], var_index_u, feat_i);
			fg->add_factor(fac_u);
		}

		// add pairwised factors
		for (int32_t t = 0; t < num_edges; t++)
		{
			SGVector<float64_t> data_t(1);
			data_t[0] = 1.0;
			SGVector<int32_t> var_index_t = edge_list.get_row_vector(t);
			auto fac_t = std::make_shared<Factor>(v_factor_type[t + num_classes], var_index_t, data_t);
			fg->add_factor(fac_t);
		}

		// add factor graph instance
		fg_feats->add_sample(fg);

		// add label
		int32_t* plabs = labels.get_column_vector(n);
		SGVector<int32_t> states_gt(num_classes);
		sg_memcpy(states_gt.vector, plabs, num_classes * sizeof(int32_t));
		SGVector<float64_t> loss_weights(num_classes);
		SGVector<float64_t>::fill_vector(loss_weights.vector, loss_weights.vlen, 1.0 / num_classes);
		auto fg_obs = std::make_shared<FactorGraphObservation>(states_gt, loss_weights);
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
void FactorGraphDataGenerator::define_factor_types(int32_t num_classes, int32_t dim, int32_t num_edges,
                                      std::vector<std::shared_ptr<TableFactorType>> &v_factor_type)
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
		auto ft = std::make_shared<TableFactorType>(tid, card_u, w_u);
		v_factor_type.push_back(ft);
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
		auto ft = std::make_shared<TableFactorType>(tid, card_t, w_t);
		v_factor_type.push_back(ft);
	}
}

float64_t FactorGraphDataGenerator::test_sosvm(EMAPInferType infer_type)
{
	SGMatrix<int32_t> labels_train;
	SGMatrix<float64_t> feats_train;

	// Generate random data
	generate_data(4, 12, 8, feats_train, labels_train);

	int32_t num_sample_train  = labels_train.num_cols;
	int32_t num_classes       = labels_train.num_rows;
	int32_t dim               = feats_train.num_rows;

	// 1.1 Get edge table
	SGMatrix< int32_t > edge_table = get_edges_full(num_classes);
	int32_t num_edges = edge_table.num_rows;

	// 1.2 Define factor type
	std::vector<std::shared_ptr<TableFactorType>> v_factor_type;
	define_factor_types(num_classes, dim, num_edges, v_factor_type);

	// 1.3 Prepare features and labels in factor graph
	auto fg_feats_train = std::make_shared<FactorGraphFeatures>(num_sample_train);

	auto fg_labels_train = std::make_shared<FactorGraphLabels>(num_sample_train);


	build_factor_graph(feats_train, labels_train, edge_table, v_factor_type, fg_feats_train, fg_labels_train);

	// 1.4 Create factor graph model
	auto model = std::make_shared<FactorGraphModel>(fg_feats_train, fg_labels_train, infer_type, false);


	// Initialize model parameters
	for (int32_t u = 0; u < num_classes; u++)
		model->add_factor_type(v_factor_type[u]);

	for (int32_t t = 0; t < num_edges; t++)
		model->add_factor_type(v_factor_type[t + num_classes]);

	// 2.1 Create SGD solver
	auto sgd = std::make_shared<StochasticSOSVM>(model, fg_labels_train, true);
	sgd->set_num_iter(150);
	sgd->set_lambda(0.0001);


	// 2.2 Train SGD
	sgd->train();

	// 3.1 Evaluation
	auto labels_sgd = sgd->apply()->as<StructuredLabels>();

	float64_t ave_loss_sgd = 0.0;
	float64_t acc_loss_sgd = 0.0;

	for (int32_t i = 0; i < num_sample_train; ++i)
	{
		auto y_pred  = labels_sgd->get_label(i);
		auto y_truth = fg_labels_train->get_label(i);
		acc_loss_sgd += model->delta_loss(y_truth, y_pred);

		auto y_t = y_truth->as<FactorGraphObservation>();
		auto y_p = y_pred->as<FactorGraphObservation>();

		SGVector<int32_t> s_t = y_t->get_data();
		SGVector<int32_t> s_p = y_p->get_data();

		// training labels are expected to be correcty predicted for this dataset
		//EXPECT_TRUE(s_t.equals(s_p));
	}

	ave_loss_sgd = acc_loss_sgd / static_cast<float64_t>(num_sample_train);







	return ave_loss_sgd;
}
