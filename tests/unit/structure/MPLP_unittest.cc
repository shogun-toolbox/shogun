#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/base/DynArray.h>
#include <shogun/base/init.h>
#include <shogun/io/SGIO.h>

#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/FactorGraphModel.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/Factor.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/structure/MPLP.h>

#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/labels/FactorGraphLabels.h>

#include <gtest/gtest.h>

using namespace shogun;

inline int grid_to_index(int32_t x, int32_t y, int32_t w = 10)
{
	return x + w * y;
}

// Test MPLP inference for a simple two nodes chain structure graph
TEST(MPLP, mplp_chain)
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
	CTableFactorType* factortype = new CTableFactorType(tid, card, w);
	SG_REF(factortype);

	SGVector<int32_t> card1(1);
	card1[0] = 2;
	SGVector<float64_t> w1(2);
	w1[0] = 0.1;
	w1[1] = 0.7;
	int32_t tid1 = 1;
	CTableFactorType* factortype1 = new CTableFactorType(tid1, card1, w1);
	SG_REF(factortype1);

	SGVector<int32_t> card2(1);
	card2[0] = 2;
	SGVector<float64_t> w2(2);
	w2[0] = 0.3;
	w2[1] = 0.6;
	int32_t tid2 = 2;
	CTableFactorType* factortype2 = new CTableFactorType(tid2, card2, w2);
	SG_REF(factortype2);

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
	CFactor* fac1 = new CFactor(factortype, var_index, data);
	fg->add_factor(fac1);

	SGVector<int32_t> var_index1(1);
	var_index1[0] = 0;
	CFactor* fac2 = new CFactor(factortype1, var_index1, data);
	fg->add_factor(fac2);

	SGVector<int32_t> var_index2(1);
	var_index2[0] = 1;
	CFactor* fac3 = new CFactor(factortype2, var_index2, data);
	fg->add_factor(fac3);

	SG_UNREF(factortype);
	SG_UNREF(factortype1);
	SG_UNREF(factortype2);

	// energy table
	fg->compute_energies();

	fg->connect_components();

	EXPECT_TRUE(fg->is_acyclic_graph());
	EXPECT_TRUE(fg->is_connected_graph());
	EXPECT_TRUE(fg->is_tree_graph());
	EXPECT_EQ(fg->get_num_edges(), 4);

	CMAPInference infer_met(fg, LP_RELAXATION);
	infer_met.inference();

	CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();
	SGVector<int32_t> assignment = fg_observ->get_data();
	SG_UNREF(fg_observ);

	EXPECT_NEAR(0.4, infer_met.get_energy(), 1E-10);

	SG_UNREF(fg);
}

// Test MPLP inference for a four nodes chain graph
// potentials are randomly generated
TEST(MPLP, mplp_random)
{
	CMath::init_random(17);
	int N = 2;
	for (int32_t rani = 0; rani < 10; rani++)
	{
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

		EXPECT_TRUE(fg->is_acyclic_graph());
		EXPECT_TRUE(fg->is_connected_graph());
		EXPECT_TRUE(fg->is_tree_graph());
		EXPECT_EQ(fg->get_num_edges(), 10);

		CMAPInference infer_met(fg, LP_RELAXATION);
		infer_met.inference();

		CFactorGraphObservation* fg_observ = infer_met.get_structured_outputs();
		SGVector<int32_t> assignment = fg_observ->get_data();
		SG_UNREF(fg_observ);

		// Find minimum energy state by exhaustive search
		SGVector<int> test_var(N * N);
		SGVector<int> min_var(N * N);
		double min_var_energy = std::numeric_limits<double>::infinity();
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
						if (orig_e < min_var_energy)
						{
							min_var = test_var.clone();
							min_var_energy = orig_e;
						}
					}
				}
			}
		}

		for (int32_t si = 0; si < N * N; si++)
		{
			EXPECT_EQ(assignment[si], min_var[si]);
		}

		EXPECT_NEAR(infer_met.get_energy(), min_var_energy, 1E-10);

		SG_UNREF(fg);
	}
}

/*---------------------------------------------------------------------------------
Test MPLP with SOSVM framework using randomly generated synthetic data
----------------------------------------------------------------------------------*/

/** Generate random data following [1]:
 * Each example has exactly one label on.
 * Each label has 40 related binary features.
 * For an example, if label i is on, 4i randomly chosen features are set to 1
 *
 * [1] Finley, Thomas, and Thorsten Joachims.
 * "Training structural SVMs when exact inference is intractable."
 * Proceedings of the 25th international conference on Machine learning. ACM, 2008.
 *
 * @param len_label label length (10)
 * @param len_feat feature length (40)
 * @param size_data training data size (50)
 * @param feats generated feature matrix
 * @param labels generated label matrix
 */
inline void generate_data(int32_t len_label, int32_t len_feat, int32_t size_data,
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
		SGVector<int32_t> random_indeces(len_feat);
		random_indeces.randperm();
		SGVector<float64_t> v_feat(len_feat);
		v_feat.zero();

		for (int32_t j = 0; j < 4 * (i + 1); j++)
		{
			int32_t r = random_indeces[j];
			v_feat[r] = 1;
		}

		for (int32_t f = 0; f < len_feat; f++)
		{
			feats(f, k) = v_feat[f];
		}

		for (int32_t l = 0; l < len_label; l++)
		{
			labels(l, k) = v_label[l];
		}
	}
}

inline SGMatrix< int32_t > get_edges_full(const int32_t num_classes)
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

inline void build_factor_graph(SGMatrix<float64_t> feats, SGMatrix<int32_t> labels,
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

inline void define_factor_types(int32_t num_classes, int32_t dim, int32_t num_edges, DynArray<CTableFactorType*> &v_factor_type)
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

TEST(MPLP, mplp_sosvm)
{
	SGMatrix<int32_t> labels_train;
	SGMatrix<float64_t> feats_train;

	// Generate random data
	sg_rand->set_seed(10); // fix the random seed
	generate_data(5, 20, 20, feats_train, labels_train);

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
	CFactorGraphModel* model = new CFactorGraphModel(fg_feats_train, fg_labels_train, LP_RELAXATION, false);
	SG_REF(model);

	// Initialize model parameters
	for (int32_t u = 0; u < num_classes; u++)
	{
		model->add_factor_type(v_factor_type[u]);
	}

	for (int32_t t = 0; t < num_edges; t++)
	{
		model->add_factor_type(v_factor_type[t + num_classes]);
	}

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
		EXPECT_TRUE(s_t.equals(s_p));

		SG_UNREF(y_pred);
		SG_UNREF(y_truth);
	}

	ave_loss_sgd = acc_loss_sgd / static_cast<float64_t>(num_sample_train);

	// training error is expected to be close to 0
	EXPECT_NEAR(ave_loss_sgd, 0, 0.1);

	SG_UNREF(labels_sgd);
	SG_UNREF(sgd);
	SG_UNREF(model);
	SG_UNREF(fg_feats_train);
	SG_UNREF(fg_labels_train);
}
