/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Giovanni De Toni,
 *          Saurabh Mahindre, Sergey Lisitsyn, Weijie Lin, Heiko Strathmann,
 *          Evgeniy Andreev, Viktor Gal, Bjoern Esser
 */

#include <shogun/base/progress.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/multiclass/tree/KDTree.h>
#include <shogun/multiclass/KNNSolver.h>

#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <utility>
#include <algorithm>
#include <iostream>

//#define DEBUG_KNN

using namespace shogun;

KNN::KNN()
: DistanceMachine()
{
	std::cout<<"entered KNN::KNN() : DistanceMachine()\n";
	init();
	std::cout<<"exiting KNN::KNN() : DistanceMachine()\n";
}

KNN::KNN(int32_t k, const std::shared_ptr<Distance>& d, const std::shared_ptr<Labels>& trainlab, KNN_SOLVER knn_solver)
: DistanceMachine()
{
	std::cout<<"entered KNN::KNN(int32_t k, const std::shared_ptr<Distance>& d, const std::shared_ptr<Labels>& trainlab, KNN_SOLVER knn_solver) : DistanceMachine()\n";
	init();

	m_k=k;

	require(d, "Distance not set.");
	require(trainlab, "Training labels not set.");

	set_distance(d);
	set_labels(trainlab);
	m_train_labels.vlen=trainlab->get_num_labels();
	m_knn_solver=knn_solver;
	std::cout<<"exiting KNN::KNN(int32_t k, const std::shared_ptr<Distance>& d, const std::shared_ptr<Labels>& trainlab, KNN_SOLVER knn_solver) : DistanceMachine()\n";
}

void KNN::init()
{
	std::cout<<"entered KNN::init()\n";
	m_k=3;
	m_q=1.0;
	m_num_classes=0;
	m_leaf_size=1;
	m_knn_solver=KNN_BRUTE;
	solver=NULL;
	m_lsh_l = 0;
	m_lsh_t = 0;

	/* use the method classify_multiply_k to experiment with different values
	 * of k */
	SG_ADD(&m_k, "k", "Parameter k");
	SG_ADD(&m_q, "q", "Parameter q", ParameterProperties::HYPER);
	SG_ADD(&m_num_classes, "num_classes", "Number of classes");
	SG_ADD(&m_leaf_size, "leaf_size", "Leaf size for KDTree");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_knn_solver, "knn_solver", "Algorithm to solve knn",
	    ParameterProperties::NONE,
	    SG_OPTIONS(KNN_BRUTE, KNN_KDTREE, KNN_COVER_TREE, KNN_LSH));
	std::cout<<"exiting KNN::init()\n";
}

KNN::~KNN()
{
}

bool KNN::train_machine(std::shared_ptr<Features> data)
{
	std::cout<<"entered KNN::train_machine(std::shared_ptr<Features> data)\n";
	require(m_labels, "No training labels provided.");
	require(distance, "No training distance provided.");

	if (data)
	{
		require(
		    m_labels->get_num_labels() == data->get_num_vectors(),
		    "Number of training vectors ({}) does not match number of labels "
		    "({})",
		    data->get_num_vectors(), m_labels->get_num_labels());
		distance->init(data, data);
	}

	SGVector<int32_t> lab=multiclass_labels(m_labels)->get_int_labels();
	m_train_labels=lab.clone();
	require(m_train_labels.vlen > 0, "Provided training labels are empty");

	// find minimal and maximal class
	auto min_class = Math::min(m_train_labels.vector, m_train_labels.vlen);
	auto max_class = Math::max(m_train_labels.vector, m_train_labels.vlen);

	linalg::add_scalar(m_train_labels, -min_class);

	m_min_label=min_class;
	m_num_classes=max_class-min_class+1;

	io::info("m_num_classes: {} ({:+d} to {:+d}) num_train: {}", m_num_classes,
			min_class, max_class, m_train_labels.vlen);

	init_solver(m_knn_solver);
	solver->train_KNN(distance);

	std::cout<<"exiting KNN::train_machine(std::shared_ptr<Features> data)\n";
	return true;
}

SGMatrix<index_t> KNN::nearest_neighbors()
{
	std::cout<<"entered KNN::nearest_neighbors()";
	//number of examples to which kNN is applied
	int32_t n=distance->get_num_vec_rhs();

	//distances to train data
	SGVector<float64_t> dists(m_train_labels.vlen);
	//indices to train data
	SGVector<index_t> train_idxs(m_train_labels.vlen);
	//pre-allocation of the nearest neighbors
	SGMatrix<index_t> NN(m_k, n);

	distance->precompute_lhs();
	distance->precompute_rhs();
	std::cout<<"n is "<<n<<'\n';
	std::cout<<"k is "<<m_k<<'\n';
	switch (m_knn_solver)
	{
	case KNN_BRUTE:
	{
		for (auto i : SG_PROGRESS(range(n)))
		{
			COMPUTATION_CONTROLLERS
			distances_lhs(dists,0,m_train_labels.vlen-1,i);

			train_idxs.range_fill(0);

			std::pair<float64_t, index_t> pairt[m_train_labels.vlen];

			
			std::cout<<"i is "<<i<<'\n';
			//std::cout<<" before sorting dists is "<<'\n';
			dists.display_vector("before sorting dists");
			//std::cout<<"before sorting train_idxs is "<<'\n';
			train_idxs.display_vector("before sorting train_idxs");

			// Storing the respective array elements in pairs.
			for (int j = 0; j < m_train_labels.vlen; j++)
			{
				pairt[j].first = dists[j];
				pairt[j].second = train_idxs[j];
			}

			sort(pairt, pairt + m_train_labels.vlen);

			for (int j = 0; j < m_train_labels.vlen; j++)
			{
				dists[j] = pairt[j].first;
				train_idxs[j] = pairt[j].second;
			}

			SG_DEBUG("\nQuick sort query {}", i);
			SG_DEBUG("{}", train_idxs.to_string());

			
			//std::cout<<"after sorting dists is "<<'\n';
			dists.display_vector("after sorting dists");
			//std::cout<<"after sorting train_idxs is "<<'\n';
			train_idxs.display_vector("after sorting train_idxs is");

			//only considering the first k elements
			SGVector<index_t> nearest_k_train_idxs(train_idxs.vector, m_k, false);

			//std::cout<<"nearest_k_train_idxs is "<<'\n';
			nearest_k_train_idxs.display_vector("nearest_k_train_idxs");

			NN.set_column(i, nearest_k_train_idxs);
		}

		distance->reset_precompute();
		break;
	}
	case KNN_KDTREE:
	{
		//auto lhs = distance->get_lhs();
		//auto kd_tree = std::make_shared<KDTree>(m_leaf_size);
		//kd_tree->build_tree(lhs->as<DenseFeatures<float64_t>>());

		//auto query = distance->get_rhs();
		//std::shared_ptr<shogun::KDTree> kd_tree = solver->get_kd_tree();
		//kd_tree->query_knn(query->as<DenseFeatures<float64_t>>(), m_k);
		
		solver->compute_nearest_neighbours();
		NN = solver->get_nearest_neighbours();

		break;
	}
	}
	std::cout<<"exiting KNN::nearest_neighbors()\n";
	NN.display_matrix();
	return NN;
}

std::shared_ptr<MulticlassLabels> KNN::apply_multiclass(std::shared_ptr<Features> data)
{
	std::cout<<"entered KNN::apply_multiclass(std::shared_ptr<Features> data)\n";
	if (data)
		init_distance(data);

	//redirecting to fast (without sorting) classify if k==1
	if (m_k == 1)
		return classify_NN();

	require(m_num_classes > 0, "Machine not trained.");
	require(distance, "Distance not set.");
	require(distance->get_num_vec_rhs(), "No vectors on right hand side.");

	int32_t num_lab=distance->get_num_vec_rhs();
	ASSERT(m_k<=distance->get_num_vec_lhs())

	//labels of the k nearest neighbors
	SGVector<int32_t> train_lab(m_k);

	io::info("{} test examples", num_lab);

	//histogram of classes and returned output
	SGVector<float64_t> classes(m_num_classes);

	//init_solver(m_knn_solver);
	std::cout<<"exiting KNN::apply_multiclass(std::shared_ptr<Features> data)\n";
	return solver->classify_objects(distance, num_lab, train_lab, classes);
}

std::shared_ptr<MulticlassLabels> KNN::classify_NN()
{
	std::cout<<"entered KNN::classify_NN()\n";
	require(distance, "Distance not set.");
	require(m_num_classes > 0, "Machine not trained.");

	int32_t num_lab = distance->get_num_vec_rhs();
	require(num_lab, "No vectors on right hand side");

	auto output = std::make_shared<MulticlassLabels>(num_lab);
	SGVector<float64_t> distances(m_train_labels.vlen);

	io::info("{} test examples", num_lab);

	distance->precompute_lhs();

	// for each test example
	for (auto i : SG_PROGRESS(range(num_lab)))
	{
		COMPUTATION_CONTROLLERS
		// get distances from i-th test example to 0..num_m_train_labels-1 train examples
		distances_lhs(distances,0,m_train_labels.vlen-1,i);
		int32_t j;

		// assuming 0th train examples as nearest to i-th test example
		int32_t out_idx = 0;
		float64_t min_dist = distances.vector[0];

		// searching for nearest neighbor by comparing distances
		for (j=0; j<m_train_labels.vlen; j++)
		{
			if (distances.vector[j]<min_dist)
			{
				min_dist = distances.vector[j];
				out_idx = j;
			}
		}

		// label i-th test example with label of nearest neighbor with out_idx index
		output->set_label(i,m_train_labels.vector[out_idx]+m_min_label);
	}

	distance->reset_precompute();
	std::cout<<"exiting KNN::classify_NN()\n";
	return output;
}

SGMatrix<int32_t> KNN::classify_for_multiple_k()
{
	std::cout<<"entered KNN::classify_for_multiple_k()\n";
	require(distance, "Distance not set.");
	require(m_num_classes > 0, "Machine not trained.");

	int32_t num_lab=distance->get_num_vec_rhs();
	require(num_lab, "No vectors on right hand side");

	require(
	    m_k <= num_lab, "Number of labels ({}) must be at least K ({}).",
	    num_lab, m_k);

	//working buffer of m_train_labels
	SGVector<int32_t> train_lab(m_k);

	//histogram of classes and returned output
	SGVector<int32_t> classes(m_num_classes);

	io::info("{} test examples", num_lab);

	init_solver(m_knn_solver);

	SGVector<int32_t> output = solver->classify_objects_k(distance, num_lab, train_lab, classes);


	std::cout<<"exiting KNN::classify_for_multiple_k()\n";
	return SGMatrix<int32_t>(output,num_lab,m_k);
}

void KNN::init_distance(std::shared_ptr<Features> data)
{
	std::cout<<"entered KNN::init_distance(std::shared_ptr<Features> data)\n";
	require(distance, "Distance not set.");
	auto lhs=distance->get_lhs();
	if (!lhs || !lhs->get_num_vectors())
	{
		error("No vectors on left hand side");
	}
	distance->init(lhs, std::move(data));
	std::cout<<"exiting KNN::init_distance(std::shared_ptr<Features> data)\n";
}

bool KNN::load(FILE* srcfile)
{
	std::cout<<"entered KNN::load(FILE* srcfile)\n";
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	std::cout<<"exiting KNN::load(FILE* srcfile)\n";
	return false;
}

bool KNN::save(FILE* dstfile)
{
	std::cout<<"entered KNN::save(FILE* dstfile)\n";
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	std::cout<<"exiting KNN::save(FILE* dstfile)\n";
	return false;
}

void KNN::init_solver(KNN_SOLVER knn_solver)
{
	std::cout<<"entered KNN::init_solver(KNN_SOLVER knn_solver)\n";
	switch (knn_solver)
	{
	case KNN_BRUTE:
	{
		SGMatrix<index_t> NN = nearest_neighbors();
		solver = std::make_shared<BruteKNNSolver>(m_k, m_q, m_num_classes, m_min_label, m_train_labels, NN);

		break;
	}
	case KNN_KDTREE:
	{
		solver = std::make_shared<KDTREEKNNSolver>(m_k, m_q, m_num_classes, m_min_label, m_train_labels, m_leaf_size);

		break;
	}
	case KNN_COVER_TREE:
	{
#ifdef USE_GPL_SHOGUN
		solver = std::make_shared<CoverTreeKNNSolver>(m_k, m_q, m_num_classes, m_min_label, m_train_labels);

		break;
#else
		gpl_only(SOURCE_LOCATION);
#endif // USE_GPL_SHOGUN
	}
	case KNN_LSH:
	{
		solver = std::make_shared<LSHKNNSolver>(m_k, m_q, m_num_classes, m_min_label, m_train_labels, m_lsh_l, m_lsh_t);

		break;
	}
	}
	std::cout<<"exiting KNN::init_solver(KNN_SOLVER knn_solver)\n";
}
