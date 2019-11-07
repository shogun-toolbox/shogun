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

#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <utility>

//#define DEBUG_KNN

using namespace shogun;

KNN::KNN()
: DistanceMachine()
{
	init();
}

KNN::KNN(int32_t k, const std::shared_ptr<Distance>& d, const std::shared_ptr<Labels>& trainlab, KNN_SOLVER knn_solver)
: DistanceMachine()
{
	init();

	m_k=k;

	require(d, "Distance not set.");
	require(trainlab, "Training labels not set.");

	set_distance(d);
	set_labels(trainlab);
	m_train_labels.vlen=trainlab->get_num_labels();
	m_knn_solver=knn_solver;
}

void KNN::init()
{
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
}

KNN::~KNN()
{
}

bool KNN::train_machine(std::shared_ptr<Features> data)
{
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

	return true;
}

SGMatrix<index_t> KNN::nearest_neighbors()
{
	//number of examples to which kNN is applied
	int32_t n=distance->get_num_vec_rhs();

	require(
	    n >= m_k,
	    "K ({}) must not be larger than the number of examples ({}).", m_k, n);

	//distances to train data
	SGVector<float64_t> dists(m_train_labels.vlen);
	//indices to train data
	SGVector<index_t> train_idxs(m_train_labels.vlen);
	//pre-allocation of the nearest neighbors
	SGMatrix<index_t> NN(m_k, n);

	distance->precompute_lhs();
	distance->precompute_rhs();

	//for each test example
	for (auto i : SG_PROGRESS(range(n)))
	{
		COMPUTATION_CONTROLLERS
		//lhs idx 0..num train examples-1 (i.e., all train examples) and rhs idx i
		distances_lhs(dists,0,m_train_labels.vlen-1,i);

		//fill in an array with 0..num train examples-1
		for (int32_t j=0; j<m_train_labels.vlen; j++)
			train_idxs[j]=j;

		//sort the distance vector between test example i and all train examples
		Math::qsort_index(dists.vector, train_idxs.vector, m_train_labels.vlen);

#ifdef DEBUG_KNN
		io::print("\nQuick sort query {}\n", i);
		for (int32_t j=0; j<m_k; j++)
			io::print("{} ", train_idxs[j]);
		io::print("\n");
#endif

		//fill in the output the indices of the nearest neighbors
		for (int32_t j=0; j<m_k; j++)
			NN(j,i) = train_idxs[j];
	}

	distance->reset_precompute();

	return NN;
}

std::shared_ptr<MulticlassLabels> KNN::apply_multiclass(std::shared_ptr<Features> data)
{
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

	init_solver(m_knn_solver);

	return solver->classify_objects(distance, num_lab, train_lab, classes);
}

std::shared_ptr<MulticlassLabels> KNN::classify_NN()
{
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

	return output;
}

SGMatrix<int32_t> KNN::classify_for_multiple_k()
{
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



	return SGMatrix<int32_t>(output,num_lab,m_k);
}

void KNN::init_distance(std::shared_ptr<Features> data)
{
	require(distance, "Distance not set.");
	auto lhs=distance->get_lhs();
	if (!lhs || !lhs->get_num_vectors())
	{
		error("No vectors on left hand side");
	}
	distance->init(lhs, std::move(data));
}

bool KNN::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool KNN::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

void KNN::init_solver(KNN_SOLVER knn_solver)
{
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
}
