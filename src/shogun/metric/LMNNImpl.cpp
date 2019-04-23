/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Heiko Strathmann, Viktor Gal
 */

#include <shogun/metric/LMNNImpl.h>

#include <algorithm>
#include <iterator>
#include <unordered_map>

#include <shogun/lib/View.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/preprocessor/PCA.h>
#include <shogun/preprocessor/PruneVarSubMean.h>

using namespace shogun;

CImpostorNode::CImpostorNode(index_t ex, index_t tar, index_t imp)
: example(ex), target(tar), impostor(imp)
{
}

bool CImpostorNode::operator<(const CImpostorNode& rhs) const
{
	if (example == rhs.example)
	{
		if (target == rhs.target)
			return impostor < rhs.impostor;
		else
			return target < rhs.target;
	}
	else
		return example < rhs.example;
}

void LMNNImpl::check_training_setup(
    std::shared_ptr<Features> features, std::shared_ptr<Labels> labels, SGMatrix<float64_t>& init_transform,
    int32_t k)
{
	REQUIRE(features->has_property(FP_DOT),
			"LMNN can only be applied to features that support dot products\n")
	REQUIRE(labels->get_label_type()==LT_MULTICLASS,
			"LMNN supports only MulticlassLabels\n")
	REQUIRE(labels->get_num_labels()==features->get_num_vectors(),
			"The number of feature vectors must be equal to the number of labels\n")
	//FIXME this requirement should be dropped in the future
	REQUIRE(features->get_feature_class()==C_DENSE,
			"Currently, LMNN supports only DenseFeatures\n")

	// cast is safe, we ensure above that features are dense
	auto x = features->as<DenseFeatures<float64_t>>();

	/// Initialize, if necessary, the initial transform
	if (init_transform.num_rows==0)
		init_transform = LMNNImpl::compute_pca_transform(x);

	REQUIRE(init_transform.num_rows==x->get_num_features() &&
			init_transform.num_rows==init_transform.num_cols,
			"The initial transform must be a square matrix of size equal to the "
			"number of features\n")

	check_maximum_k(labels, k);
}

void LMNNImpl::check_maximum_k(std::shared_ptr<Labels> labels, int32_t k)
{
	auto y = multiclass_labels(labels);
	SGVector<int32_t> int_labels = y->get_int_labels();

	// back-up initial values because they will be overwritten by unique
	std::vector<int32_t> int_labels_vec;
	std::copy(
	    int_labels.begin(), int_labels.end(),
	    std::back_inserter(int_labels_vec));

	std::sort(int_labels.begin(), int_labels.end());
	auto unique_end = std::unique(int_labels.begin(), int_labels.end());

	std::vector<int32_t> labels_histogram(
	    std::distance(int_labels.begin(), unique_end), 0);

	std::unordered_map<int32_t, int32_t> label_to_index;
	{
		int32_t next_index = 0;
		for (auto begin = int_labels.begin(); begin != unique_end; begin++)
			label_to_index.insert({*begin, next_index++});
	}

	for (auto int_label : int_labels_vec)
	{
		labels_histogram[label_to_index[int_label]] += 1;
	}

	int32_t min_num_examples =
	    *std::min_element(labels_histogram.begin(), labels_histogram.end());
	REQUIRE(
	    min_num_examples > k,
	    "The minimum number of examples of any class (%d) must be larger "
	    "than k (%d); it must be at least k+1 because any example needs "
	    "k *other* neighbors of the same class.",
	    min_num_examples, k)
}

SGMatrix<index_t> LMNNImpl::find_target_nn(std::shared_ptr<DenseFeatures<float64_t>> x,
		std::shared_ptr<MulticlassLabels> y, int32_t k)
{
	SG_SDEBUG("Entering LMNNImpl::find_target_nn().\n")

	// get the number of features
	int32_t d = x->get_num_features();
	SGMatrix<index_t> target_neighbors(k, x->get_num_vectors());
	SGVector<float64_t> unique_labels = y->get_unique_labels();
	SGVector<index_t> idxsmap(x->get_num_vectors());

	for (index_t i = 0; i < unique_labels.vlen; ++i)
	{
		int32_t slice_size = 0;
		for (int32_t j = 0; j < y->get_num_labels(); ++j)
		{
			if (y->get_label(j)==unique_labels[i])
			{
				idxsmap[slice_size] = j;
				++slice_size;
			}
		}

		SGMatrix<float64_t> slice_mat(d, slice_size);
		for (int32_t j = 0; j < slice_size; ++j)
			slice_mat.set_column(j, x->get_feature_vector(idxsmap[j]));


		//FIXME the labels are not actually necessary to get the nearest neighbors, the
		//features suffice. The labels are needed when we want to classify.
		SGVector<float64_t> labels_vec(slice_size);
		labels_vec.set_const(unique_labels[i]);

		auto features_slice = std::make_shared<DenseFeatures<float64_t>>(slice_mat);
		auto labels_slice = std::make_shared<MulticlassLabels>(labels_vec);

		auto knn = std::make_shared<KNN>(k+1, std::make_shared<EuclideanDistance>(features_slice, features_slice), labels_slice);
		SGMatrix<int32_t> target_slice = knn->nearest_neighbors();
		// sanity check
		ASSERT(target_slice.num_rows==k+1 && target_slice.num_cols==slice_size)

		for (index_t j = 0; j < target_slice.num_cols; ++j)
		{
			for (index_t l = 1; l < target_slice.num_rows; ++l)
				target_neighbors(l-1, idxsmap[j]) = idxsmap[ target_slice(l,j) ];
		}

		// clean up knn

	}

	SG_SDEBUG("Leaving LMNNImpl::find_target_nn().\n")

	return target_neighbors;
}

SGMatrix<float64_t> LMNNImpl::sum_outer_products(
    std::shared_ptr<DenseFeatures<float64_t>> x, const SGMatrix<index_t>& target_nn)
{
	// get the number of features
	int32_t d = x->get_num_features();
	// initialize the sum of outer products (sop)
	SGMatrix<float64_t> sop(d, d);

	auto X = x->get_feature_matrix();

	// sum the outer products stored in C using the indices specified in target_nn
	for (index_t i = 0; i < target_nn.num_cols; ++i)
	{
		for (index_t j = 0; j < target_nn.num_rows; ++j)
		{
			auto dx = linalg::add(
			    X.get_column(i), X.get_column(target_nn(j, i)), 1.0, -1.0);
			linalg::rank_update(sop, dx);
		}
	}

	return sop;
}

ImpostorsSetType LMNNImpl::find_impostors(
    std::shared_ptr<DenseFeatures<float64_t>> x, std::shared_ptr<MulticlassLabels> y,
    const SGMatrix<float64_t>& L, const SGMatrix<index_t>& target_nn,
    const int32_t iter, const int32_t correction)
{
	SG_SDEBUG("Entering LMNNImpl::find_impostors().\n")

	// get the number of neighbors
	int32_t k = target_nn.num_rows;

	auto X = x->get_feature_matrix();
	// transform the feature vectors
	auto LX = linalg::matrix_prod(L, X);

	// compute square distances plus margin from examples to target neighbors
	auto sqdists = LMNNImpl::compute_sqdists(LX, target_nn);

	// initialize impostors set
	ImpostorsSetType N;
	// exact impostors set shared between calls to this method
	static ImpostorsSetType Nexact;

	// impostors search
	REQUIRE(correction>0, "The number of iterations between exact updates of the "
			"impostors set must be greater than 0\n")
	if ((iter % correction)==0)
	{
		Nexact = LMNNImpl::find_impostors_exact(
		    SGMatrix<float64_t>(LX), sqdists, y, target_nn, k);
		N = Nexact;
	}
	else
	{
		N = LMNNImpl::find_impostors_approx(
		    SGMatrix<float64_t>(LX), sqdists, Nexact, target_nn);
	}

	SG_SDEBUG("Leaving LMNNImpl::find_impostors().\n")

	return N;
}

void LMNNImpl::update_gradient(
    std::shared_ptr<DenseFeatures<float64_t>> x, SGMatrix<float64_t>& G,
    const ImpostorsSetType& Nc, const ImpostorsSetType& Np,
    float64_t regularization)
{
	// compute the difference sets
	ImpostorsSetType Np_Nc, Nc_Np;
	set_difference(Np.begin(), Np.end(), Nc.begin(), Nc.end(), inserter(Np_Nc, Np_Nc.begin()));
	set_difference(Nc.begin(), Nc.end(), Np.begin(), Np.end(), inserter(Nc_Np, Nc_Np.begin()));

	auto X = x->get_feature_matrix();

	// remove the gradient contributions of the impostors that were in the previous
	// set but disappeared in the current
	for (ImpostorsSetType::iterator it = Np_Nc.begin(); it != Np_Nc.end(); ++it)
	{
		// G -= regularization*(dx1*dx1' - dx2*dx2');
		auto dx1 = linalg::add(
		    X.get_column(it->example), X.get_column(it->target), 1.0, -1.0);
		auto dx2 = linalg::add(
		    X.get_column(it->example), X.get_column(it->impostor), 1.0, -1.0);
		linalg::rank_update(G, dx1, -regularization);
		linalg::rank_update(G, dx2, regularization);
	}

	// add the gradient contributions of the new impostors
	for (ImpostorsSetType::iterator it = Nc_Np.begin(); it != Nc_Np.end(); ++it)
	{
		// G += regularization*(dx1*dx1' - dx2*dx2');
		auto dx1 = linalg::add(
		    X.get_column(it->example), X.get_column(it->target), 1.0, -1.0);
		auto dx2 = linalg::add(
		    X.get_column(it->example), X.get_column(it->impostor), 1.0, -1.0);
		linalg::rank_update(G, dx1, regularization);
		linalg::rank_update(G, dx2, -regularization);
	}
}

void LMNNImpl::gradient_step(
    SGMatrix<float64_t>& L, const SGMatrix<float64_t>& G, float64_t stepsize,
    bool diagonal)
{
	if (diagonal)
	{
		// compute M as the square of L
		auto M = linalg::matrix_prod(L, L, true, false);
		// do step in M along the gradient direction
		linalg::add(M, G, M, 1.0, -stepsize);
		// keep only the elements in the diagonal of M
		auto m = M.get_diagonal_vector();
		for (auto i : range(m.vlen))
			m[i] = m[i] > 0 ? std::sqrt(m[i]) : 0.0;

		// return to representation in L
		SGMatrix<float64_t>::create_diagonal_matrix(L.matrix, m.vector, m.vlen);
	}
	else
	{
		// do step in L along the gradient direction (no need to project M then)
		// L -= stepsize*(2*L*G)
		linalg::dgemm(-2.0 * stepsize, L, G, false, false, 1.0, L);
	}
}

void LMNNImpl::correct_stepsize(
    float64_t& stepsize, const SGVector<float64_t> obj, const int32_t iter)
{
	if (iter > 0)
	{
		// Difference between current and previous objective
		float64_t delta = obj[iter] - obj[iter-1];

		if (delta > 0)
		{
			// The objective has increased, we have probably jumped over the optimum,
			// thus, decrease the step size
			stepsize *= 0.5;
		}
		else
		{
			// The objective has decreased, we are in the right direction,
			// increase the step size
			stepsize *= 1.01;
		}
	}
}

bool LMNNImpl::check_termination(
    float64_t stepsize, const SGVector<float64_t> obj, int32_t iter,
    int32_t maxiter, float64_t stepsize_threshold, float64_t obj_threshold)
{
	if (iter >= maxiter-1)
	{
		SG_SWARNING("Maximum number of iterations reached before convergence.");
		return true;
	}

	if (stepsize < stepsize_threshold)
	{
		SG_SDEBUG("Step size too small to make more progress. Convergence reached.\n");
		return true;
	}

	if (iter >= 10)
	{
		obj_threshold *= obj[iter - 1];
		for (int32_t i = 0; i < 3; ++i)
		{
			if (Math::abs(obj[iter-i]-obj[iter-i-1]) >= obj_threshold)
				return false;
		}

		SG_SDEBUG("No more progress in the objective. Convergence reached.\n");
		return true;
	}

	// For the rest of the cases, do not stop LMNN.
	return false;
}

SGMatrix<float64_t> LMNNImpl::compute_pca_transform(std::shared_ptr<DenseFeatures<float64_t>> features)
{
	SG_SDEBUG("Initializing LMNN transform using PCA.\n");

	// Substract the mean of the features
	// Create clone of the features to keep the input features unmodified
	auto mean_substractor =
	    std::make_shared<PruneVarSubMean>(false); // false to avoid variance normalization
	mean_substractor->fit(features);
	auto centered_feats = mean_substractor->transform(features)->as<DenseFeatures<float64_t>>();

	// Obtain the linear transform applying PCA
	auto pca = std::make_shared<PCA>();
	pca->set_target_dim(centered_feats->get_num_features());
	pca->fit(centered_feats);
	SGMatrix<float64_t> pca_transform = pca->get_transformation_matrix();

	return pca_transform;
}

SGMatrix<float64_t> LMNNImpl::compute_sqdists(
    const SGMatrix<float64_t>& LX, const SGMatrix<index_t>& target_nn)
{
	// get the number of examples
	ASSERT(LX.num_cols == target_nn.num_cols)
	int32_t n = LX.num_cols;
	// get the number of neighbors
	int32_t k = target_nn.num_rows;

	/// compute square distances to target neighbors plus margin

	// create Shogun features from LX to later apply subset
	auto lx = std::make_shared<DenseFeatures<float64_t>>(LX);

	// initialize distances
	SGMatrix<float64_t> sqdists(k, n);

	for (int32_t i = 0; i < k; ++i)
	{
		//FIXME avoid copying the rows of target_nn and access them directly. Maybe
		//find_target_nn should be changed to return the output transposed wrt how it is
		//done atm.
		SGVector<index_t> subset_vec = target_nn.get_row_vector(i);
		auto lx_subset = view(lx, subset_vec);
		// after the subset, there are still n columns, i.e. the subset is used to
		// modify the order of the columns in x according to the target neighbors
		auto diff = linalg::add(LX, lx_subset->get_feature_matrix(), 1.0, -1.0);
		auto sum = linalg::colwise_sum(linalg::element_prod(diff, diff));
		for (int j = 0; j < sum.vlen; j++)
			sqdists(i, j) = sum[j] + 1;
	}

	// clean up features used to apply subset


	return sqdists;
}

ImpostorsSetType LMNNImpl::find_impostors_exact(
    const SGMatrix<float64_t>& LX, const SGMatrix<float64_t>& sqdists,
    std::shared_ptr<MulticlassLabels> y, const SGMatrix<index_t>& target_nn, int32_t k)
{
	SG_SDEBUG("Entering LMNNImpl::find_impostors_exact().\n")

	// initialize empty impostors set
	ImpostorsSetType N = ImpostorsSetType();

	// create Shogun features from LX to later apply subset
	auto lx = std::make_shared<DenseFeatures<float64_t>>(LX);

	// get a vector with unique label values
	SGVector<float64_t> unique = y->get_unique_labels();

	// for each label except from the largest one
	for (index_t i = 0; i < unique.vlen-1; ++i)
	{
		// get the indices of the examples labelled as unique[i]
		std::vector<index_t> iidxs = LMNNImpl::get_examples_label(y,unique[i]);
		// get the indices of the examples that have a larger label value, so that
		// pairwise distances are computed once
		std::vector<index_t> gtidxs = LMNNImpl::get_examples_gtlabel(y,unique[i]);

		// get distance with features indexed by iidxs and gtidxs separated
		auto euclidean = LMNNImpl::setup_distance(lx,iidxs,gtidxs);
		euclidean->set_disable_sqrt(true);

		for (int32_t j = 0; j < k; ++j)
		{
			for (std::size_t ii = 0; ii < iidxs.size(); ++ii)
			{
				for (std::size_t jj = 0; jj < gtidxs.size(); ++jj)
				{
					// FIXME study if using upper bounded distances can be an improvement
					float64_t distance = euclidean->distance(ii,jj);

					if (distance <= sqdists(j,iidxs[ii]))
						N.insert( CImpostorNode(iidxs[ii], target_nn(j,iidxs[ii]), gtidxs[jj]) );

					if (distance <= sqdists(j,gtidxs[jj]))
						N.insert( CImpostorNode(gtidxs[jj], target_nn(j,gtidxs[jj]), iidxs[ii]) );
				}
			}
		}


	}



	SG_SDEBUG("Leaving LMNNImpl::find_impostors_exact().\n")

	return N;
}

ImpostorsSetType LMNNImpl::find_impostors_approx(
    const SGMatrix<float64_t>& LX, const SGMatrix<float64_t>& sqdists,
    const ImpostorsSetType& Nexact, const SGMatrix<index_t>& target_nn)
{
	SG_SDEBUG("Entering LMNNImpl::find_impostors_approx().\n")

	// initialize empty impostors set
	ImpostorsSetType N = ImpostorsSetType();

	// compute square distances from examples to impostors
	SGVector<float64_t> impostors_sqdists = LMNNImpl::compute_impostors_sqdists(LX,Nexact);

	// find in the exact set of impostors computed last, the triplets that remain impostors
	index_t i = 0;
	for (ImpostorsSetType::iterator it = Nexact.begin(); it != Nexact.end(); ++it)
	{
		// find in target_nn(:,it->example) the position of the target neighbor it->target
		index_t target_idx = 0;
		while (target_nn(target_idx, it->example)!=it->target && target_idx<target_nn.num_rows)
			++target_idx;

		REQUIRE(target_idx<target_nn.num_rows, "The index of the target neighbour in the "
				"impostors set was not found in the target neighbours matrix. "
				"There must be a bug in find_impostors_exact.\n")

		if ( impostors_sqdists[i++] <= sqdists(target_idx, it->example) )
			N.insert(*it);
	}

	SG_SDEBUG("Leaving LMNNImpl::find_impostors_approx().\n")

	return N;
}

SGVector<float64_t> LMNNImpl::compute_impostors_sqdists(
    const SGMatrix<float64_t>& LX, const ImpostorsSetType& Nexact)
{
	// get the number of impostors
	size_t num_impostors = Nexact.size();

	/// compute square distances to impostors

	// create Shogun features from LX and distance
	auto lx = std::make_shared<DenseFeatures<float64_t>>(LX);
	auto euclidean = std::make_shared<EuclideanDistance>(lx,lx);
	euclidean->set_disable_sqrt(true);

	// initialize vector of square distances
	SGVector<float64_t> sqdists(num_impostors);
	// compute square distances
	index_t i = 0;
	for (ImpostorsSetType::iterator it = Nexact.begin(); it != Nexact.end(); ++it)
		sqdists[i++] = euclidean->distance(it->example,it->impostor);

	// clean up distance


	return sqdists;
}

std::vector<index_t> LMNNImpl::get_examples_label(std::shared_ptr<MulticlassLabels> y,
		float64_t yi)
{
	// indices of the examples with label equal to yi
	std::vector<index_t> idxs;

	for (index_t i = 0; i < index_t(y->get_num_labels()); ++i)
	{
		if (y->get_label(i) == yi)
			idxs.push_back(i);
	}

	return idxs;
}

std::vector<index_t> LMNNImpl::get_examples_gtlabel(std::shared_ptr<MulticlassLabels> y,
		float64_t yi)
{
	// indices of the examples with label equal greater than yi
	std::vector<index_t> idxs;

	for (index_t i = 0; i < index_t(y->get_num_labels()); ++i)
	{
		if (y->get_label(i) > yi)
			idxs.push_back(i);
	}

	return idxs;
}

std::shared_ptr<EuclideanDistance> LMNNImpl::setup_distance(std::shared_ptr<DenseFeatures<float64_t>> x,
		std::vector<index_t>& a, std::vector<index_t>& b)
{
	// create new features only containing examples whose indices are in a
	auto afeats = view(x, SGVector<index_t>(a.data(), a.size(), false));

	// create new features only containing examples whose indices are in b
	auto bfeats = view(x, SGVector<index_t>(b.data(), b.size(), false));

	// create and return distance
	return std::make_shared<EuclideanDistance>(afeats,bfeats);
}

