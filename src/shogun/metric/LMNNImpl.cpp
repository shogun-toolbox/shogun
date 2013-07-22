/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2013 Fernando J. Iglesias Garcia
 */

#ifdef HAVE_EIGEN3

#include <shogun/metric/LMNNImpl.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>

#include <iterator>

/// useful shorthands to perform operations with Eigen matrices

// trace of the product of two matrices computed fast using trace(A*B)=sum(A.*B')
#define	TRACE(A,B)		(((A).array()*(B).transpose().array()).sum())
// column-wise sum of the squared elements of a matrix
#define SUMSQCOLS(A)	((A).array().square().colwise().sum())

using namespace shogun;
using namespace Eigen;

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

void CLMNNImpl::check_training_setup(CFeatures* features, CLabels* labels,
		const SGMatrix<float64_t> init_transform)
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
	CDenseFeatures<float64_t>* x = static_cast<CDenseFeatures<float64_t>*>(features);

	REQUIRE(init_transform.num_rows==x->get_num_features() &&
			init_transform.num_rows==init_transform.num_cols,
			"The initial transform must be a square matrix of size equal to the "
			"number of features\n")
}

SGMatrix<index_t> CLMNNImpl::find_target_nn(CDenseFeatures<float64_t>* x,
		CMulticlassLabels* y, int32_t k)
{
	SG_SDEBUG("Entering CLMNNImpl::find_target_nn().\n")

	// get the number of features
	int32_t d = x->get_num_features();
	SGMatrix<index_t> target_neighbors(k, x->get_num_vectors());
	SGVector<float64_t> unique_labels = y->get_unique_labels();
	CDenseFeatures<float64_t>* features_slice = new CDenseFeatures<float64_t>();
	CMulticlassLabels* labels_slice = new CMulticlassLabels();
	SGVector<index_t> idxsmap(x->get_num_vectors());

	// increase ref count because a KNN instance will be created for each slice, and it
	// decreses the ref count of its labels and features when destroyed
	SG_REF(features_slice)
	SG_REF(labels_slice)

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

		MatrixXd slice_mat(d, slice_size);
		for (int32_t j = 0; j < slice_size; ++j)
			slice_mat.col(j) = Map<const VectorXd>(x->get_feature_vector(idxsmap[j]).vector, d);

		features_slice->set_feature_matrix(SGMatrix<float64_t>(slice_mat.data(), d, slice_size, false));

		//FIXME the labels are not actually necessary to get the nearest neighbors, the
		//features suffice. The labels are needed when we want to classify.
		SGVector<float64_t> labels_vec(slice_size);
		labels_vec.set_const(unique_labels[i]);
		labels_slice->set_labels(labels_vec);

		CKNN* knn = new CKNN(k+1, new CEuclideanDistance(features_slice, features_slice), labels_slice);
		SGMatrix<int32_t> target_slice = knn->nearest_neighbors();
		// sanity check
		ASSERT(target_slice.num_rows==k+1 && target_slice.num_cols==slice_size)

		for (index_t j = 0; j < target_slice.num_cols; ++j)
		{
			for (index_t l = 1; l < target_slice.num_rows; ++l)
				target_neighbors(l-1, idxsmap[j]) = idxsmap[ target_slice(l,j) ];
		}

		// clean up knn
		SG_UNREF(knn)
	}

	// clean up features and labels
	SG_UNREF(features_slice)
	SG_UNREF(labels_slice)

	SG_SDEBUG("Leaving CLMNNImpl::find_target_nn().\n")

	return target_neighbors;
}

OuterProductsMatrixType CLMNNImpl::compute_outer_products(CDenseFeatures<float64_t>* x)
{
	// get the number of examples from data
	int32_t n = x->get_num_vectors();
	// get the number of features
	int32_t d = x->get_num_features();
	// map the feature matrix (each column is a feture vector) to an Eigen matrix
	Map<const MatrixXd> X(x->get_feature_matrix().matrix, d, n);
	// outer products matrix allocation
	//FIXME avoid computing the n^2 elements
	OuterProductsMatrixType C;
	C.resize(n);
	for (int32_t i = 0; i < n; ++i)
		C[i].resize(n);

	for (int32_t i = 0; i < n; ++i)
	{
		for (int32_t j = 0; j < n; ++j)
		{
			VectorXd dx = X.col(i)-X.col(j);
			C[i][j] = dx*dx.transpose();
		}
	}

	return C;
}

MatrixXd CLMNNImpl::sum_outer_products(const OuterProductsMatrixType& C,
		const SGMatrix<index_t> target_nn)
{
	// initialize the sum of outer products (sop); assume there is at least
	// one element in C, which must be, otherwise LMNN is applied to zero examples 
	MatrixXd sop(C[0][0].rows(), C[0][0].cols());
	sop.setZero();

	// sum the outer products stored in C using the indices specified in target_nn
	for (index_t i = 0; i < target_nn.num_cols; ++i)
	{
		for (index_t j = 0; j < target_nn.num_rows; ++j)
			sop += C[i][target_nn(j,i)];
	}

	return sop;
}

ImpostorsSetType CLMNNImpl::find_impostors(CDenseFeatures<float64_t>* x,
		CMulticlassLabels* y, const MatrixXd& L, const SGMatrix<index_t> target_nn,
		const uint32_t iter, const uint32_t correction)
{
	SG_SDEBUG("Entering CLMNNImpl::find_impostors().\n")

	// get the number of examples from data
	int32_t n = x->get_num_vectors();
	// get the number of features
	int32_t d = x->get_num_features();
	// get the number of neighbors
	int32_t k = target_nn.num_rows;

	// map the feature matrix (each column is a feture vector) to an Eigen matrix
	Map<const MatrixXd> X(x->get_feature_matrix().matrix, d, n);
	// transform the feature vectors
	MatrixXd LX = L*X;

	// compute square distances plus margin from examples to target neighbors
	MatrixXd sqdists = CLMNNImpl::compute_sqdists(LX,target_nn);

	// initialize impostors set
	ImpostorsSetType N;
	// exact impostors set shared between calls to this method
	static ImpostorsSetType Nexact;

	// impostors search
	REQUIRE(correction>0, "The number of iterations between exact updates of the "
			"impostors set must be greater than 0\n")
	if ((iter % correction)==0)
	{
		Nexact = CLMNNImpl::find_impostors_exact(LX, sqdists, y, target_nn, n, k);
		N = Nexact;
	}
	else
	{
		N = CLMNNImpl::find_impostors_approx(LX, sqdists, Nexact, target_nn);
	}

	SG_SDEBUG("Leaving CLMNNImpl::find_impostors().\n")

	return N;
}

void CLMNNImpl::update_gradient(MatrixXd& G, const OuterProductsMatrixType& C,
		const ImpostorsSetType& Nc, const ImpostorsSetType& Np, float64_t regularization)
{
	// compute the difference sets
	ImpostorsSetType Np_Nc, Nc_Np;
	set_difference(Np.begin(), Np.end(), Nc.begin(), Nc.end(), inserter(Np_Nc, Np_Nc.begin()));
	set_difference(Nc.begin(), Nc.end(), Np.begin(), Np.end(), inserter(Nc_Np, Nc_Np.begin()));

	// remove the gradient contributions of the impostors that were in the previous
	// set but disappeared in the current
	for (ImpostorsSetType::iterator it = Np_Nc.begin(); it != Np_Nc.end(); ++it)
		G -= regularization*(C[it->example][it->target] - C[it->example][it->impostor]);

	// add the gradient contributions of the new impostors
	for (ImpostorsSetType::iterator it = Nc_Np.begin(); it != Nc_Np.end(); ++it)
		G += regularization*(C[it->example][it->target] - C[it->example][it->impostor]);
}

void CLMNNImpl::gradient_step(MatrixXd& L, const MatrixXd& G, float64_t stepsize)
{
	// do step in L along the gradient direction (no need to project M then)
	L -= stepsize*(2*L*G);
}

float64_t CLMNNImpl::compute_objective(const CDenseFeatures<float64_t>* x, const MatrixXd& L,
		const OuterProductsMatrixType& C, const SGMatrix<index_t> target_nn,
		const ImpostorsSetType& Nc, float64_t regularization)
{
	// get the number of examples from data
	int32_t n = x->get_num_vectors();
	// get the number of target neighbors per example (k) from the arguments
	int32_t k = target_nn.num_rows;
	// initialize the objective
	float64_t obj = 0;
	// pre-compute the Mahalanobis distance matrix
	MatrixXd M = L.transpose()*L;

	// add pull contributions to the objective
	for (int32_t i = 0; i < n; ++i) // for each training example
	{
		for (int32_t j = 0; j < k; ++j) // for each target neighbor
			obj += (1-regularization)*TRACE(M,C[i][ target_nn(j,i) ]);
	}

	// add push contributions to the objective
	for (ImpostorsSetType::iterator it = Nc.begin(); it != Nc.end(); ++it) // for each possible impostor
	{
		double hinge = 1 + TRACE(M,C[it->example][it->target]) - TRACE(M,C[it->example][it->impostor]);

		if (hinge > 0)
			obj += regularization*hinge;
	}

	return obj;
}

void CLMNNImpl::correct_stepsize(float64_t& stepsize, const SGVector<float64_t> obj, const uint32_t iter)
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

MatrixXd CLMNNImpl::compute_sqdists(MatrixXd& LX, const SGMatrix<index_t> target_nn)
{
	// get the number of examples
	ASSERT(LX.cols()==target_nn.num_cols)
	int32_t n = LX.cols();
	// get the number of features
	int32_t d = LX.rows();
	// get the number of neighbors
	int32_t k = target_nn.num_rows;

	/// compute square distances to target neighbors plus margin

	// create Shogun features from LX to later apply subset
	SGMatrix<float64_t> lx_mat(LX.data(), d, n, false);
	CDenseFeatures<float64_t>* lx = new CDenseFeatures<float64_t>(lx_mat);

	// initialize distances
	MatrixXd sqdists(k,n);
	sqdists.setZero();
	for (int32_t i = 0; i < k; ++i)
	{
		//FIXME avoid copying the rows of target_nn and access them directly. Maybe
		//find_target_nn should be changed to return the output transposed wrt how it is
		//done atm.
		SGVector<index_t> subset_vec = target_nn.get_row_vector(i);
		lx->add_subset(subset_vec);
		// after the subset, there are still n columns, i.e. the subset is used to
		// modify the order of the columns in x according to the target neighbors
		sqdists.row(i) = SUMSQCOLS(LX - Map<const MatrixXd>(lx->get_feature_matrix().matrix, d, n)) + 1;
		lx->remove_subset();
	}

	// clean up features used to apply subset
	SG_UNREF(lx);

	return sqdists;
}

ImpostorsSetType CLMNNImpl::find_impostors_exact(const MatrixXd& LX, const MatrixXd& sqdists,
		CMulticlassLabels* y, const SGMatrix<index_t> target_nn, int32_t n, int32_t k)
{
	SG_SDEBUG("Entering CLMNNImpl::find_impostors_exact().\n")

	// initialize empty impostors set
	ImpostorsSetType N = ImpostorsSetType();

	// brute force search of impostors
	for (int32_t i = 0; i < n; ++i) // for each training example
		for (int32_t j = 0; j < k; ++j) // for each target neighbor
			for (int32_t l = 0; l < n; ++l)
				if (y->get_label(i)!=y->get_label(l)) // for each possible impostor
				{
					// compute the square distance to the current training example and
					// compare with the distance plus margin to the current target neighbor	
					if ( SUMSQCOLS(LX.col(i) - LX.col(l)).coeff(0) <= sqdists(j,i) )
						N.insert( CImpostorNode(i, target_nn(j,i), l) );
				}

	SG_SDEBUG("Leaving CLMNNImpl::find_impostors_exact().\n")

	return N;
}

ImpostorsSetType CLMNNImpl::find_impostors_approx(const MatrixXd& LX, const MatrixXd& sqdists,
		const ImpostorsSetType& Nexact, const SGMatrix<index_t> target_nn)
{
	SG_SDEBUG("Entering CLMNNImpl::find_impostors_approx().\n")

	// initialize empty impostors set
	ImpostorsSetType N = ImpostorsSetType();

	// find in the exact set of impostors computed last, the triplets that remain impostors
	for (ImpostorsSetType::iterator it = Nexact.begin(); it != Nexact.end(); ++it)
	{
		// find in target_nn(:,it->example) the position of the target neighbor it->target
		index_t target_idx = 0;
		while (target_nn(target_idx, it->example)!=it->target && target_idx<target_nn.num_rows)
			++target_idx;

		REQUIRE(target_idx<target_nn.num_rows, "The index of the target neighbour in the "
				"impostors set was not found in the target neighbours matrix. "
				"There must be a bug in find_impostors_exact.\n")

		if ( SUMSQCOLS(LX.col(it->example) - LX.col(it->impostor)).coeff(0) <= sqdists(target_idx, it->example) )
			N.insert(*it);
	}

	SG_SDEBUG("Leaving CLMNNImpl::find_impostors_approx().\n")

	return N;
}

#endif /* HAVE_EIGEN3 */
