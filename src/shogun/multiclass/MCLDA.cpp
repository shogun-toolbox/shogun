/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Michele Mazzoni, Soumyajit De,
 *          Weijie Lin, Bjoern Esser, Soeren Sonnenburg, Sanuj Sharma
 */

#include <shogun/lib/common.h>


#include <shogun/multiclass/MCLDA.h>
#include <shogun/machine/NativeMulticlassMachine.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

MCLDA::MCLDA(float64_t tolerance, bool store_cov)
: NativeMulticlassMachine()
{
	init();
	m_tolerance=tolerance;
	m_store_cov=store_cov;

}

MCLDA::MCLDA(std::shared_ptr<DenseFeatures<float64_t>> traindat, std::shared_ptr<Labels> trainlab, float64_t tolerance, bool store_cov)
: NativeMulticlassMachine()
{
	init();

	m_tolerance=tolerance;
	m_store_cov=store_cov;

	set_features(traindat);
	set_labels(trainlab);
}

MCLDA::~MCLDA()
{


	cleanup();
}

void MCLDA::init()
{
	SG_ADD(&m_tolerance, "m_tolerance", "Tolerance member.", ParameterProperties::HYPER);
	SG_ADD(&m_store_cov, "m_store_cov", "Store covariance member");
	SG_ADD((std::shared_ptr<SGObject>*) &m_features, "m_features", "Feature object.");
	SG_ADD(&m_means, "m_means", "Mean vectors list");
	SG_ADD(&m_cov, "m_cov", "covariance matrix");
	SG_ADD(&m_xbar, "m_xbar", "total mean");
	SG_ADD(&m_scalings, "m_scalings", "scalings");
	SG_ADD(&m_rank, "m_rank", "rank");
	SG_ADD(&m_dim, "m_dim", "dimension of feature space");
	SG_ADD(
	    &m_num_classes, "m_num_classes", "number of classes");
	SG_ADD(&m_coef, "m_coef", "weight vector");
	SG_ADD(&m_intercept, "m_intercept", "intercept");

	m_features  = NULL;
	m_num_classes=0;
	m_dim=0;
	m_rank=0;
}

void MCLDA::cleanup()
{
	m_num_classes = 0;
}

std::shared_ptr<MulticlassLabels> MCLDA::apply_multiclass(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type DotFeatures\n")

		set_features(data->as<DotFeatures>());
	}

	if (!m_features)
		return NULL;

	int32_t num_vecs = m_features->get_num_vectors();
	ASSERT(num_vecs > 0)
	ASSERT( m_dim == m_features->get_dim_feature_space() );

	// collect features into a matrix
	auto rf = m_features->as<DenseFeatures<float64_t>>();

	MatrixXd X(num_vecs, m_dim);

	int32_t vlen;
	bool vfree;
	float64_t* vec;
	Map< VectorXd > Em_xbar(m_xbar, m_dim);
	for (int i = 0; i < num_vecs; i++)
	{
		vec = rf->get_feature_vector(i, vlen, vfree);
		ASSERT(vec)

		Map< VectorXd > Evec(vec, vlen);

		X.row(i) = Evec - Em_xbar;

		rf->free_feature_vector(vec, i, vfree);
	}

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying X ...\n")
	SGMatrix< float64_t >::display_matrix(X.data(), num_vecs, m_dim);
#endif

	// center and scale data
	MatrixXd Xs(num_vecs, m_rank);
	Map< MatrixXd > Em_scalings(m_scalings.matrix, m_dim, m_rank);
	Xs = X*Em_scalings;

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying Xs ...\n")
	SGMatrix< float64_t >::display_matrix(Xs.data(), num_vecs, m_rank);
#endif

	// decision function
	MatrixXd d(num_vecs, m_num_classes);
	Map< MatrixXd > Em_coef(m_coef.matrix, m_num_classes, m_rank);
	Map< VectorXd > Em_intercept(m_intercept.vector, m_num_classes);
	d = (Xs*Em_coef.transpose()).rowwise() + Em_intercept.transpose();

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying d ...\n")
	SGMatrix< float64_t >::display_matrix(d.data(), num_vecs, m_num_classes);
#endif

	// argmax to apply labels
	auto out = std::make_shared<MulticlassLabels>(num_vecs);
	for (int i = 0; i < num_vecs; i++)
		out->set_label(i, Math::arg_max(d.data()+i, num_vecs, m_num_classes));

	return out;
}

bool MCLDA::train_machine(std::shared_ptr<Features> data)
{
	if (!m_labels)
		SG_ERROR("No labels allocated in MCLDA training\n")

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Speficied features are not of type DotFeatures\n")

		set_features(data->as<DotFeatures>());
	}

	if (!m_features)
		SG_ERROR("No features allocated in MCLDA training\n")

	SGVector< int32_t > train_labels = multiclass_labels(m_labels)->get_int_labels();

	if (!train_labels.vector)
		SG_ERROR("No train_labels allocated in MCLDA training\n")

	cleanup();

	m_num_classes = multiclass_labels(m_labels)->get_num_classes();
	m_dim = m_features->get_dim_feature_space();
	int32_t num_vec  = m_features->get_num_vectors();

	if (num_vec != train_labels.vlen)
		SG_ERROR("Dimension mismatch between features and labels in MCLDA training")

	int32_t* class_idxs = SG_MALLOC(int32_t, num_vec*m_num_classes);
	int32_t* class_nums = SG_MALLOC(int32_t, m_num_classes); // number of examples of each class
	memset(class_nums, 0, m_num_classes*sizeof(int32_t));

	for (int i = 0; i < train_labels.vlen; i++)
	{
		int32_t class_idx = train_labels.vector[i];

		if (class_idx < 0 || class_idx >= m_num_classes)
		{
			SG_ERROR(
			    "Label %d with value %d is out of {0, 1, 2, ..., "
			    "num_classes-1}, where num_classes is %d\n",
			    i, class_idx, m_num_classes);
			return false;
		}
		else
		{
			class_idxs[ class_idx*num_vec + class_nums[class_idx]++ ] = i;
		}
	}

	for (int i = 0; i < m_num_classes; i++)
	{
		if (class_nums[i] <= 0)
		{
			SG_ERROR("What? One class with no elements\n")
			return false;
		}
	}

	auto rf = m_features->as<DenseFeatures<float64_t>>();

	// if ( m_store_cov )
		index_t * cov_dims = SG_MALLOC(index_t, 3);
		cov_dims[0] = m_dim;
		cov_dims[1] = m_dim;
		cov_dims[2] = m_num_classes;
		SGNDArray< float64_t > covs(cov_dims, 3);

	m_means = SGMatrix< float64_t >(m_dim, m_num_classes, true);

	// matrix of all samples
	MatrixXd X =  MatrixXd::Zero(num_vec, m_dim);
	int32_t iX = 0;

	m_means.zero();

	int32_t vlen;
	bool vfree;
	float64_t* vec;
	for (int k = 0; k < m_num_classes; k++)
	{
		// gather all the samples for class k into buffer and calculate the mean of class k
		MatrixXd buffer(class_nums[k], m_dim);
		Map< VectorXd > Em_mean(m_means.get_column_vector(k), m_dim);
		for (int i = 0; i < class_nums[k]; i++)
		{
			vec = rf->get_feature_vector(class_idxs[k*num_vec + i], vlen, vfree);
			ASSERT(vec)

			Map< VectorXd > Evec(vec, vlen);
			Em_mean += Evec;
			buffer.row(i) = Evec;

			rf->free_feature_vector(vec, class_idxs[k*num_vec + i], vfree);
		}

		Em_mean /= class_nums[k];

		// subtract the mean of class k from each sample of class k and store the centered data in Xc
		for (int i = 0; i < class_nums[k]; i++)
		{
			buffer.row(i) -= Em_mean;
			X.row(iX) += buffer.row(i);
			iX++;
		}

		if (m_store_cov)
		{
			// calc cov = buffer.T * buffer
			Map< MatrixXd > Em_cov_k(covs.get_matrix(k), m_dim, m_dim);
			Em_cov_k = buffer.transpose() * buffer;
		}
	}

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying means ...\n")
	SGMatrix< float64_t >::display_matrix(m_means.matrix, m_dim, m_num_classes);
#endif

	if (m_store_cov)
	{
		m_cov = SGMatrix< float64_t >(m_dim, m_dim, true);
		m_cov.zero();
		Map< MatrixXd > Em_cov(m_cov.matrix, m_dim, m_dim);

		for (int k = 0; k < m_num_classes; k++)
		{
			Map< MatrixXd > Em_cov_k(covs.get_matrix(k), m_dim, m_dim);
			Em_cov += Em_cov_k;
		}

		Em_cov /= m_num_classes;
	}

#ifdef DEBUG_MCLDA
	if (m_store_cov)
	{
		SG_PRINT("\n>>> Displaying cov ...\n")
		SGMatrix< float64_t >::display_matrix(m_cov.matrix, m_dim, m_dim);
	}
#endif

	///////////////////////////////////////////////////////////
	// 1) within (univariate) scaling by with classes std-dev

	// std-dev of X
	m_xbar = SGVector< float64_t >(m_dim);
	m_xbar.zero();
	Map< VectorXd > Em_xbar(m_xbar.vector, m_dim);
	Em_xbar = X.colwise().sum();
	Em_xbar /= num_vec;

	VectorXd std = VectorXd::Zero(m_dim);
	std = (X.rowwise() - Em_xbar.transpose()).array().pow(2).colwise().sum();
	std = std.array() / num_vec;

	for (int j = 0; j < m_dim; j++)
		if(std[j] == 0)
			std[j] = 1;

	float64_t fac = 1.0 / (num_vec - m_num_classes);

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying m_xbar ...\n")
	SGVector< float64_t >::display_vector(m_xbar.vector, m_dim);

	SG_PRINT("\n>>> Displaying std ...\n")
	SGVector< float64_t >::display_vector(std.data(), m_dim);
#endif

	///////////////////////////////
	// 2) Within variance scaling
	for (int i = 0; i < num_vec; i++)
		X.row(i) = sqrt(fac) * X.row(i).array() / std.transpose().array();


	// SVD of centered (within)scaled data
	VectorXd S(m_dim);
	MatrixXd V(m_dim, m_dim);

	Eigen::JacobiSVD<MatrixXd> eSvd;
	eSvd.compute(X,Eigen::ComputeFullV);
	sg_memcpy(S.data(), eSvd.singularValues().data(), m_dim*sizeof(float64_t));
	sg_memcpy(V.data(), eSvd.matrixV().data(), m_dim*m_dim*sizeof(float64_t));
	V.transposeInPlace();

	int rank = 0;
	while (rank < m_dim && S[rank] > m_tolerance)
	{
		rank++;
	}

	if (rank < m_dim)
		SG_ERROR("Warning: Variables are collinear\n")

	MatrixXd scalings(m_dim, rank);
	for (int i = 0; i < m_dim; i++)
		for (int j = 0; j < rank; j++)
			scalings(i,j) = V(j,i) / std[j] / S[j];

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying scalings ...\n")
	SGMatrix< float64_t >::display_matrix(scalings.data(), m_dim, rank);
#endif

	///////////////////////////////
	// 3) Between variance scaling

	// Xc = m_means dot scalings
	MatrixXd Xc(m_num_classes, rank);
	Map< MatrixXd > Em_means(m_means.matrix, m_dim, m_num_classes);
	Xc = (Em_means.transpose()*scalings);

	for (int i = 0; i < m_num_classes; i++)
		Xc.row(i) *= sqrt(class_nums[i] * fac);

	// Centers are living in a space with n_classes-1 dim (maximum)
	// Use svd to find projection in the space spanned by the
	// (n_classes) centers
	S = VectorXd(rank);
	V = MatrixXd(rank, rank);

	eSvd.compute(Xc,Eigen::ComputeFullV);
	sg_memcpy(S.data(), eSvd.singularValues().data(), rank*sizeof(float64_t));
	sg_memcpy(V.data(), eSvd.matrixV().data(), rank*rank*sizeof(float64_t));

	m_rank = 0;
	while (m_rank < rank && S[m_rank] > m_tolerance*S[0])
	{
		m_rank++;
	}

	// compose the scalings
	m_scalings  = SGMatrix< float64_t >(rank, m_rank);
	Map< MatrixXd > Em_scalings(m_scalings.matrix, rank, m_rank);
	Em_scalings = scalings * V.leftCols(m_rank);

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying m_scalings ...\n")
	SGMatrix< float64_t >::display_matrix(m_scalings.matrix, rank, m_rank);
#endif

	// weight vectors / centroids
	MatrixXd meansc(m_dim, m_num_classes);
	meansc = Em_means.colwise() - Em_xbar;

	m_coef = SGMatrix< float64_t >(m_num_classes, m_rank);
	Map< MatrixXd > Em_coef(m_coef.matrix, m_num_classes, m_rank);
	Em_coef = meansc.transpose() * Em_scalings;

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying m_coefs ...\n")
	SGMatrix< float64_t >::display_matrix(m_coef.matrix, m_num_classes, m_rank);
#endif

	// intercept
	m_intercept  = SGVector< float64_t >(m_num_classes);
	m_intercept.zero();
	for (int j = 0; j < m_num_classes; j++)
		m_intercept[j] = -0.5*m_coef[j]*m_coef[j] + log(class_nums[j]/float(num_vec));

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying m_intercept ...\n")
	SGVector< float64_t >::display_vector(m_intercept.vector, m_num_classes);
#endif

	SG_FREE(class_idxs);
	SG_FREE(class_nums);

	return true;
}

