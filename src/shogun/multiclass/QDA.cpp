/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Soeren Sonnenburg, Sergey Lisitsyn, Michele Mazzoni,
 *          Heiko Strathmann, Sanuj Sharma, Weijie Lin, Bjoern Esser,
 *          Youssef Emad El-Din, Sourav Singh, Pan Deng
 */

#include <shogun/lib/common.h>


#include <shogun/multiclass/QDA.h>
#include <shogun/machine/NativeMulticlassMachine.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

QDA::QDA() : NativeMulticlassMachine(), m_num_classes(0), m_dim(0)
{
	init();
}

QDA::QDA(float64_t tolerance, bool store_covs)
: NativeMulticlassMachine(), m_num_classes(0), m_dim(0)
{
	init();
	m_tolerance = tolerance;
	m_store_covs = store_covs;
}

QDA::QDA(std::shared_ptr<DenseFeatures<float64_t>> traindat, std::shared_ptr<Labels> trainlab)
: NativeMulticlassMachine(), m_num_classes(0), m_dim(0)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
}

QDA::QDA(std::shared_ptr<DenseFeatures<float64_t>> traindat, std::shared_ptr<Labels> trainlab, float64_t tolerance)
: NativeMulticlassMachine(), m_num_classes(0), m_dim(0)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
	m_tolerance = tolerance;
}

QDA::QDA(std::shared_ptr<DenseFeatures<float64_t>> traindat, std::shared_ptr<Labels> trainlab, bool store_covs)
: NativeMulticlassMachine(), m_num_classes(0), m_dim(0)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
	m_store_covs = store_covs;
}



QDA::QDA(std::shared_ptr<DenseFeatures<float64_t>> traindat, std::shared_ptr<Labels> trainlab, float64_t tolerance, bool store_covs)
: NativeMulticlassMachine(), m_num_classes(0), m_dim(0)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
	m_tolerance = tolerance;
	m_store_covs = store_covs;
}

QDA::~QDA()
{


	cleanup();
}

void QDA::init()
{
	m_tolerance = 1e-4;
	m_store_covs = false;
	SG_ADD(&m_tolerance, "m_tolerance", "Tolerance member.", ParameterProperties::HYPER);
	SG_ADD(&m_store_covs, "m_store_covs", "Store covariances member");
	SG_ADD((std::shared_ptr<SGObject>*) &m_features, "m_features", "Feature object.");
	SG_ADD(&m_means, "m_means", "Mean vectors list");
	SG_ADD(&m_slog, "m_slog", "Vector used in classification");
	SG_ADD(&m_dim, "m_dim", "dimension of feature space");
	SG_ADD(
	    &m_num_classes, "m_num_classes", "number of classes");
	SG_ADD(&m_M, "m_M", "Matrices used in classification");

	m_features  = NULL;
}

void QDA::cleanup()
{
	m_means=SGMatrix<float64_t>();

	m_num_classes = 0;
}

std::shared_ptr<MulticlassLabels> QDA::apply_multiclass(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			error("Specified features are not of type CDotFeatures");

		set_features(data->as<DotFeatures>());
	}

	if ( !m_features )
		return NULL;

	int32_t num_vecs = m_features->get_num_vectors();
	ASSERT(num_vecs > 0)
	ASSERT( m_dim == m_features->get_dim_feature_space() )

	auto rf = m_features->as<DenseFeatures<float64_t>>();

	MatrixXd X(num_vecs, m_dim);
	MatrixXd A(num_vecs, m_dim);
	VectorXd norm2(num_vecs*m_num_classes);
	norm2.setZero();

	SGVector<float64_t> vec;
	for (int k = 0; k < m_num_classes; k++)
	{
		// X = features - means
		for (int i = 0; i < num_vecs; i++)
		{
			vec = rf->get_feature_vector(i);
			ASSERT(vec.vector)

			Map< VectorXd > Evec(vec, vec.vlen);
			Map< VectorXd > Em_means_col(m_means.get_column_vector(k), m_dim);

			X.row(i) = Evec - Em_means_col;

			rf->free_feature_vector(vec, i);
		}

		Map<MatrixXd> Em_M(m_M.slice(m_dim * k, m_dim * (k + 1)));
		A = X*Em_M;

		for (int i = 0; i < num_vecs; i++)
			norm2(i + k*num_vecs) = A.row(i).array().square().sum();

	}

	for (int i = 0; i < num_vecs; i++)
		for (int k = 0; k < m_num_classes; k++)
		{
			norm2[i + k*num_vecs] += m_slog[k];
			norm2[i + k*num_vecs] *= -0.5;
		}


	auto out = std::make_shared<MulticlassLabels>(num_vecs);

	for (int i = 0 ; i < num_vecs; i++)
		out->set_label(i, Math::arg_max(norm2.data()+i, num_vecs, m_num_classes));

	return out;
}

bool QDA::train_machine(std::shared_ptr<Features> data)
{
	if (!m_labels)
		error("No labels allocated in QDA training");

	if ( data )
	{
		if (!data->has_property(FP_DOT))
			error("Speficied features are not of type CDotFeatures");

		set_features(data->as<DotFeatures>());
	}

	if (!m_features)
		error("No features allocated in QDA training");

	SGVector< int32_t > train_labels = multiclass_labels(m_labels)->get_int_labels();

	if (!train_labels.vector)
		error("No train_labels allocated in QDA training");

	cleanup();

	m_num_classes = multiclass_labels(m_labels)->get_num_classes();
	m_dim = m_features->get_dim_feature_space();
	int32_t num_vec  = m_features->get_num_vectors();

	if (num_vec != train_labels.vlen)
		error("Dimension mismatch between features and labels in QDA training");

	int32_t* class_idxs = SG_MALLOC(int32_t, num_vec*m_num_classes); // number of examples of each class
	int32_t* class_nums = SG_MALLOC(int32_t, m_num_classes);
	memset(class_nums, 0, m_num_classes*sizeof(int32_t));
	int32_t class_idx;

	for (int i = 0; i < train_labels.vlen; i++)
	{
		class_idx = train_labels.vector[i];

		if (class_idx < 0 || class_idx >= m_num_classes)
		{
			error("found label out of {0, 1, 2, ..., num_classes-1}...");
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
			error("What? One class with no elements");
			return false;
		}
	}

	if (m_store_covs)
	{
		// cov_dims will be free in m_covs.destroy_ndarray()
		index_t * cov_dims = SG_MALLOC(index_t, 3);
		cov_dims[0] = m_dim;
		cov_dims[1] = m_dim;
		cov_dims[2] = m_num_classes;
		m_covs = SGNDArray< float64_t >(cov_dims, 3);
	}

	m_means = SGMatrix< float64_t >(m_dim, m_num_classes, true);
	SGMatrix< float64_t > scalings  = SGMatrix< float64_t >(m_dim, m_num_classes);

	// rot_dims will be freed in rotations.destroy_ndarray()
	index_t* rot_dims = SG_MALLOC(index_t, 3);
	rot_dims[0] = m_dim;
	rot_dims[1] = m_dim;
	rot_dims[2] = m_num_classes;
	SGNDArray< float64_t > rotations = SGNDArray< float64_t >(rot_dims, 3);

	auto rf = m_features->as<DenseFeatures<float64_t>>();

	m_means.zero();

	SGVector<float64_t> vec;
	for (int k = 0; k < m_num_classes; k++)
	{
		MatrixXd buffer(class_nums[k], m_dim);
		Map< VectorXd > Em_means(m_means.get_column_vector(k), m_dim);
		for (int i = 0; i < class_nums[k]; i++)
		{
			vec = rf->get_feature_vector(class_idxs[k*num_vec + i]);
			ASSERT(vec.vector)

			Map< VectorXd > Evec(vec, vec.vlen);
			Em_means += Evec;
			buffer.row(i) = Evec;

			rf->free_feature_vector(vec, class_idxs[k*num_vec + i]);
		}

		Em_means /= class_nums[k];

		for (int i = 0; i < class_nums[k]; i++)
			buffer.row(i) -= Em_means;

		// SVD
		float64_t * col = scalings.get_column_vector(k);
		float64_t * rot_mat = rotations.get_matrix(k);

		Eigen::JacobiSVD<MatrixXd> eSvd;
		eSvd.compute(buffer,Eigen::ComputeFullV);
		sg_memcpy(col, eSvd.singularValues().data(), m_dim*sizeof(float64_t));
		sg_memcpy(rot_mat, eSvd.matrixV().data(), m_dim*m_dim*sizeof(float64_t));

		SGVector<float64_t>::vector_multiply(col, col, col, m_dim);
		SGVector<float64_t>::scale_vector(1.0/(class_nums[k]-1), col, m_dim);

		if (m_store_covs)
		{
			SGMatrix< float64_t > M(m_dim ,m_dim);
			MatrixXd MEig = Map<MatrixXd>(rot_mat,m_dim,m_dim);
			for (int i = 0; i < m_dim; i++)
				for (int j = 0; j < m_dim; j++)
					M(i,j)*=scalings[k*m_dim + j];
			MatrixXd rotE = Map<MatrixXd>(rot_mat,m_dim,m_dim);
			MatrixXd resE(m_dim,m_dim);
			resE = MEig * rotE.transpose();
			sg_memcpy(m_covs.get_matrix(k),resE.data(),m_dim*m_dim*sizeof(float64_t));
		}
	}

	/* Computation of terms required for classification */
	SGVector< float32_t > sinvsqrt(m_dim);

	// m_num_classes matrices of dimension (m_dim, m_dim) stacked horizontally
	m_M = SGMatrix<float64_t>(m_dim, m_dim * m_num_classes);

	m_slog = SGVector< float32_t >(m_num_classes);
	m_slog.zero();

	index_t idx = 0;
	for (int k = 0; k < m_num_classes; k++)
	{
		for (int j = 0; j < m_dim; j++)
		{
			sinvsqrt[j] = 1.0 / std::sqrt(scalings[k * m_dim + j]);
			m_slog[k] += std::log(scalings[k * m_dim + j]);
		}

		for (int i = 0; i < m_dim; i++)
			for (int j = 0; j < m_dim; j++)
			{
				idx = k*m_dim*m_dim + i + j*m_dim;
				m_M[idx] = rotations[idx] * sinvsqrt[j];
			}
	}


	SG_FREE(class_idxs);
	SG_FREE(class_nums);
	return true;
}
