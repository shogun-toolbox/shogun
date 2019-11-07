/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, Viktor Gal,
 *          Evan Shelhamer, Evgeniy Andreev, Marc Zimmermann, Bjoern Esser
 */
#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/PCA.h>

using namespace shogun;
using namespace Eigen;

PCA::PCA(
    bool do_whitening, EPCAMode mode, float64_t thresh, EPCAMethod method,
    EPCAMemoryMode mem_mode)
    : DensePreprocessor<float64_t>()
{
	init();
	m_whitening = do_whitening;
	m_mode = mode;
	m_thresh = thresh;
	m_mem_mode = mem_mode;
	m_method = method;
}

PCA::PCA(EPCAMethod method, bool do_whitening, EPCAMemoryMode mem_mode)
    : DensePreprocessor<float64_t>()
{
	init();
	m_whitening = do_whitening;
	m_mem_mode = mem_mode;
	m_method = method;
}

void PCA::init()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_mean_vector = SGVector<float64_t>();
	m_eigenvalues_vector = SGVector<float64_t>();
	num_dim = 0;
	m_whitening = false;
	m_mode = FIXED_NUMBER;
	m_thresh = 1e-6;
	m_mem_mode = MEM_REALLOCATE;
	m_method = AUTO;
	m_eigenvalue_zero_tolerance = 1e-15;
	m_target_dim = 1;

	SG_ADD(
	    &m_transformation_matrix, "transformation_matrix",
	    "Transformation matrix (Eigenvectors of covariance matrix).");
	SG_ADD(&m_mean_vector, "mean_vector", "Mean Vector.");
	SG_ADD(
	    &m_eigenvalues_vector, "eigenvalues_vector",
	    "Vector with Eigenvalues.");
	SG_ADD(
	    &m_whitening, "whitening", "Whether data shall be whitened.",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_thresh, "thresh", "Cutoff threshold.", ParameterProperties::HYPER);
	SG_ADD(
	    &m_eigenvalue_zero_tolerance, "eigenvalue_zero_tolerance",
	    "zero tolerance"
	    " for determining zero eigenvalues during whitening to avoid numerical "
	    "issues");
	SG_ADD(
	    &m_target_dim, "target_dim", "target dimensionality of preprocessor",
	    ParameterProperties::HYPER);
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_mode, "mode", "PCA Mode.",
	    ParameterProperties::HYPER,
	    SG_OPTIONS(THRESHOLD, VARIANCE_EXPLAINED, FIXED_NUMBER));
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_mem_mode, "mem_mode",
	    "Memory mode (in-place or reallocation).", ParameterProperties::NONE,
	    SG_OPTIONS(MEM_REALLOCATE, MEM_IN_PLACE));
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_method, "method",
	    "Method used for PCA calculation", ParameterProperties::NONE,
	    SG_OPTIONS(AUTO, SVD, EVD));
}

PCA::~PCA()
{
}

void PCA::fit(std::shared_ptr<Features> features)
{
	if (m_fitted)
		cleanup();

	auto feature_matrix =
	    features->as<DenseFeatures<float64_t>>()->get_feature_matrix();
	auto num_vectors = feature_matrix.num_cols;
	auto num_features = feature_matrix.num_rows;
	io::info("num_examples: {} num_features: {}", num_vectors, num_features);

	// max target dim allowed
	auto max_dim_allowed = std::min(num_vectors, num_features);
	num_dim = 0;

	require(
	    m_target_dim <= max_dim_allowed,
	    "target dimension should be less or equal to than minimum of N and D");

	// center data
	Map<MatrixXd> fmatrix(feature_matrix.matrix, num_features, num_vectors);

	m_mean_vector = SGVector<float64_t>(num_features);
	Map<VectorXd> data_mean(m_mean_vector.vector, num_features);
	data_mean = fmatrix.rowwise().sum() / (float64_t)num_vectors;
	fmatrix = fmatrix.colwise() - data_mean;

	m_eigenvalues_vector = SGVector<float64_t>(max_dim_allowed);

	if (m_method == AUTO)
		m_method = (num_vectors > num_features) ? EVD : SVD;

	if (m_method == EVD)
		init_with_evd(feature_matrix, max_dim_allowed);
	else
		init_with_svd(feature_matrix, max_dim_allowed);

	// restore feature matrix
	fmatrix = fmatrix.colwise() + data_mean;
	m_fitted = true;
}

void PCA::init_with_evd(const SGMatrix<float64_t>& feature_matrix, int32_t max_dim_allowed)
{
	int32_t num_vectors = feature_matrix.num_cols;
	int32_t num_features = feature_matrix.num_rows;

	Map<MatrixXd> fmatrix(feature_matrix.matrix, num_features, num_vectors);
	Map<VectorXd> eigenValues(m_eigenvalues_vector.vector, max_dim_allowed);

	// covariance matrix
	MatrixXd cov_mat(num_features, num_features);
	cov_mat = fmatrix*fmatrix.transpose();
	cov_mat /= (num_vectors-1);

	io::info("Computing Eigenvalues");
	// eigen value computed
	SelfAdjointEigenSolver<MatrixXd> eigenSolve =
			SelfAdjointEigenSolver<MatrixXd>(cov_mat);
	eigenValues = eigenSolve.eigenvalues().tail(max_dim_allowed);

	// target dimension
	switch (m_mode)
	{
		case FIXED_NUMBER :
			num_dim = m_target_dim;
			break;

		case VARIANCE_EXPLAINED :
			{
				float64_t eig_sum = eigenValues.sum();
				float64_t com_sum = 0;
				for (int32_t i=num_features-1; i<-1; i++)
				{
					num_dim++;
					com_sum += m_eigenvalues_vector.vector[i];
					if (com_sum/eig_sum>=m_thresh)
						break;
				}
			}
			break;

		case THRESHOLD :
			for (int32_t i=num_features-1; i<-1; i++)
			{
				if (m_eigenvalues_vector.vector[i]>m_thresh)
					num_dim++;
				else
					break;
			}
			break;
	};
	io::info("Reducing from {} to {} features", num_features, num_dim);

	m_transformation_matrix = SGMatrix<float64_t>(num_features,num_dim);
	Map<MatrixXd> transformMatrix(m_transformation_matrix.matrix,
						 num_features, num_dim);
	num_old_dim = num_features;

	// eigenvector matrix
	transformMatrix = eigenSolve.eigenvectors().block(0,
				num_features-num_dim, num_features,num_dim);
	if (m_whitening)
	{
		for (int32_t i=0; i<num_dim; i++)
		{
			if (Math::fequals_abs<float64_t>(0.0, eigenValues[i+max_dim_allowed-num_dim],
									m_eigenvalue_zero_tolerance))
			{
				io::warn(
				    "Covariance matrix has almost zero Eigenvalue (ie "
				    "Eigenvalue within a tolerance of {:E} around 0) at "
				    "dimension {}. Consider reducing its dimension.",
				    m_eigenvalue_zero_tolerance,
				    i + max_dim_allowed - num_dim + 1);

				transformMatrix.col(i) = MatrixXd::Zero(num_features,1);
				continue;
			}

			transformMatrix.col(i) /= std::sqrt(
			    eigenValues[i + max_dim_allowed - num_dim] * (num_vectors - 1));
		}
	}
}

void PCA::init_with_svd(const SGMatrix<float64_t> &feature_matrix, int32_t max_dim_allowed)
{
	int32_t num_vectors = feature_matrix.num_cols;
	int32_t num_features = feature_matrix.num_rows;

	Map<MatrixXd> fmatrix(feature_matrix.matrix, num_features, num_vectors);
	Map<VectorXd> eigenValues(m_eigenvalues_vector.vector, max_dim_allowed);

	// compute SVD of data matrix
	JacobiSVD<MatrixXd> svd(fmatrix.transpose(), ComputeThinU | ComputeThinV);

	// compute non-negative eigen values from singular values
	eigenValues = svd.singularValues();
	eigenValues = eigenValues.cwiseProduct(eigenValues) / (num_vectors - 1);

	// target dimension
	switch (m_mode)
	{
		case FIXED_NUMBER:
			num_dim = m_target_dim;
        	break;

		case VARIANCE_EXPLAINED:
		{
			float64_t eig_sum = eigenValues.sum();
			float64_t com_sum = 0;
			for (int32_t i = 0; i < num_features; i++) {
				num_dim++;
				com_sum += m_eigenvalues_vector.vector[i];
				if (com_sum / eig_sum >= m_thresh)
					break;
			}
		} break;

		case THRESHOLD:
			for (int32_t i = 0; i < num_features; i++) {
				if (m_eigenvalues_vector.vector[i] > m_thresh)
					num_dim++;
				else
					break;
			}
			break;
	};
	io::info("Reducing from {} to {} features...", num_features, num_dim);

	// right singular vectors form eigenvectors
	m_transformation_matrix = SGMatrix<float64_t>(num_features, num_dim);
	Map<MatrixXd> transformMatrix(m_transformation_matrix.matrix, num_features, num_dim);
	num_old_dim = num_features;
	transformMatrix = svd.matrixV().block(0, 0, num_features, num_dim);

	if (m_whitening)
	{
		for (int32_t i = 0; i < num_dim; i++)
		{
			if (Math::fequals_abs<float64_t>(0.0, eigenValues[i], m_eigenvalue_zero_tolerance))
			{

				io::warn("Covariance matrix has almost zero Eigenvalue (ie "
					"Eigenvalue within a tolerance of {:E} around 0) at "
					"dimension {}. Consider reducing its dimension.",
					m_eigenvalue_zero_tolerance, i + 1);

				transformMatrix.col(i) = MatrixXd::Zero(num_features, 1);
				continue;
			}

			transformMatrix.col(i) /=
			    std::sqrt(eigenValues[i] * (num_vectors - 1));
		}
	}
}

void PCA::cleanup()
{
	m_transformation_matrix=SGMatrix<float64_t>();
        m_mean_vector = SGVector<float64_t>();
        m_eigenvalues_vector = SGVector<float64_t>();
	    m_fitted = false;
}

SGMatrix<float64_t> PCA::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	assert_fitted();

	auto num_vectors = matrix.num_cols;
	auto num_features = matrix.num_rows;

	io::info("Transforming feature matrix");
	Map<MatrixXd> transform_matrix(m_transformation_matrix.matrix,
			m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);

	if (m_mem_mode == MEM_IN_PLACE)
	{
		io::info("Preprocessing feature matrix");
		Map<MatrixXd> feature_matrix(matrix.matrix, num_features, num_vectors);
		VectorXd data_mean =
		    feature_matrix.rowwise().sum() / (float64_t)num_vectors;
		feature_matrix = feature_matrix.colwise() - data_mean;

		feature_matrix.block(0, 0, num_dim, num_vectors) =
		    transform_matrix.transpose() * feature_matrix;

		io::info("Form matrix of target dimension");
		for (int32_t col = 0; col < num_vectors; col++)
		{
			for (int32_t row = 0; row < num_dim; row++)
				matrix.matrix[col * num_dim + row] = feature_matrix(row, col);
		}
		matrix.num_rows = num_dim;
		matrix.num_cols = num_vectors;

		return matrix;
	}
	else
	{
		SGMatrix<float64_t> ret(num_dim, num_vectors);
		Map<MatrixXd> ret_matrix(ret.matrix, num_dim, num_vectors);

		io::info("Preprocessing feature matrix");
		Map<MatrixXd> feature_matrix(matrix.matrix, num_features, num_vectors);
		VectorXd data_mean =
		    feature_matrix.rowwise().sum() / (float64_t)num_vectors;
		feature_matrix = feature_matrix.colwise() - data_mean;

		ret_matrix = transform_matrix.transpose() * feature_matrix;

		return ret;
	}
}

SGVector<float64_t> PCA::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SGVector<float64_t> result = SGVector<float64_t>(num_dim);
	Map<VectorXd> resultVec(result.vector, num_dim);
	Map<VectorXd> inputVec(vector.vector, vector.vlen);

	Map<VectorXd> mean(m_mean_vector.vector, m_mean_vector.vlen);
	Map<MatrixXd> transformMat(m_transformation_matrix.matrix,
		 m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);

	inputVec = inputVec-mean;
	resultVec = transformMat.transpose()*inputVec;
	inputVec = inputVec+mean;

	return result;
}

SGMatrix<float64_t> PCA::get_transformation_matrix()
{
	return m_transformation_matrix;
}

SGVector<float64_t> PCA::get_eigenvalues()
{
	return m_eigenvalues_vector;
}

SGVector<float64_t> PCA::get_mean()
{
	return m_mean_vector;
}

EPCAMemoryMode PCA::get_memory_mode() const
{
	return m_mem_mode;
}

void PCA::set_memory_mode(EPCAMemoryMode e)
{
	m_mem_mode = e;
}

void PCA::set_eigenvalue_zero_tolerance(float64_t eigenvalue_zero_tolerance)
{
	m_eigenvalue_zero_tolerance = eigenvalue_zero_tolerance;
}

float64_t PCA::get_eigenvalue_zero_tolerance() const
{
	return m_eigenvalue_zero_tolerance;
}

void PCA::set_target_dim(int32_t dim)
{
	ASSERT(dim > 0)
	m_target_dim = dim;
}

int32_t PCA::get_target_dim() const
{
	return m_target_dim;
}
