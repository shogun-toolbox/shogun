/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, Viktor Gal,
 *          Evan Shelhamer, Evgeniy Andreev, Marc Zimmermann, Bjoern Esser
 */
#include <shogun/lib/config.h>

#include <shogun/preprocessor/PCA.h>
#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CPCA::CPCA(
    bool do_whitening, EPCAMode mode, float64_t thresh, EPCAMethod method,
    EPCAMemoryMode mem_mode)
    : CDensePreprocessor<float64_t>()
{
	init();
	m_whitening = do_whitening;
	m_mode = mode;
	m_thresh = thresh;
	m_mem_mode = mem_mode;
	m_method = method;
}

CPCA::CPCA(EPCAMethod method, bool do_whitening, EPCAMemoryMode mem_mode)
    : CDensePreprocessor<float64_t>()
{
	init();
	m_whitening = do_whitening;
	m_mem_mode = mem_mode;
	m_method = method;
}

void CPCA::init()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_mean_vector = SGVector<float64_t>();
	m_eigenvalues_vector = SGVector<float64_t>();
	num_dim = 0;
	m_initialized = false;
	m_whitening = false;
	m_mode = FIXED_NUMBER;
	m_thresh = 1e-6;
	m_mem_mode = MEM_REALLOCATE;
	m_method = AUTO;
	m_eigenvalue_zero_tolerance=1e-15;

	SG_ADD(&m_transformation_matrix, "transformation_matrix",
	    "Transformation matrix (Eigenvectors of covariance matrix).",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_mean_vector, "mean_vector", "Mean Vector.", MS_NOT_AVAILABLE);
	SG_ADD(&m_eigenvalues_vector, "eigenvalues_vector",
	    "Vector with Eigenvalues.", MS_NOT_AVAILABLE);
	SG_ADD(&m_initialized, "initalized", "True when initialized.",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_whitening, "whitening", "Whether data shall be whitened.",
	    MS_AVAILABLE);
	SG_ADD((machine_int_t*) &m_mode, "mode", "PCA Mode.", MS_AVAILABLE);
	SG_ADD(&m_thresh, "m_thresh", "Cutoff threshold.", MS_AVAILABLE);
	SG_ADD((machine_int_t*) &m_mem_mode, "m_mem_mode",
		"Memory mode (in-place or reallocation).", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_method, "m_method",
		"Method used for PCA calculation", MS_NOT_AVAILABLE);
	SG_ADD(&m_eigenvalue_zero_tolerance, "eigenvalue_zero_tolerance", "zero tolerance"
	" for determining zero eigenvalues during whitening to avoid numerical issues", MS_NOT_AVAILABLE);
	SG_ADD(
	    &m_target_dim, "target_dim", "target dimensionality of preprocessor",
	    MS_AVAILABLE);
}

CPCA::~CPCA()
{
}

void CPCA::fit(CFeatures* features)
{
	if (m_initialized)
		cleanup();

	auto feature_matrix =
	    features->as<CDenseFeatures<float64_t>>()->get_feature_matrix();
	int32_t num_vectors = feature_matrix.num_cols;
	int32_t num_features = feature_matrix.num_rows;
	SG_INFO("num_examples: %d num_features: %d\n", num_vectors, num_features)

	// max target dim allowed
	int32_t max_dim_allowed = CMath::min(num_vectors, num_features);
	num_dim = 0;

	REQUIRE(
	    m_target_dim <= max_dim_allowed,
	    "target dimension should be less or equal to than minimum of N and D")

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
	m_initialized = true;
}

void CPCA::init_with_evd(const SGMatrix<float64_t>& feature_matrix, int32_t max_dim_allowed)
{
	int32_t num_vectors = feature_matrix.num_cols;
	int32_t num_features = feature_matrix.num_rows;

	Map<MatrixXd> fmatrix(feature_matrix.matrix, num_features, num_vectors);
	Map<VectorXd> eigenValues(m_eigenvalues_vector.vector, max_dim_allowed);

	// covariance matrix
	MatrixXd cov_mat(num_features, num_features);
	cov_mat = fmatrix*fmatrix.transpose();
	cov_mat /= (num_vectors-1);

	SG_INFO("Computing Eigenvalues\n")
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
	SG_INFO("Reducing from %i to %i features\n", num_features, num_dim)

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
			if (CMath::fequals_abs<float64_t>(0.0, eigenValues[i+max_dim_allowed-num_dim],
									m_eigenvalue_zero_tolerance))
			{
				SG_WARNING(
				    "Covariance matrix has almost zero Eigenvalue (ie "
				    "Eigenvalue within a tolerance of %E around 0) at "
				    "dimension %d. Consider reducing its dimension.\n",
				    m_eigenvalue_zero_tolerance,
				    i + max_dim_allowed - num_dim + 1)

				transformMatrix.col(i) = MatrixXd::Zero(num_features,1);
				continue;
			}

			transformMatrix.col(i) /= std::sqrt(
			    eigenValues[i + max_dim_allowed - num_dim] * (num_vectors - 1));
		}
	}
}

void CPCA::init_with_svd(const SGMatrix<float64_t> &feature_matrix, int32_t max_dim_allowed)
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
	SG_INFO("Reducing from %i to %i features...\n", num_features, num_dim)

	// right singular vectors form eigenvectors
	m_transformation_matrix = SGMatrix<float64_t>(num_features, num_dim);
	Map<MatrixXd> transformMatrix(m_transformation_matrix.matrix, num_features, num_dim);
	num_old_dim = num_features;
	transformMatrix = svd.matrixV().block(0, 0, num_features, num_dim);

	if (m_whitening)
	{
		for (int32_t i = 0; i < num_dim; i++)
		{
			if (CMath::fequals_abs<float64_t>(0.0, eigenValues[i], m_eigenvalue_zero_tolerance))
			{

				SG_WARNING("Covariance matrix has almost zero Eigenvalue (ie "
					"Eigenvalue within a tolerance of %E around 0) at "
					"dimension %d. Consider reducing its dimension.",
					m_eigenvalue_zero_tolerance, i + 1)

				transformMatrix.col(i) = MatrixXd::Zero(num_features, 1);
				continue;
			}

			transformMatrix.col(i) /=
			    std::sqrt(eigenValues[i] * (num_vectors - 1));
		}
	}
}

void CPCA::cleanup()
{
	m_transformation_matrix=SGMatrix<float64_t>();
        m_mean_vector = SGVector<float64_t>();
        m_eigenvalues_vector = SGVector<float64_t>();
	m_initialized = false;
}

SGMatrix<float64_t> CPCA::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	ASSERT(m_initialized)

	auto num_vectors = matrix.num_cols;
	auto num_features = matrix.num_rows;

	SG_INFO("Transforming feature matrix\n")
	Map<MatrixXd> transform_matrix(m_transformation_matrix.matrix,
			m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);

	if (m_mem_mode == MEM_IN_PLACE)
	{
		SG_INFO("Preprocessing feature matrix\n")
		Map<MatrixXd> feature_matrix(matrix.matrix, num_features, num_vectors);
		VectorXd data_mean =
		    feature_matrix.rowwise().sum() / (float64_t)num_vectors;
		feature_matrix = feature_matrix.colwise() - data_mean;

		feature_matrix.block(0, 0, num_dim, num_vectors) =
		    transform_matrix.transpose() * feature_matrix;

		SG_INFO("Form matrix of target dimension\n")
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

		SG_INFO("Preprocessing feature matrix\n")
		Map<MatrixXd> feature_matrix(matrix.matrix, num_features, num_vectors);
		VectorXd data_mean =
		    feature_matrix.rowwise().sum() / (float64_t)num_vectors;
		feature_matrix = feature_matrix.colwise() - data_mean;

		ret_matrix = transform_matrix.transpose() * feature_matrix;

		return ret;
	}
}

SGVector<float64_t> CPCA::apply_to_feature_vector(SGVector<float64_t> vector)
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

SGMatrix<float64_t> CPCA::get_transformation_matrix()
{
	return m_transformation_matrix;
}

SGVector<float64_t> CPCA::get_eigenvalues()
{
	return m_eigenvalues_vector;
}

SGVector<float64_t> CPCA::get_mean()
{
	return m_mean_vector;
}

EPCAMemoryMode CPCA::get_memory_mode() const
{
	return m_mem_mode;
}

void CPCA::set_memory_mode(EPCAMemoryMode e)
{
	m_mem_mode = e;
}

void CPCA::set_eigenvalue_zero_tolerance(float64_t eigenvalue_zero_tolerance)
{
	m_eigenvalue_zero_tolerance = eigenvalue_zero_tolerance;
}

float64_t CPCA::get_eigenvalue_zero_tolerance() const
{
	return m_eigenvalue_zero_tolerance;
}

void CPCA::set_target_dim(int32_t dim)
{
	ASSERT(dim > 0)
	m_target_dim = dim;
}

int32_t CPCA::get_target_dim() const
{
	return m_target_dim;
}
