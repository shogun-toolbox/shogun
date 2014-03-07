#include <shogun/preprocessor/WhiteningPreprocessor.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/preprocessor/DensePreprocessor.h>


#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;
using namespace shogun;

CWhiteningPreprocessor::CWhiteningPreprocessor()
{

}

CWhiteningPreprocessor::~CWhiteningPreprocessor()
{

}

bool CWhiteningPreprocessor::init(CFeatures* features)
{
	ASSERT(features->get_feature_class()==C_DENSE)
	ASSERT(features->get_feature_type()==F_DREAL)

	return true;
}

SGVector<float64_t> CWhiteningPreprocessor::apply_to_feature_vector(SGVector<float64_t> vector)
{

}

SGMatrix<float64_t> CWhiteningPreprocessor::apply_to_feature_matrix(CFeatures* features)
{
	SGMatrix<float64_t> feature_matrix = ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	int rows = feature_matrix.num_rows;
	int cols = feature_matrix.num_cols;

	Map<MatrixXd> Efeature_matrix(feature_matrix.matrix,rows,cols);
	VectorXd mean = Efeature_matrix.rowwise().sum() / (float64_t)cols;
	MatrixXd mean_normalized_feature_matrix = Efeature_matrix.colwise() - mean;

	MatrixXd cov_matrix = (mean_normalized_feature_matrix * mean_normalized_feature_matrix.transpose()) / (float64_t)cols;

	SelfAdjointEigenSolver<MatrixXd> eigen(cov_matrix);

	VectorXd eigen_values = eigen.eigenvalues();

	MatrixXd D = eigen_values.cwiseSqrt().cwiseInverse().asDiagonal();
	MatrixXd W = D*eigen.eigenvectors();

	MatrixXd whitenedMatrix = W*Efeature_matrix;

	return SGMatrix<float64_t>(whitenedMatrix.data(),rows,cols);

}

// clean up allocated memory
void CWhiteningPreprocessor::cleanup()
{
}

/// initialize preprocessor from file
bool CWhiteningPreprocessor::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CWhiteningPreprocessor::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}


#endif
