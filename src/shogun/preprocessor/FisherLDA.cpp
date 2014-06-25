#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/preprocessor/FisherLDA.h>
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace shogun;

CFisherLDA::CFisherLDA (int32_t num_components):
	CDimensionReductionPreprocessor()
{
	init();
	num_dim = num_components;
}

void CFisherLDA::init()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_mean_vector = SGVector<float64_t>();
	m_eigenvalues_vector = SGVector<float64_t>();
	num_dim = 0;
	
	SG_ADD(&m_transformation_matrix, "transformation_matrix",
		    "Transformation matrix (Eigenvectors of covariance matrix).",
		    MS_NOT_AVAILABLE);
	SG_ADD(&m_mean_vector, "mean_vector", "Mean Vector.", MS_NOT_AVAILABLE);
	SG_ADD(&m_eigenvalues_vector, "eigenvalues_vector",
		    "Vector with Eigenvalues.", MS_NOT_AVAILABLE);    
}

CFisherLDA::~CFisherLDA()
{
}

bool CFisherLDA::init (CFeatures *features, CLabels *labels)
{
	REQUIRE(features->get_feature_class()==C_DENSE, "LDA only works with dense features")
	REQUIRE(features->get_feature_type()==F_DREAL, "LDA only works with real features")

	SGMatrix<float64_t> feature_matrix = ((CDenseFeatures<float64_t> *)features)
	                                     ->get_feature_matrix();
	SGVector<float64_t> labels_vector = ((CMulticlassLabels *)labels)
	                                    ->get_labels();

	int32_t num_vectors = feature_matrix.num_cols;
	int32_t num_features = feature_matrix.num_rows;

	REQUIRE(num_vectors > num_features,"No. of Dimension should be less than no. of samples")
	REQUIRE(labels_vector.vlen==num_vectors, "The number of samples must be equal to the number of labels")

	// C holds the number of unique classes.
	int32_t C = ((CMulticlassLabels *) labels)->get_num_classes();

	REQUIRE(C>1,"At least two classes are needed to perform LDA.")

	num_old_dim = num_features;
	// max target dimension allowed.
	int32_t max_dim_allowed = C - 1;

	// clip number if Dimensions to be a valid number
	if ((num_dim <= 0) || (num_dim > (C - 1)))
		num_dim = (C - 1);

	MatrixXd fmatrix = Map<MatrixXd> (feature_matrix.matrix, num_features,
	                                  num_vectors);
	Map<VectorXd> lvector (labels_vector.vector, num_vectors);

	// holds the total mean
	m_mean_vector = SGVector<float64_t> (num_features);
	Map<VectorXd> mean_total (m_mean_vector.vector, num_features);

	// holds the mean for each class
	vector<VectorXd> mean_class (C);

	// holds the frequency for each class.
	// i.e the i'th element holds the number
	// of times class i is observed.
	VectorXd num_class = VectorXd::Zero (C);

	// calculate the class means and the total means.
	for (int i = 0; i < C; i++)
	{
		mean_class[i] = VectorXd::Zero (num_features);
		for (int j = 0; j < num_vectors; j++)
		{
			if (i == lvector[j])
			{
				num_class[i]++;
				mean_class[i] += fmatrix.col (j);
			}
		}

		mean_class[i] /= num_class[i];
		mean_total += mean_class[i];	
	}
	mean_total /= C;

	// Subtract the class means from the 'respective' data.
	//e.g all data belonging to class 0 is subtracted by
	//the mean of class 0 data.
	for (int i = 0; i < C; i++)
		for (int j = 0; j < num_vectors; j++)
			if (i == lvector[j])
				fmatrix.col (j) -= mean_class[i];

	// Calculate the within class scatter.
	MatrixXd Sw = fmatrix * fmatrix.transpose();

	// Calculate the between class scatter.
	//MatrixXd Sb(num_features, num_features);
	MatrixXd Sb (num_features, C);

	for (int i = 0; i < C; i++)
		Sb.col (i) = mean_class[i];

	Sb = Sb - mean_total.rowwise().replicate (C);
	Sb = Sb * Sb.transpose();

	// calculate the Ax=b problem
	// where A=Sw
	// b=Sb
	// x=M
	MatrixXd M = Sw.colPivHouseholderQr().solve (Sb);

	// calculate the eigenvalues and eigenvectors of M.
	EigenSolver<MatrixXd> es (M);

	MatrixXcd all_eigenvectors = es.eigenvectors();
	VectorXcd all_eigenvalues = es.eigenvalues();

	std::vector<pair<float64_t, int32_t> > data(num_features);
	for (int i = 0; i < num_features; i++)
	{
		data[i].first=all_eigenvalues[i].real();
		data[i].second=i;
	}
    // sort the eigenvalues.
	std::sort (data.begin(), data.end());

	// keep 'num_dim' numbers of top Eigenvalues
	m_eigenvalues_vector = SGVector<float64_t> (num_dim);
	Map<VectorXd> eigenValues (m_eigenvalues_vector.vector, num_dim);

	// keep 'num_dim' numbers of EigenVectors
	// corresponding to their respective eigenvalues
	m_transformation_matrix = SGMatrix<float64_t> (num_features, num_dim);
	Map<MatrixXd> eigenVectors (m_transformation_matrix.matrix, num_features,
	                            num_dim);

	for (int i = 0; i < num_dim; i++)
	{
		eigenValues[i] = data[num_features-i-1].first;
		eigenVectors.col (i) = all_eigenvectors.col
            (data[num_features-i-1].second).real();
	}

	return true;
}

void CFisherLDA::cleanup()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_mean_vector = SGVector<float64_t>();
	m_eigenvalues_vector = SGVector<float64_t>();
}

SGMatrix<float64_t> CFisherLDA::apply_to_feature_matrix (CFeatures *features)
{
	SGMatrix<float64_t> m = ((CDenseFeatures<float64_t> *)
	                         features)->get_feature_matrix();
	int32_t num_vectors = m.num_cols;
	int32_t num_features = m.num_rows;

	Map<MatrixXd> transform_matrix (m_transformation_matrix.matrix,
	         m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);
	
    Map<MatrixXd> feature_matrix (m.matrix, num_features, num_vectors);
	feature_matrix.block (0, 0, num_dim, num_vectors) =
	        transform_matrix.transpose() * feature_matrix;

	for (int32_t col = 0; col < num_vectors; col++)
	{
		for (int32_t row = 0; row < num_dim; row++)
			m.matrix[col * num_dim + row] = feature_matrix (row, col);
	}

	m.num_rows = num_dim;
	m.num_cols = num_vectors;

	((CDenseFeatures<float64_t> *) features)->set_feature_matrix (m);
	return m;
}

SGVector<float64_t> CFisherLDA::apply_to_feature_vector (
        SGVector<float64_t> vector)
{
	SGVector<float64_t> result = SGVector<float64_t> (num_dim);
	Map<VectorXd> resultVec (result.vector, num_dim);
	Map<VectorXd> inputVec (vector.vector, vector.vlen);

	Map<VectorXd> mean (m_mean_vector.vector, m_mean_vector.vlen);
	Map<MatrixXd> transformMat (m_transformation_matrix.matrix,
	                            m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);

	inputVec=inputVec-mean;
	resultVec=transformMat.transpose()*inputVec;
	inputVec=inputVec+mean;

	return result;
}

SGMatrix<float64_t> CFisherLDA::get_transformation_matrix()
{
	return m_transformation_matrix;
}

SGVector<float64_t> CFisherLDA::get_eigenvalues()
{
	return m_eigenvalues_vector;
}

SGVector<float64_t> CFisherLDA::get_mean()
{
	return m_mean_vector;
}
#endif//HAVE_EIGEN3
