#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
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

CFisherLDA::CFisherLDA (EFLDAMethod method, float64_t thresh):
	CDimensionReductionPreprocessor()
{
	init();
	m_method=method;
	m_threshold=thresh;
}

void CFisherLDA::init()
{
	m_method=AUTO_FLDA;
	m_threshold=0.01;
	m_num_dim=0;
	SG_ADD(&m_method, "FLDA_method","method for performing FLDA", 
			MS_NOT_AVAILABLE);
	SG_ADD(&m_num_dim, "final_dimensions","dimensions to be retained",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_transformation_matrix, "transformation_matrix","Transformation"
			" matrix (Eigenvectors of covariance matrix).", MS_NOT_AVAILABLE);
	SG_ADD(&m_mean_vector, "mean_vector", "Mean Vector.", MS_NOT_AVAILABLE);
	SG_ADD(&m_eigenvalues_vector, "eigenvalues_vector",
			"Vector with Eigenvalues.", MS_NOT_AVAILABLE);
}

CFisherLDA::~CFisherLDA()
{
}

bool CFisherLDA::init (CFeatures *features, CLabels *labels, int32_t num_dimensions)
{
	REQUIRE(features, "Features are not provided!\n")

	REQUIRE(features->get_feature_class()==C_DENSE,
			"LDA only works with dense features. you provided %s\n", 
			features->get_name());
	
	REQUIRE(features->get_feature_type()==F_DREAL,
			"LDA only works with real features.\n");
	
	REQUIRE(labels, "Labels for the given features are not specified!\n")
	
	REQUIRE(labels->get_label_type()==LT_MULTICLASS, "The labels should be of "
			"the type MulticlassLabels! you provided %s\n", labels->get_name());

	SGMatrix<float64_t> feature_matrix=((CDenseFeatures<float64_t>*)features)
										->get_feature_matrix();
	
	SGVector<float64_t> labels_vector=((CMulticlassLabels*)labels)->get_labels();

	int32_t num_vectors=feature_matrix.num_cols;
	int32_t num_features=feature_matrix.num_rows;

	REQUIRE(labels_vector.vlen==num_vectors,"The number of samples provided (%d)"
			" must be equal to the number of labels provided(%d)\n",num_vectors,
			labels_vector.vlen);

	// C holds the number of unique classes.
	int32_t C=((CMulticlassLabels*)labels)->get_num_classes();

	REQUIRE(C>1, "At least two classes are needed to perform LDA.\n")
	
	int32_t i=0;
	int32_t j=0;

	m_num_dim=num_dimensions;
	m_num_old_dim=num_features;
	// max target dimension allowed.
	// int32_t max_dim_allowed=C-1;

	// clip number if Dimensions to be a valid number
	if ((m_num_dim<=0) || (m_num_dim>(C-1)))
		m_num_dim=(C-1);

	MatrixXd fmatrix=Map<MatrixXd>(feature_matrix.matrix, num_features,
									num_vectors);
	Map<VectorXd> lvector(labels_vector.vector, num_vectors);

	// holds the total mean
	m_mean_vector=SGVector<float64_t>(num_features);
	Map<VectorXd>mean_total (m_mean_vector.vector, num_features);
	mean_total=VectorXd::Zero(num_features);
	// holds the mean for each class
	vector<VectorXd> mean_class(C);

	// holds the frequency for each class.
	// i.e the i'th element holds the number
	// of times class i is observed.
	VectorXd num_class=VectorXd::Zero(C);

	// calculate the class means and the total means.
	for (i=0; i<C; i++)
	{
		mean_class[i]=VectorXd::Zero(num_features);
		for (j=0; j<num_vectors; j++)
		{
			if (i==lvector[j])
			{
				num_class[i]++;
				mean_class[i]+=fmatrix.col(j);
			}
		}
		mean_class[i]/=(float64_t)num_class[i];
		mean_total+=mean_class[i];
	}
	mean_total/=(float64_t)C;

	// Subtract the class means from the 'respective' data.
	// e.g all data belonging to class 0 is subtracted by
	// the mean of class 0 data.
	for (i=0; i<C; i++)
		for (j=0; j<num_vectors; j++)
			if (i==lvector[j])
				fmatrix.col(j)-=mean_class[i];

	if ((m_method==CANVAR_FLDA) || 
			(m_method==AUTO_FLDA && num_vectors<num_features))
	{
		// holds the  fmatrix for each class
		vector<MatrixXd> centered_class_i(C);
		VectorXd temp=num_class;
		MatrixXd Sw=MatrixXd::Zero(num_features, num_features);
		for (i=0; i<C; i++)
		{
			centered_class_i[i]=MatrixXd::Zero(num_features, num_class[i]);
			for (j=0; j<num_vectors; j++)
				if (i==lvector[j])
					centered_class_i[i].col(num_class[i]-(temp[i]--))
						=fmatrix.col(j);
			Sw+=(centered_class_i[i]*centered_class_i[i].transpose())
				*num_class[i]/(float64_t)(num_class[i]-1);
		}

		// within class matrix for cannonical variates implementation
		MatrixXd Sb(num_features, C);
		for (i=0; i<C; i++)
		Sb.col(i)=sqrt(num_class[i])*(mean_class[i]-mean_total);

		MatrixXd fmatrix1=Map<MatrixXd>(feature_matrix.matrix, num_features,
									num_vectors);

		JacobiSVD<MatrixXd> svd(fmatrix1, ComputeThinU | ComputeThinV);
		// basis to represent the solution
		MatrixXd Q;

		if(num_features>num_vectors)
		{	
			j=0;
			for (i=0;i<num_vectors;i++)
				if (svd.singularValues()(i)>m_threshold)
					j++;
				else
					break;
			Q=svd.matrixU().leftCols(j);
		}
		else 
			Q=svd.matrixU();
	  
		// Sb is the modified between scatter
		Sb=(Q.transpose())*Sb*(Sb.transpose())*Q;
		// Sw is the modified within scatter
		Sw=Q.transpose()*Sw*Q;

		// to find SVD((inverse(Chol(Sw)))' * Sb * (inverse(Chol(Sw))))
		//1.get Cw=Chol(Sw)
		//find the decomposition of Cw'
		HouseholderQR<MatrixXd> decomposition(Sw.llt().matrixU().transpose());
		//2.get P=inv(Cw')*Sb
		//MatrixXd P=decomposition.solve(Sb);
		//3. final value to be put in SVD will be therefore:
		// final_ output = (inv(Cw')*(P'))';
		//MatrixXd X_final_chol=(decomposition.solve(P.transpose())).transpose();
		JacobiSVD<MatrixXd> svd2(decomposition.solve
				(decomposition.solve(Sb).transpose()).transpose(),ComputeThinU);
		m_transformation_matrix=SGMatrix<float64_t> (num_features, m_num_dim);
		Map<MatrixXd> eigenVectors(m_transformation_matrix.matrix, num_features,
									m_num_dim);
		
		eigenVectors=Q*(svd2.matrixU()).leftCols(m_num_dim);

		m_eigenvalues_vector=SGVector<float64_t>(m_num_dim);
		Map<VectorXd> eigenValues (m_eigenvalues_vector.vector, m_num_dim);
		eigenValues=svd2.singularValues().topRows(m_num_dim);
	}

	else
	{
		// For holding the within class scatter.
		MatrixXd Sw=fmatrix*fmatrix.transpose();

		// For holding the between class scatter.
		MatrixXd Sb(num_features, C);

		for (i=0; i<C; i++)
			Sb.col(i)=mean_class[i];

		Sb=Sb-mean_total.rowwise().replicate(C);
		Sb=Sb*Sb.transpose();

		// calculate the Ax=b problem
		// where A=Sw
		// b=Sb
		// x=M
		// MatrixXd M=Sw.householderQr().solve(Sb);
		// calculate the eigenvalues and eigenvectors of M.
		EigenSolver<MatrixXd> es(Sw.householderQr().solve(Sb));

		MatrixXd all_eigenvectors=es.eigenvectors().real();
		VectorXd all_eigenvalues=es.eigenvalues().real();

		std::vector<pair<float64_t, int32_t> > data(num_features);
		for (i=0; i<num_features; i++)
		{
			data[i].first=all_eigenvalues[i];
			data[i].second=i;
		}
		// sort the eigenvalues.
		std::sort (data.begin(), data.end());

		// keep 'm_num_dim' numbers of top Eigenvalues
		m_eigenvalues_vector=SGVector<float64_t> (m_num_dim);
		Map<VectorXd> eigenValues(m_eigenvalues_vector.vector, m_num_dim);

		// keep 'm_num_dim' numbers of EigenVectors
		// corresponding to their respective eigenvalues
		m_transformation_matrix=SGMatrix<float64_t> (num_features, m_num_dim);
		Map<MatrixXd> eigenVectors(m_transformation_matrix.matrix, num_features,
									m_num_dim);

		for (i=0; i<m_num_dim; i++)
		{
			eigenValues[i]=data[num_features-i-1].first;
			eigenVectors.col(i)=all_eigenvectors.col(data[num_features-i-1].second);
		}
	}
	return true;
}

void CFisherLDA::cleanup()
{
	m_transformation_matrix=SGMatrix<float64_t>();
	m_mean_vector=SGVector<float64_t>();
	m_eigenvalues_vector=SGVector<float64_t>();
}

SGMatrix<float64_t> CFisherLDA::apply_to_feature_matrix(CFeatures*features)
{
	REQUIRE(features->get_feature_class()==C_DENSE,
			"LDA only works with dense features\n");
	
	REQUIRE(features->get_feature_type()==F_DREAL,
			"LDA only works with real features\n");
	
	SGMatrix<float64_t> m=((CDenseFeatures<float64_t>*)
							features)->get_feature_matrix();
	
	int32_t num_vectors=m.num_cols;
	int32_t num_features=m.num_rows;

	SG_INFO("Transforming feature matrix\n")
	Map<MatrixXd> transform_matrix(m_transformation_matrix.matrix,
			m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);

	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features)

	Map<MatrixXd> feature_matrix (m.matrix, num_features, num_vectors);
	
	feature_matrix.block (0, 0, m_num_dim, num_vectors)=
			transform_matrix.transpose()*feature_matrix;

	SG_INFO("Form matrix of target dimension") 
	for (int32_t col=0; col<num_vectors; col++)
	{
		for (int32_t row=0; row<m_num_dim; row++)
			m[col*m_num_dim+row]=feature_matrix(row, col);
	}
	m.num_rows=m_num_dim;
	m.num_cols=num_vectors;
	((CDenseFeatures<float64_t>*)features)->set_feature_matrix(m);
	return m;
}

SGVector<float64_t> CFisherLDA::apply_to_feature_vector(SGVector<float64_t> vector)
{	
	SGVector<float64_t> result = SGVector<float64_t>(m_num_dim);
	Map<VectorXd> resultVec(result.vector, m_num_dim);
	Map<VectorXd> inputVec(vector.vector, vector.vlen);

	Map<VectorXd> mean(m_mean_vector.vector, m_mean_vector.vlen);
	Map<MatrixXd> transformMat(m_transformation_matrix.matrix,
		m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);

	resultVec=transformMat.transpose()*inputVec;
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
