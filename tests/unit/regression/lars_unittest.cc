/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 Parijat Mazumdar
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
* either expressed or implied, of the Shogun Development Team.
*/

#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/labels/RegressionLabels.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/NormOne.h>

using namespace Eigen;
using namespace shogun;

#ifdef HAVE_CXX11
//Generate the Data that N is greater than D
void generate_data_n_greater_d(SGMatrix<float64_t> &data, SGVector<float64_t> &lab)
{
	data(0,0)=0.044550005575722;
	data(1,0)=-0.433969606728583;
	data(2,0)=-0.397935396933392;
	data(0,1)=-0.778754072066602;
	data(1,1)=-0.620105076569903;
	data(2,1)=-0.542538248707627;
	data(0,2)=0.334313094513960;
	data(1,2)=0.421985645755003;
	data(2,2)=0.263031426076997;
	data(0,3)=0.516043376162584;
	data(1,3)=0.159041471773470;
	data(2,3)=0.691318725364356;
	data(0,4)=-0.116152404185664;
	data(1,4)=0.473047565770014;
	data(2,4)=-0.013876505800334;

	lab[0]=-0.196155100498902;
	lab[1]=-5.376485285422094;
	lab[2]=-1.717489351713958;
	lab[3]=4.506538567065213;
	lab[4]=2.783591170569741;
}

//Generate the Data that N is less than D
void generate_data_n_less_d(SGMatrix<float64_t> &data, SGVector<float64_t> &lab)
{
	data(0,0)=0.217778502400306;
	data(1,0)=0.660755393455389;
	data(2,0)=0.492143169178889;
	data(3,0)=0.668618163874328;
	data(4,0)=0.806098163441828;
	data(0,1)=-0.790379818206924;
	data(1,1)=-0.745771163834136;
	data(2,1)=-0.810293460958058;
	data(3,1)=-0.740156729710306;
	data(4,1)=-0.515540473266151;
	data(0,2)=0.572601315806618;
	data(1,2)=0.085015770378747;
	data(2,2)=0.318150291779169;
	data(3,2)=0.071538565835978;
	data(4,2)=-0.290557690175677;

	lab[0]=3.771471612608209;
	lab[1]=-3.218048715328546;
	lab[2]=-0.553422897279663;
}

TEST(LeastAngleRegression, lasso_n_greater_than_d)
{
	SGMatrix<float64_t> data(3,5);
	SGVector<float64_t> lab(5);
	generate_data_n_greater_d(data, lab);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	SG_REF(features);
	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);
	CLeastAngleRegression* lars=new CLeastAngleRegression();
	lars->set_labels((CLabels*) labels);
	lars->train(features);

	SGVector<float64_t> active3=SGVector<float64_t>(lars->get_w_for_var(3));
	SGVector<float64_t> active2=SGVector<float64_t>(lars->get_w_for_var(2));
	SGVector<float64_t> active1=SGVector<float64_t>(lars->get_w_for_var(1));

	float64_t epsilon=0.000000000001;
	EXPECT_NEAR(active3[0],2.911072069591,epsilon);
	EXPECT_NEAR(active3[1],1.290672330338,epsilon);
	EXPECT_NEAR(active3[2],2.208741384416,epsilon);

	EXPECT_NEAR(active2[0],1.747958837898,epsilon);
	EXPECT_NEAR(active2[1],0.000000000000,epsilon);
	EXPECT_NEAR(active2[2],1.840553057519,epsilon);

	EXPECT_NEAR(active1[0],0.000000000000,epsilon);
	EXPECT_NEAR(active1[1],0.000000000000,epsilon);
	EXPECT_NEAR(active1[2],0.092594219621,epsilon);

	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(LeastAngleRegression, lasso_n_less_than_d)
{
	SGMatrix<float64_t> data(5,3);
	SGVector<float64_t> lab(3);
	generate_data_n_less_d(data,lab);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	SG_REF(features);
	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);
	CLeastAngleRegression* lars=new CLeastAngleRegression();
	lars->set_labels(labels);
	lars->train(features);

	SGVector<float64_t> active2=SGVector<float64_t>(lars->get_w_for_var(2));
	SGVector<float64_t> active1=SGVector<float64_t>(lars->get_w_for_var(1));

	float64_t epsilon=0.000000000001;
	EXPECT_NEAR(active1[0],0.000000000000,epsilon);
	EXPECT_NEAR(active1[1],0.000000000000,epsilon);
	EXPECT_NEAR(active1[2],0.000000000000,epsilon);
	EXPECT_NEAR(active1[3],0.039226231353,epsilon);
	EXPECT_NEAR(active1[4],0.000000000000,epsilon);

	EXPECT_NEAR(active2[0],0.000000000000,epsilon);
	EXPECT_NEAR(active2[1],0.000000000000,epsilon);
	EXPECT_NEAR(active2[2],0.000000000000,epsilon);
	EXPECT_NEAR(active2[3],2.578863294056,epsilon);
	EXPECT_NEAR(active2[4],2.539637062702,epsilon);

	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(LeastAngleRegression, lars_n_greater_than_d)
{
	SGMatrix<float64_t> data(3,5);
	SGVector<float64_t> lab(5);
	generate_data_n_greater_d(data, lab);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	SG_REF(features);
	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);
	CLeastAngleRegression* lars=new CLeastAngleRegression(false);
	lars->set_labels((CLabels*) labels);
	lars->train(features);

	SGVector<float64_t> active3=SGVector<float64_t>(lars->get_w_for_var(3));
	SGVector<float64_t> active2=SGVector<float64_t>(lars->get_w_for_var(2));
	SGVector<float64_t> active1=SGVector<float64_t>(lars->get_w_for_var(1));

	float64_t epsilon=0.000000000001;
	EXPECT_NEAR(active3[0],2.911072069591,epsilon);
	EXPECT_NEAR(active3[1],1.290672330338,epsilon);
	EXPECT_NEAR(active3[2],2.208741384416,epsilon);

	EXPECT_NEAR(active2[0],1.747958837898,epsilon);
	EXPECT_NEAR(active2[1],0.000000000000,epsilon);
	EXPECT_NEAR(active2[2],1.840553057519,epsilon);

	EXPECT_NEAR(active1[0],0.000000000000,epsilon);
	EXPECT_NEAR(active1[1],0.000000000000,epsilon);
	EXPECT_NEAR(active1[2],0.092594219621,epsilon);

	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(LeastAngleRegression, lars_n_less_than_d)
{
	SGMatrix<float64_t> data(5,3);
	SGVector<float64_t> lab(3);
	generate_data_n_less_d(data,lab);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	SG_REF(features);
	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);
	CLeastAngleRegression* lars=new CLeastAngleRegression(false);
	lars->set_labels(labels);
	lars->train(features);

	SGVector<float64_t> active2=SGVector<float64_t>(lars->get_w_for_var(2));
	SGVector<float64_t> active1=SGVector<float64_t>(lars->get_w_for_var(1));

	float64_t epsilon=0.000000000001;
	EXPECT_NEAR(active1[0],0.000000000000,epsilon);
	EXPECT_NEAR(active1[1],0.000000000000,epsilon);
	EXPECT_NEAR(active1[2],0.000000000000,epsilon);
	EXPECT_NEAR(active1[3],0.039226231353,epsilon);
	EXPECT_NEAR(active1[4],0.000000000000,epsilon);

	EXPECT_NEAR(active2[0],0.000000000000,epsilon);
	EXPECT_NEAR(active2[1],0.000000000000,epsilon);
	EXPECT_NEAR(active2[2],0.000000000000,epsilon);
	EXPECT_NEAR(active2[3],2.578863294056,epsilon);
	EXPECT_NEAR(active2[4],2.539637062702,epsilon);

	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}

template <typename ST>
void lars_n_less_than_d_feature_test_templated()
{
	SGMatrix<float64_t> data_64(5,3);
	SGVector<float64_t> lab(3);
	generate_data_n_less_d(data_64,lab);

	//copy data_64 into a ST SGMatrix
	SGMatrix<ST> data(5,3);
	for(index_t c = 0; c < data_64.num_cols; ++c)
		for(index_t r = 0; r < data_64.num_rows; ++r)
			data(r, c) = (ST) data_64(r, c);

	CDenseFeatures<ST>* features = new CDenseFeatures<ST>(data);
	SG_REF(features);
	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);
	CLeastAngleRegression* lars=new CLeastAngleRegression(false);
	SG_REF(lars)

	lars->set_labels(labels);

	//Catch exceptions thrown when training, clean up
	try
	{
		lars->train(features);
	}
	catch(...)
	{
		SG_UNREF(lars);
		SG_UNREF(features);
		SG_UNREF(labels);

		throw;
	}

	SGVector<float64_t> active2 = lars->get_w_for_var(2);
	SGVector<float64_t> active1 = lars->get_w_for_var(1);

	float64_t epsilon=0.0001;
	EXPECT_NEAR(active1[0],0.000000000000,epsilon);
	EXPECT_NEAR(active1[1],0.000000000000,epsilon);
	EXPECT_NEAR(active1[2],0.000000000000,epsilon);
	EXPECT_NEAR(active1[3],0.039226231353,epsilon);
	EXPECT_NEAR(active1[4],0.000000000000,epsilon);

	EXPECT_NEAR(active2[0],0.000000000000,epsilon);
	EXPECT_NEAR(active2[1],0.000000000000,epsilon);
	EXPECT_NEAR(active2[2],0.000000000000,epsilon);
	EXPECT_NEAR(active2[3],2.578863294056,epsilon);
	EXPECT_NEAR(active2[4],2.539637062702,epsilon);

	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(LeastAngleRegression, lars_template_test_bool)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<bool>());
}

TEST(LeastAngleRegression, lars_template_test_char)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<char>());
}

TEST(LeastAngleRegression, lars_template_test_int8)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<int8_t>());
}

TEST(LeastAngleRegression, lars_template_test_unit8)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<uint8_t>());
}

TEST(LeastAngleRegression, lars_template_test_int16)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<int16_t>());
}

TEST(LeastAngleRegression, lars_template_test_uint16)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<uint16_t>());
}

TEST(LeastAngleRegression, lars_template_test_int32)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<int32_t>());
}

TEST(LeastAngleRegression, lars_template_test_uint32)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<uint32_t>());
}

TEST(LeastAngleRegression, lars_template_test_int64)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<int64_t>());
}

TEST(LeastAngleRegression, lars_template_test_uint64)
{
	EXPECT_ANY_THROW(lars_n_less_than_d_feature_test_templated<uint64_t>());
}

TEST(LeastAngleRegression, lars_template_test_float32)
{
	lars_n_less_than_d_feature_test_templated<float32_t>();
}

TEST(LeastAngleRegression, lars_template_test_float64)
{
	lars_n_less_than_d_feature_test_templated<float64_t>();
}

TEST(LeastAngleRegression, lars_template_test_floatmax)
{
	lars_n_less_than_d_feature_test_templated<floatmax_t>();
}

#ifndef USE_VIENNACL_GLOBAL
TEST(LeastAngleRegression, cholesky_insert)
{
	class lars_helper: public CLeastAngleRegression
	{
		public:
			SGMatrix<float64_t> cholesky_insert_helper(const SGMatrix<float64_t>& X,
					const SGMatrix<float64_t>& X_active, SGMatrix<float64_t>& R, int32_t i, int32_t n)
			{
				return CLeastAngleRegression::cholesky_insert<float64_t>(X, X_active, R, i, n);
			}
	};

	int32_t num_feats=5, num_vec=6;
	SGMatrix<float64_t> R(num_feats, num_feats);
	SGMatrix<float64_t> mat(num_vec, num_feats-1);
	SGMatrix<float64_t> matnew(num_vec, num_feats);
	
	SGVector<float64_t> vec(num_vec);
	vec.random(0.0,1.0);
	Map<VectorXd> map_vec(vec.vector, vec.size());	

	for (index_t i=0; i<num_vec; i++)
	{	
		for (index_t j=0; j<num_feats-1; j++)
		{
			mat(i,j)=CMath::random(0.0,1.0);
			matnew(i,j)=mat(i,j);
		}
	}
	for (index_t i=0 ; i<num_vec; i++)
		matnew(i, num_feats-1)=vec[i];

	Map<MatrixXd> mat_old(mat.matrix, num_vec, num_feats-1);
	Map<MatrixXd> mat_new(matnew.matrix, num_vec, num_feats);
	Map<MatrixXd> map_R(R.matrix, num_feats, num_feats);	
	
	MatrixXd XX=mat_old.transpose()*mat_old;
	// Compute matrix R which has to be updated
	SGMatrix<float64_t> R_old=linalg::cholesky(XX, false);

	// Update cholesky decomposition matrix R
	lars_helper lars = lars_helper();
	SGMatrix<float64_t> R_new = lars.cholesky_insert_helper(matnew, mat, R_old, 4, 4);
	Map<MatrixXd> map_R_new(R_new.matrix, R_new.num_rows, R_new.num_cols);	

	// Compute true cholesky decomposition		
	MatrixXd XX_new=mat_new.transpose()*mat_new;
	SGMatrix<float64_t> R_true=linalg::cholesky(XX_new, false);

	Map<MatrixXd> map_R_true(R_true.matrix, num_feats, num_feats);	
	EXPECT_NEAR( (map_R_true - map_R_new).norm(), 0.0, 1E-12);	

}

TEST(LeastAngleRegression, ols_equivalence)
{
	int32_t n_feat=25, n_vec=100;
	SGMatrix<float64_t> data(n_feat, n_vec);		
	for (index_t i=0; i<n_feat; i++)
	{	
		for (index_t j=0; j<n_vec; j++)
			data(i,j)=CMath::random(0.0,1.0);
	}
	
	SGVector<float64_t> lab=SGVector<float64_t>(n_vec);
	lab.random(0.0,1.0);
	float64_t mean=linalg::mean(lab);
	
	for (index_t i=0; i<lab.size(); i++)
		lab[i]-=mean;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	SG_REF(features);

	CPruneVarSubMean* proc1=new CPruneVarSubMean();
	CNormOne* proc2=new CNormOne();
	proc1->init(features);
	proc1->apply_to_feature_matrix(features);
	proc2->init(features);
	proc2->apply_to_feature_matrix(features);

	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);
	CLeastAngleRegression* lars=new CLeastAngleRegression(false);
	lars->set_labels((CLabels*) labels);
	lars->train(features);
	// Full LAR model
	SGVector<float64_t> w=lars->get_w();
	Map<VectorXd> map_w(w.vector, w.size());	

	SGMatrix<float64_t> mat=features->get_feature_matrix();
	Map<MatrixXd> feat(mat.matrix, mat.num_rows, mat.num_cols);
	Map<VectorXd> l(lab.vector, lab.size());
	// OLS
#if EIGEN_WITH_TRANSPOSITION_BUG
	MatrixXd feat_t = feat.transpose().eval();
	VectorXd solve=feat_t.colPivHouseholderQr().solve(l);
#else
	VectorXd solve=feat.transpose().colPivHouseholderQr().solve(l);
#endif

	// Check if full LAR model is equivalent to OLS
	EXPECT_EQ( w.size(), n_feat);
	EXPECT_NEAR( (map_w - solve).norm(), 0.0, 1E-12);
	

	SG_UNREF(proc1);
	SG_UNREF(proc2);
	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}
#endif

TEST(LeastAngleRegression, early_stop_l1_norm)
{
	SGMatrix<float64_t> data(3,5);
	SGVector<float64_t> lab(5);
	generate_data_n_greater_d(data, lab);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	SG_REF(features);
	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);
	CLeastAngleRegression* lars=new CLeastAngleRegression(false);
	lars->set_labels((CLabels*) labels);
	// set max l1 norm
	lars->set_max_l1_norm(1);
	lars->train(features);

	SGVector<float64_t> active2=SGVector<float64_t>(lars->get_w_for_var(2));
	SGVector<float64_t> active1=SGVector<float64_t>(lars->get_w_for_var(1));

	float64_t epsilon=0.000000000001;

	EXPECT_NEAR(active2[0],0.453702890189,epsilon);
	EXPECT_NEAR(active2[1],0.000000000000,epsilon);
	EXPECT_NEAR(active2[2],0.546297109810,epsilon);

	EXPECT_NEAR(active1[0],0.000000000000,epsilon);
	EXPECT_NEAR(active1[1],0.000000000000,epsilon);
	EXPECT_NEAR(active1[2],0.092594219621,epsilon);

	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}
#endif
