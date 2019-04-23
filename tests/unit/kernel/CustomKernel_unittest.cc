/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Viktor Gal, Liang Pang, Soumyajit De,
 *          Thoralf Klein, Fernando Iglesias, Pan Deng
 */

#include <gtest/gtest.h>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/IndexFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;
using namespace Eigen;

TEST(CustomKernelTest,add_row_subset)
{
	index_t seed = 17;
	index_t m=3;

	std::mt19937_64 prng(seed);
	auto gen=std::make_shared<MeanShiftDataGenerator>(0, 2);
	auto feats=gen->get_streamed_features(m);

	auto gauss=std::make_shared<GaussianKernel>(10, 3);
	gauss->init(feats, feats);
	auto custom=std::make_shared<CustomKernel>(gauss);

	SGVector<index_t> inds(m);
	inds.range_fill();

	index_t num_runs=10;
	for (index_t i=0; i<num_runs; ++i)
	{
		random::shuffle(inds, prng);

		feats->add_subset(inds);
		custom->add_row_subset(inds);
		custom->add_col_subset(inds);
		gauss->init(feats, feats); // to make sure digonal is fine

		SGMatrix<float64_t> gauss_matrix=gauss->get_kernel_matrix();
		SGMatrix<float64_t> custom_matrix=custom->get_kernel_matrix();
//		gauss_matrix.display_matrix("gauss");
//		gauss_matrix.display_matrix("custom");
		for (index_t j=0; j<m*m; ++j)
			EXPECT_LE(Math::abs(gauss_matrix.matrix[j]-custom_matrix.matrix[j]), 1E-6);

		feats->remove_subset();
		custom->remove_row_subset();
		custom->remove_col_subset();
	}
}

TEST(CustomKernelTest,add_row_subset_constructor)
{
	index_t seed = 17;
	index_t n=4;
	std::mt19937_64 prng(seed);
	auto gen=std::make_shared<MeanShiftDataGenerator>(1, 2, 0);
	auto feats=
			gen->get_streamed_features(n)->as<DenseFeatures<float64_t>>();
	auto gaussian=std::make_shared<GaussianKernel>(feats, feats, 2, 10);
	auto main_kernel=std::make_shared<CustomKernel>(gaussian);
	gen->put("seed", seed);

	/* create custom kernel copy of gaussien and assert equalness */
	SGMatrix<float64_t> kmg=gaussian->get_kernel_matrix();
	SGMatrix<float64_t> km=main_kernel->get_kernel_matrix();
	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_LE(Math::abs(kmg(i, j)-km(i, j)), 1E-7);
	}

	/* create copy of custom kernel and assert equalness */
	auto copy=std::make_shared<CustomKernel>(km);
	SGMatrix<float64_t> kmc=copy->get_kernel_matrix();
	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_EQ(km(i, j), kmc(i, j));
	}

	/* add a subset to the custom kernel, create copy, create another kernel
	 * from this, assert equalness */
	SGVector<index_t> inds(n);
	inds.range_fill();
	random::shuffle(inds, prng);
	main_kernel->add_row_subset(inds);
	SGMatrix<float64_t> main_subset_matrix=main_kernel->get_kernel_matrix();
	main_kernel->remove_row_subset();
	auto main_subset_copy=std::make_shared<CustomKernel>(main_subset_matrix);
	SGMatrix<float64_t> main_subset_copy_matrix=main_subset_copy->get_kernel_matrix();
	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_EQ(main_subset_matrix(i, j), main_subset_copy_matrix(i, j));
	}






}

//Generate the Data
void generate_data(SGMatrix<float64_t> &data)
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
}

TEST(CustomKernelTest,index_features_subset)
{
	index_t seed = 17;
	float64_t epsilon=1e-7;
	index_t n=5;

	std::mt19937_64 prng(seed);

	// Generate the features
	SGMatrix<float64_t> data(3,n);
	generate_data(data);
	auto feats=std::make_shared<DenseFeatures<float64_t>>(data);


	// Generate the Kernels
	auto gaussian=std::make_shared<GaussianKernel>(feats, feats, 2, 10);

	auto main_kernel=std::make_shared<CustomKernel>(gaussian);


	/* create custom kernel copy of gaussien and assert equalness */
	SGMatrix<float64_t> kmg=gaussian->get_kernel_matrix();
	SGMatrix<float64_t> km=main_kernel->get_kernel_matrix();

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_NEAR(kmg(i, j), km(i, j), epsilon);
	}

	/* add a subset to the custom kernel, create copy, create another kernel
	 * from this, assert equalness */
	SGVector<index_t> r_idx(n);
	SGVector<index_t> c_idx(n);
	r_idx.range_fill();
	random::shuffle(r_idx, prng);
	c_idx.range_fill();
	random::shuffle(c_idx, prng);

	/* Create IndexFeatures instances */
	auto feat_r_idx = std::make_shared<IndexFeatures>(r_idx);
	auto feat_c_idx = std::make_shared<IndexFeatures>(c_idx);



	main_kernel->init(feat_r_idx,feat_c_idx);
	SGMatrix<float64_t> main_subset_matrix = main_kernel->get_kernel_matrix();

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_NEAR(main_subset_matrix(i, j), kmg(r_idx[i], c_idx[j]), epsilon);
	}






}

TEST(CustomKernelTest, sum_symmetric_block)
{
	const index_t m=17;
	const index_t n=31;
	const index_t d=3;

	// use fix seed for random data generation
	srand(100);

	// create feature X
	SGMatrix<float64_t> data_p(d, m);
	Map<MatrixXd> data_pm(data_p.matrix, data_p.num_rows, data_p.num_cols);
	data_pm=MatrixXd::Random(d, m);
	auto feats_p=std::make_shared<DenseFeatures<float64_t>>(data_p);

	// create feature Y
	SGMatrix<float64_t> data_q(d, n);
	Map<MatrixXd> data_qm(data_q.matrix, data_q.num_rows, data_q.num_cols);
	data_qm=MatrixXd::Random(d, n);
	auto feats_q=std::make_shared<DenseFeatures<float64_t>>(data_q);

	auto merged_feats=
		feats_p->create_merged_copy(feats_q)->as<DenseFeatures<float64_t>>();
	auto kernel=std::make_shared<GaussianKernel>(merged_feats, merged_feats, 2);
	auto precomputed_kernel=std::make_shared<CustomKernel>(kernel);

	// first block - sum(K(X, X')), X!=X' (no diag)
	float64_t sum1=kernel->sum_symmetric_block(0, m);
	float64_t sum2=precomputed_kernel->sum_symmetric_block(0, m);

	EXPECT_NEAR(sum1, sum2, 1E-4);

	// first block - sum(K(X, X')) (with diag)
	sum1=kernel->sum_symmetric_block(0, m, false);
	sum2=precomputed_kernel->sum_symmetric_block(0, m, false);

	EXPECT_NEAR(sum1, sum2, 1E-4);

	// second block - sum(K(Y, Y')), Y!=Y' (no diag)
	sum1=kernel->sum_symmetric_block(m, n);
	sum2=precomputed_kernel->sum_symmetric_block(m, n);

	EXPECT_NEAR(sum1, sum2, 1E-4);

	// second block - sum(K(Y, Y')) (with diag)
	sum1=kernel->sum_symmetric_block(m, n, false);
	sum2=precomputed_kernel->sum_symmetric_block(m, n, false);

	EXPECT_NEAR(sum1, sum2, 1E-4);

	// overall - sum(K(X.Y, X'.Y')) X!=X' && Y!=Y' (no diag)
	sum1=kernel->sum_symmetric_block(0, m+n);
	sum2=precomputed_kernel->sum_symmetric_block(0, m+n);

	EXPECT_NEAR(sum1, sum2, 1E-3);

	// overall - sum(K(X.Y, X'.Y')) (with diag)
	sum1=kernel->sum_symmetric_block(0, m+n, false);
	sum2=precomputed_kernel->sum_symmetric_block(0, m+n, false);

	EXPECT_NEAR(sum1, sum2, 1E-3);





}

TEST(CustomKernelTest, sum_block)
{
	const index_t m=17;
	const index_t n=31;
	const index_t d=3;

	// use fix seed for random data generation
	srand(100);

	// create feature X
	SGMatrix<float64_t> data_p(d, m);
	Map<MatrixXd> data_pm(data_p.matrix, data_p.num_rows, data_p.num_cols);
	data_pm=MatrixXd::Random(d, m);
	auto feats_p=std::make_shared<DenseFeatures<float64_t>>(data_p);

	// create feature Y
	SGMatrix<float64_t> data_q(d, n);
	Map<MatrixXd> data_qm(data_q.matrix, data_q.num_rows, data_q.num_cols);
	data_qm=MatrixXd::Random(d, n);
	auto feats_q=std::make_shared<DenseFeatures<float64_t>>(data_q);

	auto merged_feats=
		feats_p->create_merged_copy(feats_q)->as<DenseFeatures<float64_t>>();
	auto kernel=std::make_shared<GaussianKernel>(merged_feats, merged_feats, 2);
	auto precomputed_kernel=std::make_shared<CustomKernel>(kernel);

	// block - sum(K(X, Y))
	float64_t sum1=kernel->sum_block(0, m, m, n);
	float64_t sum2=precomputed_kernel->sum_block(0, m, m, n);

	EXPECT_NEAR(sum1, sum2, 1E-3);

	float64_t sum3=kernel->sum_block(m, 0, n, m);
	float64_t sum4=precomputed_kernel->sum_block(m, 0, n, m);

	EXPECT_NEAR(sum3, sum4, 1E-3);
	EXPECT_NEAR(sum1, sum3, 1E-3);





}

TEST(CustomKernelTest, sum_block_no_diag)
{
	const index_t m=17;
	const index_t n=17;
	const index_t d=3;

	// use fix seed for random data generation
	srand(100);

	// create feature X
	SGMatrix<float64_t> data_p(d, m);
	Map<MatrixXd> data_pm(data_p.matrix, data_p.num_rows, data_p.num_cols);
	data_pm=MatrixXd::Random(d, m);
	auto feats_p=std::make_shared<DenseFeatures<float64_t>>(data_p);

	// create feature Y
	SGMatrix<float64_t> data_q(d, n);
	Map<MatrixXd> data_qm(data_q.matrix, data_q.num_rows, data_q.num_cols);
	data_qm=MatrixXd::Random(d, n);
	auto feats_q=std::make_shared<DenseFeatures<float64_t>>(data_q);

	auto merged_feats=
		feats_p->create_merged_copy(feats_q)->as<DenseFeatures<float64_t>>();
	auto kernel=std::make_shared<GaussianKernel>(merged_feats, merged_feats, 2);
	auto precomputed_kernel=std::make_shared<CustomKernel>(kernel);

	// block - sum(K(X, Y)) (no diag)
	float64_t sum1=kernel->sum_block(0, m, m, n, true);
	float64_t sum2=precomputed_kernel->sum_block(0, m, m, n, true);

	EXPECT_NEAR(sum1, sum2, 1E-4);

	float64_t sum3=kernel->sum_block(m, 0, n, m, true);
	float64_t sum4=precomputed_kernel->sum_block(m, 0, n, m, true);

	EXPECT_NEAR(sum3, sum4, 1E-4);
	EXPECT_NEAR(sum1, sum3, 1E-4);





}

TEST(CustomKernelTest, row_wise_sum_symmetric_block)
{
	const index_t m=17;
	const index_t n=31;
	const index_t d=3;

	// use fix seed for random data generation
	srand(100);

	// create feature X
	SGMatrix<float64_t> data_p(d, m);
	Map<MatrixXd> data_pm(data_p.matrix, data_p.num_rows, data_p.num_cols);
	data_pm=MatrixXd::Random(d, m);
	auto feats_p=std::make_shared<DenseFeatures<float64_t>>(data_p);

	// create feature Y
	SGMatrix<float64_t> data_q(d, n);
	Map<MatrixXd> data_qm(data_q.matrix, data_q.num_rows, data_q.num_cols);
	data_qm=MatrixXd::Random(d, n);
	auto feats_q=std::make_shared<DenseFeatures<float64_t>>(data_q);

	auto merged_feats=
		feats_p->create_merged_copy(feats_q)->as<DenseFeatures<float64_t>>();
	auto kernel=std::make_shared<GaussianKernel>(merged_feats, merged_feats, 2);
	auto precomputed_kernel=std::make_shared<CustomKernel>(kernel);

	// first block - sum(K(X, X')), X!=X' (no diag)
	SGVector<float64_t> sum1=kernel->row_wise_sum_symmetric_block(0, m);
	SGVector<float64_t> sum2=precomputed_kernel->row_wise_sum_symmetric_block(0, m);

	EXPECT_EQ(sum1.vlen, sum2.vlen);
	for (index_t i=0; i<sum1.vlen; ++i)
		EXPECT_NEAR(sum1[i], sum2[i], 1E-5);

	// first block - sum(K(X, X')) (with diag)
	sum1=kernel->row_wise_sum_symmetric_block(0, m, false);
	sum2=precomputed_kernel->row_wise_sum_symmetric_block(0, m, false);

	EXPECT_EQ(sum1.vlen, sum2.vlen);
	for (index_t i=0; i<sum1.vlen; ++i)
		EXPECT_NEAR(sum1[i], sum2[i], 1E-5);

	// second block - sum(K(Y, Y')), Y!=Y' (no diag)
	sum1=kernel->row_wise_sum_symmetric_block(m, n);
	sum2=precomputed_kernel->row_wise_sum_symmetric_block(m, n);

	EXPECT_EQ(sum1.vlen, sum2.vlen);
	for (index_t i=0; i<sum1.vlen; ++i)
		EXPECT_NEAR(sum1[i], sum2[i], 1E-5);

	// second block - sum(K(Y, Y')) (with diag)
	sum1=kernel->row_wise_sum_symmetric_block(m, n, false);
	sum2=precomputed_kernel->row_wise_sum_symmetric_block(m, n, false);

	EXPECT_EQ(sum1.vlen, sum2.vlen);
	for (index_t i=0; i<sum1.vlen; ++i)
		EXPECT_NEAR(sum1[i], sum2[i], 1E-5);

	// overall - sum(K(X.Y, X'.Y')) X!=X' && Y!=Y' (no diag)
	sum1=kernel->row_wise_sum_symmetric_block(0, m+n);
	sum2=precomputed_kernel->row_wise_sum_symmetric_block(0, m+n);

	EXPECT_EQ(sum1.vlen, sum2.vlen);
	for (index_t i=0; i<sum1.vlen; ++i)
		EXPECT_NEAR(sum1[i], sum2[i], 1E-5);

	// overall - sum(K(X.Y, X'.Y')) (with diag)
	sum1=kernel->row_wise_sum_symmetric_block(0, m+n, false);
	sum2=precomputed_kernel->row_wise_sum_symmetric_block(0, m+n, false);

	EXPECT_EQ(sum1.vlen, sum2.vlen);
	for (index_t i=0; i<sum1.vlen; ++i)
		EXPECT_NEAR(sum1[i], sum2[i], 1E-5);





}

TEST(CustomKernelTest, row_wise_sum_squared_sum_symmetric_block)
{
	const index_t m=17;
	const index_t n=31;
	const index_t d=3;

	// use fix seed for random data generation
	srand(100);

	// create feature X
	SGMatrix<float64_t> data_p(d, m);
	Map<MatrixXd> data_pm(data_p.matrix, data_p.num_rows, data_p.num_cols);
	data_pm=MatrixXd::Random(d, m);
	auto feats_p=std::make_shared<DenseFeatures<float64_t>>(data_p);

	// create feature Y
	SGMatrix<float64_t> data_q(d, n);
	Map<MatrixXd> data_qm(data_q.matrix, data_q.num_rows, data_q.num_cols);
	data_qm=MatrixXd::Random(d, n);
	auto feats_q=std::make_shared<DenseFeatures<float64_t>>(data_q);

	auto merged_feats=
		feats_p->create_merged_copy(feats_q)->as<DenseFeatures<float64_t>>();
	auto kernel=std::make_shared<GaussianKernel>(merged_feats, merged_feats, 2);
	auto precomputed_kernel=std::make_shared<CustomKernel>(kernel);

	SGMatrix<float64_t> sum1, sum2;

	// first block - sum(K(X, X')), X!=X' (no diag)
	sum1=kernel->row_wise_sum_squared_sum_symmetric_block(0, m);
	sum2=precomputed_kernel->row_wise_sum_squared_sum_symmetric_block(0, m);

	EXPECT_EQ(sum1.num_rows, sum2.num_rows);
	EXPECT_EQ(sum1.num_cols, sum2.num_cols);
	for (index_t i=0; i<sum1.num_rows; ++i)
	{
		EXPECT_NEAR(sum1(i,0), sum2(i,0), 1E-5);
		EXPECT_NEAR(sum1(i,1), sum2(i,1), 1E-5);
	}

	// first block - sum(K(X, X')) (with diag)
	sum1=kernel->row_wise_sum_squared_sum_symmetric_block(0, m, false);
	sum2=precomputed_kernel->row_wise_sum_squared_sum_symmetric_block(0, m, false);

	EXPECT_EQ(sum1.num_rows, sum2.num_rows);
	EXPECT_EQ(sum1.num_cols, sum2.num_cols);
	for (index_t i=0; i<sum1.num_rows; ++i)
	{
		EXPECT_NEAR(sum1(i,0), sum2(i,0), 1E-5);
		EXPECT_NEAR(sum1(i,1), sum2(i,1), 1E-5);
	}

	// second block - sum(K(Y, Y')), Y!=Y' (no diag)
	sum1=kernel->row_wise_sum_squared_sum_symmetric_block(m, n);
	sum2=precomputed_kernel->row_wise_sum_squared_sum_symmetric_block(m, n);

	EXPECT_EQ(sum1.num_rows, sum2.num_rows);
	EXPECT_EQ(sum1.num_cols, sum2.num_cols);
	for (index_t i=0; i<sum1.num_rows; ++i)
	{
		EXPECT_NEAR(sum1(i,0), sum2(i,0), 1E-5);
		EXPECT_NEAR(sum1(i,1), sum2(i,1), 1E-5);
	}

	// second block - sum(K(Y, Y')) (with diag)
	sum1=kernel->row_wise_sum_squared_sum_symmetric_block(m, n, false);
	sum2=precomputed_kernel->row_wise_sum_squared_sum_symmetric_block(m, n, false);

	EXPECT_EQ(sum1.num_rows, sum2.num_rows);
	EXPECT_EQ(sum1.num_cols, sum2.num_cols);
	for (index_t i=0; i<sum1.num_rows; ++i)
	{
		EXPECT_NEAR(sum1(i,0), sum2(i,0), 1E-5);
		EXPECT_NEAR(sum1(i,1), sum2(i,1), 1E-5);
	}

	// overall - sum(K(X.Y, X'.Y')) X!=X' && Y!=Y' (no diag)
	sum1=kernel->row_wise_sum_squared_sum_symmetric_block(0, m+n);
	sum2=precomputed_kernel->row_wise_sum_squared_sum_symmetric_block(0, m+n);

	EXPECT_EQ(sum1.num_rows, sum2.num_rows);
	EXPECT_EQ(sum1.num_cols, sum2.num_cols);
	for (index_t i=0; i<sum1.num_rows; ++i)
	{
		EXPECT_NEAR(sum1(i,0), sum2(i,0), 1E-5);
		EXPECT_NEAR(sum1(i,1), sum2(i,1), 1E-5);
	}

	// overall - sum(K(X.Y, X'.Y')) (with diag)
	sum1=kernel->row_wise_sum_squared_sum_symmetric_block(0, m+n, false);
	sum2=precomputed_kernel->row_wise_sum_squared_sum_symmetric_block(0, m+n, false);

	EXPECT_EQ(sum1.num_rows, sum2.num_rows);
	EXPECT_EQ(sum1.num_cols, sum2.num_cols);
	for (index_t i=0; i<sum1.num_rows; ++i)
	{
		EXPECT_NEAR(sum1(i,0), sum2(i,0), 1E-5);
		EXPECT_NEAR(sum1(i,1), sum2(i,1), 1E-5);
	}





}

TEST(CustomKernelTest, row_col_wise_sum_block)
{
	const index_t m=17;
	const index_t n=31;
	const index_t d=3;

	// use fix seed for random data generation
	srand(100);

	// create feature X
	SGMatrix<float64_t> data_p(d, m);
	Map<MatrixXd> data_pm(data_p.matrix, data_p.num_rows, data_p.num_cols);
	data_pm=MatrixXd::Random(d, m);
	auto feats_p=std::make_shared<DenseFeatures<float64_t>>(data_p);

	// create feature Y
	SGMatrix<float64_t> data_q(d, n);
	Map<MatrixXd> data_qm(data_q.matrix, data_q.num_rows, data_q.num_cols);
	data_qm=MatrixXd::Random(d, n);
	auto feats_q=std::make_shared<DenseFeatures<float64_t>>(data_q);

	auto merged_feats=
		feats_p->create_merged_copy(feats_q)->as<DenseFeatures<float64_t>>();
	auto kernel=std::make_shared<GaussianKernel>(merged_feats, merged_feats, 2);
	auto precomputed_kernel=std::make_shared<CustomKernel>(kernel);

	// block - sum(K(X, Y))
	SGVector<float64_t> sum1=kernel->row_col_wise_sum_block(0, m, m, n);
	SGVector<float64_t> sum2=precomputed_kernel->row_col_wise_sum_block(0, m, m, n);

	EXPECT_EQ(sum1.vlen, sum2.vlen);
	for (index_t i=0; i<sum1.vlen; ++i)
		EXPECT_NEAR(sum1[i], sum2[i], 1E-5);

	SGVector<float64_t> sum3=kernel->row_col_wise_sum_block(m, 0, n, m);
	SGVector<float64_t> sum4=precomputed_kernel->row_col_wise_sum_block(m, 0, n, m);

	EXPECT_EQ(sum3.vlen, sum4.vlen);
	for (index_t i=0; i<sum3.vlen; ++i)
		EXPECT_NEAR(sum3[i], sum4[i], 1E-5);

	for (index_t i=0; i<m; ++i)
		EXPECT_NEAR(sum1[i], sum3[i+n], 1E-5);

	for (index_t i=0; i<n; ++i)
		EXPECT_NEAR(sum1[i+m], sum3[i], 1E-5);





}

TEST(CustomKernelTest, row_col_wise_sum_block_no_diag)
{
	const index_t m=17;
	const index_t n=17;
	const index_t d=3;

	// use fix seed for random data generation
	srand(100);

	// create feature X
	SGMatrix<float64_t> data_p(d, m);
	Map<MatrixXd> data_pm(data_p.matrix, data_p.num_rows, data_p.num_cols);
	data_pm=MatrixXd::Random(d, m);
	auto feats_p=std::make_shared<DenseFeatures<float64_t>>(data_p);

	// create feature Y
	SGMatrix<float64_t> data_q(d, n);
	Map<MatrixXd> data_qm(data_q.matrix, data_q.num_rows, data_q.num_cols);
	data_qm=MatrixXd::Random(d, n);
	auto feats_q=std::make_shared<DenseFeatures<float64_t>>(data_q);

	auto merged_feats=
		feats_p->create_merged_copy(feats_q)->as<DenseFeatures<float64_t>>();
	auto kernel=std::make_shared<GaussianKernel>(merged_feats, merged_feats, 2);
	auto precomputed_kernel=std::make_shared<CustomKernel>(kernel);

	// block - sum(K(X, Y))
	SGVector<float64_t> sum1=kernel->row_col_wise_sum_block(0, m, m, n, true);
	SGVector<float64_t> sum2=precomputed_kernel->row_col_wise_sum_block(0, m, m, n, true);

	EXPECT_EQ(sum1.vlen, sum2.vlen);
	for (index_t i=0; i<sum1.vlen; ++i)
		EXPECT_NEAR(sum1[i], sum2[i], 1E-5);

	SGVector<float64_t> sum3=kernel->row_col_wise_sum_block(m, 0, n, m, true);
	SGVector<float64_t> sum4=precomputed_kernel->row_col_wise_sum_block(m, 0, n, m, true);

	EXPECT_EQ(sum3.vlen, sum4.vlen);
	for (index_t i=0; i<sum3.vlen; ++i)
		EXPECT_NEAR(sum3[i], sum4[i], 1E-5);

	for (index_t i=0; i<m; ++i)
		EXPECT_NEAR(sum1[i], sum3[i+n], 1E-5);

	for (index_t i=0; i<n; ++i)
		EXPECT_NEAR(sum1[i+m], sum3[i], 1E-5);





}
