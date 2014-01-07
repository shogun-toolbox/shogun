#include <features/RandomFourierDotFeatures.h>

#include <gtest/gtest.h>

using namespace shogun;

TEST(RandomFourierDotFeatures, dot_test)
{
	int32_t num_dims = 50;
	int32_t vecs = 5;
	int32_t D = 100;

	SGMatrix<float64_t> w(num_dims+1,D);

	for (index_t i=0; i<D; i++)
	{
		for (index_t j=0; j<num_dims; j++)
			w(j,i) = i+j;
		w(num_dims,i) = 1;
	}

	SGMatrix<int32_t> data(num_dims, vecs);
	for (index_t i=0; i<vecs; i++)
	{
		for (index_t j=0; j<num_dims; j++)
			data(j,i) = i+j;
	}

	CDenseFeatures<int32_t>* d_feats = new CDenseFeatures<int32_t>(data);
	SGVector<float64_t> params(1);
	params[0] = 8;
	CRandomFourierDotFeatures* r_feats = new CRandomFourierDotFeatures(
			d_feats, D, GAUSSIAN, params, w);

	SGMatrix<float64_t> cross_dot_matrix(vecs, vecs);
	for (index_t i=0; i<vecs; i++)
	{
		for (index_t j=0; j<vecs; j++)
		{
			cross_dot_matrix(i,j) = r_feats->dot(i,r_feats,j);
		}
	}

	float64_t e = 1e-13;
	float64_t mat []= {
		1.00555015129804581, 0.0502863828725347,
		0.00023246982925897, -0.02515703711183189,
		-0.00876801723029572, 0.05028638287253472,
		1.00104181673378067, 0.02521590366359591,
		-0.00354749195264035, -0.01140342190110809,
		0.00023246982925894, 0.02521590366359589,
		0.99726185495188157, 0.03896951887431968,
		-0.00074544906564935, -0.02515703711183188,
		-0.00354749195264038, 0.03896951887431966,
		1.00006389783887250, 0.02708505456315889,
		-0.00876801723029571, -0.01140342190110809,
		-0.00074544906564934, 0.02708505456315890,
		0.99501156944854796};

	SGMatrix<float64_t> precomputed_mat(mat, 5, 5, false);
	for (index_t i=0; i<5; i++)
	{
		for (index_t j=0; j<5; j++)
			EXPECT_NEAR(precomputed_mat(i,j), cross_dot_matrix(i,j), e);
	}
	SG_UNREF(r_feats);
}

TEST(RandomFourierDotFeatures, dense_dot_test)
{
	int32_t num_dims = 50;
	int32_t vecs = 5;
	int32_t D = 100;

	SGMatrix<float64_t> w(num_dims+1,D);

	for (index_t i=0; i<D; i++)
	{
		for (index_t j=0; j<num_dims; j++)
			w(j,i) = i+j;
		w(num_dims,i) = 1;
	}

	SGMatrix<int32_t> data(num_dims, vecs);
	for (index_t i=0; i<vecs; i++)
	{
		for (index_t j=0; j<num_dims; j++)
			data(j,i) = i+j;
	}

	CDenseFeatures<int32_t>* d_feats = new CDenseFeatures<int32_t>(data);
	SGVector<float64_t> params(1);
	params[0] = 8;
	CRandomFourierDotFeatures* r_feats = new CRandomFourierDotFeatures(
			d_feats, D, GAUSSIAN, params, w);

	SGMatrix<float64_t> cross_dot_matrix(vecs, vecs);
	for (index_t i=0; i<vecs; i++)
	{
		for (index_t j=0; j<vecs; j++)
		{
			cross_dot_matrix(i,j) = r_feats->dot(i,r_feats,j);
		}
	}

	float64_t e = 1e-13;
	float64_t vec[] = {0.0449317122413237, -0.2909428095069972, -0.0361875564777414, 0.1185535017124422,
					-0.0018001695930624};
	SGVector<float64_t> precomputed_vec(vec, 5, false);
	for (index_t i=0; i<5; i++)
	{
		SGVector<float64_t> ones(D);
		SGVector<float64_t>::fill_vector(ones.vector, ones.vlen, 1);
		float64_t dot = r_feats->dense_dot(i, ones.vector, ones.vlen);
		EXPECT_NEAR(dot, vec[i], e);
	}
	SG_UNREF(r_feats);
}

TEST(RandomFourierDotFeatures, add_to_dense_test)
{
	int32_t num_dims = 50;
	int32_t vecs = 5;
	int32_t D = 100;

	SGMatrix<float64_t> w(num_dims+1,D);

	for (index_t i=0; i<D; i++)
	{
		for (index_t j=0; j<num_dims; j++)
			w(j,i) = i+j;
		w(num_dims,i) = 1;
	}

	SGMatrix<int32_t> data(num_dims, vecs);
	for (index_t i=0; i<vecs; i++)
	{
		for (index_t j=0; j<num_dims; j++)
			data(j,i) = i+j;
	}

	CDenseFeatures<int32_t>* d_feats = new CDenseFeatures<int32_t>(data);
	SGVector<float64_t> params(1);
	params[0] = 8;
	CRandomFourierDotFeatures* r_feats = new CRandomFourierDotFeatures(
			d_feats, D, GAUSSIAN, params, w);

	SGMatrix<float64_t> cross_dot_matrix(vecs, vecs);
	for (index_t i=0; i<vecs; i++)
	{
		for (index_t j=0; j<vecs; j++)
		{
			cross_dot_matrix(i,j) = r_feats->dot(i,r_feats,j);
		}
	}

	float64_t e = 1e-13;
	float64_t vec[] = {0.0449317122413237, -0.2909428095069972, -0.0361875564777414, 0.1185535017124422,
					-0.0018001695930624};
	SGVector<float64_t> precomputed_vec(vec, 5, false);
	for (index_t i=0; i<5; i++)
	{
		SGVector<float64_t> zeros(D);
		SGVector<float64_t>::fill_vector(zeros.vector, zeros.vlen, 0);
		r_feats->add_to_dense_vec(1, i, zeros.vector, zeros.vlen, false);
		float64_t sum = 0;
		for (index_t j=0; j<D; j++)
			sum += zeros[j];
		EXPECT_NEAR(sum, vec[i], e);
	}
	SG_UNREF(r_feats);
}

