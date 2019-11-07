#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <cstdio>
#include <random>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_PROTOBUF
#include <shogun/io/ProtobufFile.h>

TEST(ProtobufFileTest, vector_int32)
{
	int32_t seed = 100;
	int32_t len=1024*1024;
	SGVector<int32_t> data(len);

	std::mt19937_64 prng(seed);
  	UniformIntDistribution<int32_t> uniform_int_dist(0, len);

	for (int32_t i=0; i<len; i++)
		data[i]=(int32_t) uniform_int_dist(prng);

	auto fout=std::make_shared<ProtobufFile>("ProtobufFileTest_vector_int32_output.txt",'w');
	fout->set_vector(data.vector, len);
	fout.reset();

	SGVector<int32_t> data_from_file(true);
	auto fin=std::make_shared<ProtobufFile>("ProtobufFileTest_vector_int32_output.txt",'r');
	fin->get_vector(data_from_file.vector, data_from_file.vlen);
	EXPECT_EQ(data_from_file.vlen, len);

	for (int32_t i=0; i<data_from_file.vlen; i++)
	{
		EXPECT_EQ(data_from_file[i], data[i]);
	}
	unlink("ProtobufFileTest_vector_int32_output.txt");
}

TEST(ProtobufFileTest, vector_float64)
{
	int32_t seed = 100;
	int32_t len=1024*1024;
	SGVector<float64_t> data(len);

	std::mt19937_64 prng(seed);
  	UniformRealDistribution<float64_t> uniform_real_dist(0., 1.);

	for (int32_t i=0; i<len; i++)
		data[i]=(float64_t) uniform_real_dist(prng);

	auto fout=std::make_shared<ProtobufFile>("ProtobufFileTest_vector_float64_output.txt",'w');
	fout->set_vector(data.vector, len);
	fout.reset();

	SGVector<float64_t> data_from_file(true);
	auto fin=std::make_shared<ProtobufFile>("ProtobufFileTest_vector_float64_output.txt",'r');
	fin->get_vector(data_from_file.vector, data_from_file.vlen);
	EXPECT_EQ(data_from_file.vlen, len);

	for (int32_t i=0; i<data_from_file.vlen; i++)
	{
		EXPECT_NEAR(data_from_file[i], data[i], 1E-14);
	}

	unlink("ProtobufFileTest_vector_float64_output.txt");
}

TEST(ProtobufFileTest, matrix_int32)
{
	int32_t seed = 100;
	int32_t num_rows=1024;
	int32_t num_cols=512;
	SGMatrix<int32_t> data(num_rows, num_cols);

	std::mt19937_64 prng(seed);
  	UniformIntDistribution<int32_t> uniform_int_dist(0, num_rows);

	for (int32_t i=0; i<num_rows; i++)
	{
		for (int32_t j=0; j<num_cols; j++)
			data(i, j)=uniform_int_dist(prng);
	}


	auto fout=std::make_shared<ProtobufFile>("ProtobufFileTest_matrix_int32_output.txt",'w');
	fout->set_matrix(data.matrix, num_cols, num_rows);
	fout.reset();

	SGMatrix<int32_t> data_from_file(true);
	auto fin=std::make_shared<ProtobufFile>("ProtobufFileTest_matrix_int32_output.txt",'r');
	fin->get_matrix(data_from_file.matrix, data_from_file.num_cols, data_from_file.num_rows);
	EXPECT_EQ(data_from_file.num_rows, num_rows);
	EXPECT_EQ(data_from_file.num_cols, num_cols);

	for (int32_t i=0; i<num_rows; i++)
	{
		for (int32_t j=0; j<num_cols; j++)
			EXPECT_EQ(data_from_file(i, j), data(i, j));
	}

	unlink("ProtobufFileTest_matrix_int32_output.txt");
}

TEST(ProtobufFileTest, matrix_float64)
{
	int32_t seed = 100;
	int32_t num_rows=1024;
	int32_t num_cols=512;
	SGMatrix<float64_t> data(num_rows, num_cols);

	std::mt19937_64 prng(seed);
  	UniformRealDistribution<float64_t> uniform_real_dist(0., 1.);

	for (int32_t i=0; i<num_rows; i++)
	{
		for (int32_t j=0; j<num_cols; j++)
			data(i, j)=(float64_t) uniform_real_dist(prng);
	}

	auto fout=std::make_shared<ProtobufFile>("ProtobufFileTest_matrix_float64_output.txt",'w');
	fout->set_matrix(data.matrix, num_cols, num_rows);
	fout.reset();

	SGMatrix<float64_t> data_from_file(true);
	auto fin=std::make_shared<ProtobufFile>("ProtobufFileTest_matrix_float64_output.txt",'r');
	fin->get_matrix(data_from_file.matrix, data_from_file.num_cols, data_from_file.num_rows);
	EXPECT_EQ(data_from_file.num_rows, num_rows);
	EXPECT_EQ(data_from_file.num_cols, num_cols);

	for (int32_t i=0; i<num_rows; i++)
	{
		for (int32_t j=0; j<num_cols; j++)
			EXPECT_NEAR(data_from_file(i, j), data(i, j), 1E-14);
	}

	unlink("ProtobufFileTest_matrix_float64_output.txt");
}

TEST(ProtobufFileTest, sparse_matrix_int32)
{
	int32_t seed = 100;
	int32_t max_num_entries=512;
	int32_t max_entry_value=1024;
	int32_t num_vec=1024;
	int32_t num_feat=0;

	SGSparseVector<int32_t>* data=SG_MALLOC(SGSparseVector<int32_t>, num_vec);
	std::mt19937_64 prng(seed);
  	UniformIntDistribution<int32_t> uniform_int_dist(0, max_entry_value);
	for (int32_t i=0; i<num_vec; i++)
	{
		data[i]=SGSparseVector<int32_t>(uniform_int_dist(prng));
		for (int32_t j=0; j<data[i].num_feat_entries; j++)
		{
			int32_t feat_index=(j+1)*2;
			if (feat_index>num_feat)
				num_feat=feat_index;

			data[i].features[j].feat_index=feat_index-1;
			data[i].features[j].entry=uniform_int_dist(prng);
		}
	}

	int32_t num_vec_from_file=0;
	int32_t num_feat_from_file=0;
	SGSparseVector<int32_t>* data_from_file;

	auto fout=std::make_shared<ProtobufFile>("ProtobufFileTest_sparse_matrix_int32_output.txt", 'w');
	fout->set_sparse_matrix(data, num_feat, num_vec);
	fout.reset();

	auto fin=std::make_shared<ProtobufFile>("ProtobufFileTest_sparse_matrix_int32_output.txt", 'r');
	fin->get_sparse_matrix(data_from_file, num_feat_from_file, num_vec_from_file);

	EXPECT_EQ(num_vec_from_file, num_vec);
	EXPECT_EQ(num_feat_from_file, num_feat);
	for (int32_t i=0; i<num_vec; i++)
	{
		for (int32_t j=0; j<data[i].num_feat_entries; j++)
		{
			EXPECT_EQ(data[i].features[j].feat_index,
					data_from_file[i].features[j].feat_index);

			EXPECT_EQ(data[i].features[j].entry,
					data_from_file[i].features[j].entry);
		}
	}

	SG_FREE(data);
	SG_FREE(data_from_file);

	unlink("ProtobufFileTest_sparse_matrix_int32_output.txt");
}

TEST(ProtobufFileTest, sparse_matrix_float64)
{
	int32_t seed = 100;
	int32_t max_num_entries=512;
	int32_t num_vec=1024;
	int32_t num_feat=0;

	SGSparseVector<float64_t>* data=SG_MALLOC(SGSparseVector<float64_t>, num_vec);
	std::mt19937_64 prng(seed);
  	UniformIntDistribution<int32_t> uniform_int_dist(0, max_num_entries);
  	UniformRealDistribution<float64_t> uniform_real_dist(0., 1.);
	for (int32_t i=0; i<num_vec; i++)
	{
		data[i]=SGSparseVector<float64_t>(uniform_int_dist(prng));
		for (int32_t j=0; j<data[i].num_feat_entries; j++)
		{
			int32_t feat_index=(j+1)*2;
			if (feat_index>num_feat)
				num_feat=feat_index;

			data[i].features[j].feat_index=feat_index-1;
			data[i].features[j].entry=uniform_real_dist(prng);
		}
	}

	int32_t num_vec_from_file=0;
	int32_t num_feat_from_file=0;
	SGSparseVector<float64_t>* data_from_file;

	auto fout=std::make_shared<ProtobufFile>("ProtobufFileTest_sparse_matrix_float64_output.txt", 'w');
	fout->set_sparse_matrix(data, num_feat, num_vec);
	fout.reset();

	auto fin=std::make_shared<ProtobufFile>("ProtobufFileTest_sparse_matrix_float64_output.txt", 'r');
	fin->get_sparse_matrix(data_from_file, num_feat_from_file, num_vec_from_file);

	EXPECT_EQ(num_vec_from_file, num_vec);
	EXPECT_EQ(num_feat_from_file, num_feat);
	for (int32_t i=0; i<num_vec; i++)
	{
		for (int32_t j=0; j<data[i].num_feat_entries; j++)
		{
			EXPECT_EQ(data[i].features[j].feat_index,
					data_from_file[i].features[j].feat_index);

			EXPECT_NEAR(data[i].features[j].entry,
					data_from_file[i].features[j].entry, 1E-14);
		}
	}

	SG_FREE(data);
	SG_FREE(data_from_file);

	unlink("ProtobufFileTest_sparse_matrix_float64_output.txt");
}

TEST(ProtobufFileTest, DISABLED_string_list_char)
{
	int32_t seed = 100;
	int32_t num_str=1024;
	int32_t max_string_len=1024;

	SGVector<char>* strings=SG_MALLOC(SGVector<char>, num_str);
	std::mt19937_64 prng(seed);
  	UniformIntDistribution<int32_t> uniform_int_dist;
	for (int32_t i=0; i<num_str; i++)
	{
		strings[i]=SGVector<char>(uniform_int_dist(prng, {1, max_string_len}));
		for (int32_t j=0; j<strings[i].vlen; j++)
			strings[i].vector[j]=(char) uniform_int_dist(prng, {0, 255});
	}

	auto fout=std::make_shared<ProtobufFile>("ProtobufFileTest_string_list_char_output.txt",'w');
	fout->set_string_list(strings, num_str);
	fout.reset();

	SGVector<char>* data_from_file=NULL;
	int32_t num_str_from_file=0;
	int32_t max_string_len_from_file=0;
	auto fin=std::make_shared<ProtobufFile>("ProtobufFileTest_string_list_char_output.txt",'r');
	fin->get_string_list(data_from_file, num_str_from_file, max_string_len_from_file);
	EXPECT_EQ(num_str_from_file, num_str);

	for (int32_t i=0; i<num_str; i++)
	{
		for (int32_t j=0; j<strings[i].vlen; j++)
			EXPECT_EQ(strings[i].vector[j], data_from_file[i].vector[j]);
	}

	SG_FREE(strings);
	SG_FREE(data_from_file);
	unlink("ProtobufFileTest_string_list_char_output.txt");
}

#endif /* HAVE_PROTOBUF */
