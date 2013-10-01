#include <shogun/io/LibSVMFile.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/Random.h>

#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

using namespace shogun;

TEST(LibSVMFileTest, sparse_matrix_int32)
{
	int32_t max_num_entries=512;
	int32_t max_label_value=1;
	int32_t max_entry_value=1024;
	CRandom* rand=new CRandom();

	int32_t num_vec=1024;
	int32_t num_feat=0;

	SGSparseVector<int32_t>* data=SG_MALLOC(SGSparseVector<int32_t>, num_vec);
	float64_t* labels=SG_MALLOC(float64_t, num_vec);
	for (int32_t i=0; i<num_vec; i++)
	{
		data[i]=SGSparseVector<int32_t>(rand->random(0, max_num_entries));
		labels[i]=(float64_t) rand->random(-max_label_value, max_label_value);
		for (int32_t j=0; j<data[i].num_feat_entries; j++)
		{
			int32_t feat_index=(j+1)*2;
			if (feat_index>num_feat)
				num_feat=feat_index;

			data[i].features[j].feat_index=feat_index-1;
			data[i].features[j].entry=rand->random(0, max_entry_value);
		}
	}

	int32_t num_vec_from_file=0;
	int32_t num_feat_from_file=0;
	SGSparseVector<int32_t>* data_from_file;
	float64_t* labels_from_file;

	CLibSVMFile* fout=new CLibSVMFile("LibSVMFileTest_sparse_matrix_int32_output.txt", 'w', NULL);
	fout->set_sparse_matrix(data, num_feat, num_vec, labels);
	SG_UNREF(fout);

	CLibSVMFile* fin=new CLibSVMFile("LibSVMFileTest_sparse_matrix_int32_output.txt", 'r', NULL);
	fin->get_sparse_matrix(data_from_file, num_feat_from_file, num_vec_from_file, 
				labels_from_file);

	EXPECT_EQ(num_vec_from_file, num_vec);
	EXPECT_EQ(num_feat_from_file, num_feat);
	for (int32_t i=0; i<num_vec; i++)
	{
		EXPECT_NEAR(labels[i], labels_from_file[i], 1E-14);
		for (int32_t j=0; j<data[i].num_feat_entries; j++)
		{
			EXPECT_EQ(data[i].features[j].feat_index, 
					data_from_file[i].features[j].feat_index);

			EXPECT_EQ(data[i].features[j].entry, 
					data_from_file[i].features[j].entry);
		}
	}
	SG_UNREF(fin);

	SG_UNREF(rand);
	SG_FREE(data);
	SG_FREE(labels);
	SG_FREE(data_from_file);
	SG_FREE(labels_from_file);

	unlink("LibSVMFileTest_sparse_matrix_int32_output.txt");
}

TEST(LibSVMFileTest, sparse_matrix_float64)
{
	int32_t max_num_entries=512;
	int32_t max_label_value=1;
	CRandom* rand=new CRandom();

	int32_t num_vec=1024;
	int32_t num_feat=0;

	SGSparseVector<float64_t>* data=SG_MALLOC(SGSparseVector<float64_t>, num_vec);
	float64_t* labels=SG_MALLOC(float64_t, num_vec);
	for (int32_t i=0; i<num_vec; i++)
	{
		data[i]=SGSparseVector<float64_t>(rand->random(0, max_num_entries));
		labels[i]=(float64_t) rand->random(-max_label_value, max_label_value);
		for (int32_t j=0; j<data[i].num_feat_entries; j++)
		{
			int32_t feat_index=(j+1)*2;
			if (feat_index>num_feat)
				num_feat=feat_index;

			data[i].features[j].feat_index=feat_index-1;
			data[i].features[j].entry=rand->random(0., 1.);
		}
	}

	int32_t num_vec_from_file=0;
	int32_t num_feat_from_file=0;
	SGSparseVector<float64_t>* data_from_file;
	float64_t* labels_from_file;

	CLibSVMFile* fout=new CLibSVMFile("LibSVMFileTest_sparse_matrix_float64_output.txt", 'w', NULL);
	fout->set_sparse_matrix(data, num_feat, num_vec, labels);
	SG_UNREF(fout);

	CLibSVMFile* fin=new CLibSVMFile("LibSVMFileTest_sparse_matrix_float64_output.txt", 'r', NULL);
	fin->get_sparse_matrix(data_from_file, num_feat_from_file, num_vec_from_file, 
				labels_from_file);

	EXPECT_EQ(num_vec_from_file, num_vec);
	EXPECT_EQ(num_feat_from_file, num_feat);
	for (int32_t i=0; i<num_vec; i++)
	{
		EXPECT_NEAR(labels[i], labels_from_file[i], 1E-14);
		for (int32_t j=0; j<data[i].num_feat_entries; j++)
		{
			EXPECT_EQ(data[i].features[j].feat_index, 
					data_from_file[i].features[j].feat_index);

			EXPECT_NEAR(data[i].features[j].entry, 
					data_from_file[i].features[j].entry, 1E-14);
		}
	}
	SG_UNREF(fin);

	SG_UNREF(rand);
	SG_FREE(data);
	SG_FREE(labels);
	SG_FREE(data_from_file);
	SG_FREE(labels_from_file);
	unlink("LibSVMFileTest_sparse_matrix_float64_output.txt");
}
