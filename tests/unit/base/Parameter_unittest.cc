/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soumyajit De, Bjoern Esser
 */

#include <shogun/base/Parameter.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/SparseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(TParameter,copy_ptype_BOOL)
{
	bool a=true;
	bool b=false;
	EXPECT_TRUE(TParameter::copy_ptype(PT_BOOL, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_CHAR)
{
	char a=1;
	char b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_CHAR, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_INT8)
{
	int8_t a=1;
	int8_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_INT8, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_UINT8)
{
	uint8_t a=1;
	uint8_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_UINT8, &a, &b));
	EXPECT_EQ(a, b);
}

TEST(TParameter,copy_ptype_INT16)
{
	int16_t a=1;
	int16_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_INT16, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_UINT16)
{
	uint16_t a=1;
	uint16_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_UINT16, &a, &b));
	EXPECT_EQ(a, b);
}

TEST(TParameter,copy_ptype_INT32)
{
	int32_t a=1;
	int32_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_INT32, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_UINT32)
{
	uint32_t a=1;
	uint32_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_UINT32, &a, &b));
	EXPECT_EQ(a, b);
}

TEST(TParameter,copy_ptype_INT64)
{
	int64_t a=1;
	int64_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_INT64, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_UINT64)
{
	uint64_t a=1;
	uint64_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_UINT64, &a, &b));
	EXPECT_EQ(a, b);
}

TEST(TParameter,copy_ptype_FLOAT32)
{
	float32_t a=1;
	float32_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_FLOAT32, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_FLOAT64)
{
	float64_t a=1;
	float64_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_FLOAT64, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_FLOATMAX)
{
	floatmax_t a=1;
	floatmax_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_FLOATMAX, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, 1);
}

TEST(TParameter,copy_ptype_COMPLEX128)
{
	complex128_t a(1.0, 1.0);
	complex128_t b(2.0, 2.0);
	EXPECT_TRUE(TParameter::copy_ptype(PT_COMPLEX128, &a, &b));
	EXPECT_EQ(a, b);
	EXPECT_EQ(b, complex128_t(1.0,1.0));
}

TEST(TParameter,copy_stype_NONE)
{
	int32_t a=1;
	int32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_TRUE(TParameter::copy_stype(ST_NONE, PT_INT32, param1->m_parameter, param2->m_parameter));
	EXPECT_EQ(a,b);
	EXPECT_EQ(b, 1);

	delete param1;
	delete param2;
}

TEST(TParameter,copy_target_null)
{
	int32_t a=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "", "");

	EXPECT_FALSE(param1->copy(NULL));

	delete param1;
}

TEST(TParameter,copy_different_name)
{
	int32_t a=1;
	int32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "a", "");
	TParameter* param2=new TParameter(&type, &b, "b", "");

	EXPECT_FALSE(param1->copy(param2));

	delete param1;
	delete param2;
}

TEST(TParameter,copy_own_parameter_null)
{
	int32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, NULL, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_FALSE(param1->copy(param2));

	delete param1;
	delete param2;
}

TEST(TParameter,copy_target_parameter_null)
{
	int32_t a=1;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, NULL, "", "");

	EXPECT_FALSE(param1->copy(param2));

	delete param1;
	delete param2;
}

TEST(TParameter,copy_SCALAR)
{
	int32_t a=1;
	int32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_TRUE(param1->copy(param2));
	EXPECT_EQ(a,b);
	EXPECT_EQ(b, 1);

	delete param1;
	delete param2;
}

TEST(TParameter,copy_VECTOR_SCALAR_same_size)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);

	a[0]=1;
	a[1]=1;
	b[0]=2;
	b[1]=2;

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "", "");
	TParameter* param2=new TParameter(&type, &b.vector, "", "");

	EXPECT_TRUE(param1->copy(param2));

	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_EQ(a[i], b[i]);
		EXPECT_EQ(a[i], 1);
	}

	delete param1;
	delete param2;
}

TEST(TParameter,copy_VECTOR_SCALAR_different_size)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(3);

	a[0]=1;
	a[1]=1;
	b[0]=2;
	b[1]=2;
	b[2]=2;

	TSGDataType type_a(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TSGDataType type_b(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &b.vlen);
	TParameter* param1=new TParameter(&type_a, &a.vector, "", "");
	TParameter* param2=new TParameter(&type_b, &b.vector, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(a.vlen, 2);
	EXPECT_EQ(b.vlen, a.vlen);
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_EQ(a[i], b[i]);
		EXPECT_EQ(a[i], 1);
	}

	delete param1;
	delete param2;
}

TEST(TParameter,copy_VECTOR_SCALAR_target_empty)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b;

	a[0]=1;
	a[1]=1;

	TSGDataType type_a(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TSGDataType type_b(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &b.vlen);
	TParameter* param1=new TParameter(&type_a, &a.vector, "", "");
	TParameter* param2=new TParameter(&type_b, &b.vector, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(a.vlen, 2);
	EXPECT_EQ(b.vlen, a.vlen);
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_EQ(a[i], b[i]);
		EXPECT_EQ(a[i], 1);
	}

	delete param1;
	delete param2;
}

TEST(TParameter,copy_VECTOR_SCALAR_source_and_target_empty)
{
	SGVector<float64_t> a;
	SGVector<float64_t> b;

	TSGDataType type_a(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TSGDataType type_b(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &b.vlen);
	TParameter* param1=new TParameter(&type_a, &a.vector, "", "");
	TParameter* param2=new TParameter(&type_b, &b.vector, "", "");

	EXPECT_TRUE(param1->copy(param2));
	EXPECT_EQ(a.vlen, 0);
	EXPECT_EQ(b.vlen, a.vlen);

	delete param1;
	delete param2;
}

TEST(TParameter,copy_MATRIX_SCALAR_source_and_target_empty)
{
	SGMatrix<float64_t> a;
	SGMatrix<float64_t> b;

	TSGDataType type_a(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TSGDataType type_b(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &b.num_rows, &b.num_cols);
	TParameter* param1=new TParameter(&type_a, &a.matrix, "", "");
	TParameter* param2=new TParameter(&type_b, &b.matrix, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(a.num_rows, 0);
	EXPECT_EQ(a.num_cols, 0);
	EXPECT_EQ(b.num_rows, a.num_rows);
	EXPECT_EQ(b.num_cols, a.num_cols);

	delete param1;
	delete param2;
}

TEST(TParameter,copy_MATRIX_SCALAR_target_empty)
{
	SGMatrix<float64_t> a(2,2);
	SGMatrix<float64_t> b;

	a(0,0)=1;
	a(0,1)=1;
	a(1,0)=1;
	a(1,1)=1;

	TSGDataType type_a(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TSGDataType type_b(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &b.num_rows, &b.num_cols);
	TParameter* param1=new TParameter(&type_a, &a.matrix, "", "");
	TParameter* param2=new TParameter(&type_b, &b.matrix, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(a.num_rows, 2);
	EXPECT_EQ(a.num_cols, 2);
	EXPECT_EQ(b.num_rows, a.num_rows);
	EXPECT_EQ(b.num_cols, a.num_cols);

	for (index_t i=0; i<a.num_rows*a.num_cols; ++i)
	{
		EXPECT_EQ(a.matrix[i], b.matrix[i]);
		EXPECT_EQ(a.matrix[i], 1);
	}

	delete param1;
	delete param2;
}

TEST(TParameter,copy_MATRIX_SCALAR_different_size)
{
	SGMatrix<float64_t> a(2,2);
	SGMatrix<float64_t> b(3,3);

	a(0,0)=1;
	a(0,1)=1;
	a(1,0)=1;
	a(1,1)=1;

	b(0,0)=2;
	b(0,1)=2;
	b(0,2)=2;
	b(1,0)=2;
	b(1,1)=2;
	b(1,2)=2;
	b(2,0)=2;
	b(2,1)=2;
	b(2,2)=2;

	TSGDataType type_a(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TSGDataType type_b(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &b.num_rows, &b.num_cols);
	TParameter* param1=new TParameter(&type_a, &a.matrix, "", "");
	TParameter* param2=new TParameter(&type_b, &b.matrix, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(a.num_rows, 2);
	EXPECT_EQ(a.num_cols, 2);
	EXPECT_EQ(b.num_rows, a.num_rows);
	EXPECT_EQ(b.num_cols, a.num_cols);

	for (index_t i=0; i<a.num_rows*a.num_cols; ++i)
	{
		EXPECT_EQ(a.matrix[i], b.matrix[i]);
		EXPECT_EQ(a.matrix[i], 1);
	}

	delete param1;
	delete param2;
}

TEST(TParameter,copy_MATRIX_SCALAR_same_size)
{
	SGMatrix<float64_t> a(2,2);
	SGMatrix<float64_t> b(2,2);

	a(0,0)=1;
	a(0,1)=1;
	a(1,0)=1;
	a(1,1)=1;

	b(0,0)=2;
	b(0,1)=2;
	b(1,0)=2;
	b(1,1)=2;

	TSGDataType type_a(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TSGDataType type_b(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &b.num_rows, &b.num_cols);
	TParameter* param1=new TParameter(&type_a, &a.matrix, "", "");
	TParameter* param2=new TParameter(&type_b, &b.matrix, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(a.num_rows, 2);
	EXPECT_EQ(a.num_cols, 2);
	EXPECT_EQ(b.num_rows, a.num_rows);
	EXPECT_EQ(b.num_cols, a.num_cols);

	for (index_t i=0; i<a.num_rows*a.num_cols; ++i)
	{
		EXPECT_EQ(a.matrix[i], b.matrix[i]);
		EXPECT_EQ(a.matrix[i], 1);
	}

	delete param1;
	delete param2;
}

TEST(TParameter,copy_STRING_SCALAR_same_length)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);

	a.set_const(1);
	b.set_const(2);

	SGString<float64_t> str1(a);
	SGString<float64_t> str2(b);

	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &str1.string, "", "");
	TParameter* param2=new TParameter(&type, &str2.string, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(a.vlen, 2);
	EXPECT_EQ(a.vlen, b.vlen);
	EXPECT_EQ(a[0], b[0]);
	EXPECT_EQ(a[1], b[1]);

	delete param1;
	delete param2;
}

TEST(TParameter,copy_STRING_SCALAR_different_length)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(3);

	a.set_const(1);
	b.set_const(2);

	SGString<float64_t> str1(a);
	SGString<float64_t> str2(b);

	TSGDataType type_a(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TSGDataType type_b(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TParameter* param1=new TParameter(&type_a, &str1.string, "", "");
	TParameter* param2=new TParameter(&type_b, &str2.string, "", "");

	EXPECT_TRUE(param1->copy(param2));

	/* to avoid memory errors when cleaning up, string is changed but vector
	 * still points to old array, which cannot be freed anymore. */
	b.vector=str2.string;
	b.vlen=str2.slen;

	EXPECT_EQ(str1.slen, str2.slen);
	EXPECT_EQ(str1.slen, 2);
	EXPECT_EQ(a[0], b[0]);
	EXPECT_EQ(a[1], b[1]);

	delete param1;
	delete param2;
}

TEST(TParameter,copy_STRING_SCALAR_target_empty)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b;

	a.set_const(1);

	SGString<float64_t> str1(a);
	SGString<float64_t> str2(b);

	TSGDataType type_a(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TSGDataType type_b(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TParameter* param1=new TParameter(&type_a, &str1.string, "", "");
	TParameter* param2=new TParameter(&type_b, &str2.string, "", "");

	EXPECT_TRUE(param1->copy(param2));

	/* to avoid memory errors when cleaning up, string is changed but vector
	 * still points to old array, which cannot be freed anymore. */
	b.vector=str2.string;
	b.vlen=str2.slen;

	EXPECT_EQ(str1.slen, str2.slen);
	EXPECT_EQ(str1.slen, 2);
	EXPECT_EQ(a[0], b[0]);
	EXPECT_EQ(a[1], b[1]);

	delete param1;
	delete param2;
}

TEST(TParameter,copy_STRING_SCALAR_source_and_target_empty)
{
	SGVector<float64_t> a;
	SGVector<float64_t> b;

	SGString<float64_t> str1(a);
	SGString<float64_t> str2(b);

	TSGDataType type_a(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TSGDataType type_b(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TParameter* param1=new TParameter(&type_a, &str1.string, "", "");
	TParameter* param2=new TParameter(&type_b, &str2.string, "", "");

	EXPECT_TRUE(param1->copy(param2));

	/* to avoid memory errors when cleaning up, string is changed but vector
	 * still points to old array, which cannot be freed anymore. */
	b.vector=str2.string;
	b.vlen=str2.slen;

	EXPECT_EQ(str1.slen, str2.slen);
	EXPECT_EQ(str1.slen, 0);

	delete param1;
	delete param2;
}

TEST(TParameter,copy_SPARSE_SCALAR_same_length)
{
	SGMatrix<float64_t> a(2,1);
	SGMatrix<float64_t> b(2,1);

	a.set_const(1);
	b.set_const(2);

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseVector<float64_t> vec1=s1->get_sparse_feature_vector(0);
	vec1.features[1].feat_index=2;
	SGSparseVector<float64_t> vec2=s2->get_sparse_feature_vector(0);

	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &vec1, "", "");
	TParameter* param2=new TParameter(&type, &vec2, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(vec1.num_feat_entries, vec2.num_feat_entries);
	EXPECT_EQ(vec1.num_feat_entries, 2);
	EXPECT_EQ(vec1.features[0].feat_index, vec2.features[0].feat_index);
	EXPECT_EQ(vec1.features[1].feat_index, vec2.features[1].feat_index);
	EXPECT_EQ(vec1.features[0].entry, vec2.features[0].entry);
	EXPECT_EQ(vec1.features[1].entry, vec2.features[1].entry);

	delete param1;
	delete param2;
	s1->free_sparse_feature_vector(0);
	s2->free_sparse_feature_vector(0);
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,copy_SPARSE_SCALAR_different_length)
{
	SGMatrix<float64_t> a(2,1);
	SGMatrix<float64_t> b(3,1);

	a.set_const(1);
	b.set_const(2);

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseVector<float64_t> vec1=s1->get_sparse_feature_vector(0);
	SGSparseVector<float64_t> vec2=s2->get_sparse_feature_vector(0);

	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &vec1, "", "");
	TParameter* param2=new TParameter(&type, &vec2, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(vec1.num_feat_entries, vec2.num_feat_entries);
	EXPECT_EQ(vec2.num_feat_entries, 2);
	EXPECT_EQ(vec1.features[0].feat_index, vec2.features[0].feat_index);
	EXPECT_EQ(vec1.features[1].feat_index, vec2.features[1].feat_index);
	EXPECT_EQ(vec1.features[0].entry, vec2.features[0].entry);
	EXPECT_EQ(vec1.features[1].entry, vec2.features[1].entry);

	delete param1;
	delete param2;
	s1->free_sparse_feature_vector(0);
	s2->free_sparse_feature_vector(0);
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,copy_SPARSE_SCALAR_target_empty)
{
	SGMatrix<float64_t> a(2,1);
	SGMatrix<float64_t> b(3,1);

	a.set_const(1);
	b.set_const(2);

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseVector<float64_t> vec1=s1->get_sparse_feature_vector(0);
	SGSparseVector<float64_t> vec2=s2->get_sparse_feature_vector(0);
	void* temp=vec2.features;
	vec2.features=NULL;
	vec2.num_feat_entries=0;

	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &vec1, "", "");
	TParameter* param2=new TParameter(&type, &vec2, "", "");

	EXPECT_TRUE(param1->copy(param2));

	EXPECT_EQ(vec1.num_feat_entries, vec2.num_feat_entries);
	EXPECT_EQ(vec2.num_feat_entries, 2);
	EXPECT_EQ(vec1.features[0].feat_index, vec2.features[0].feat_index);
	EXPECT_EQ(vec1.features[1].feat_index, vec2.features[1].feat_index);
	EXPECT_EQ(vec1.features[0].entry, vec2.features[0].entry);
	EXPECT_EQ(vec1.features[1].entry, vec2.features[1].entry);

	delete param1;
	delete param2;
	SG_FREE(temp);
	s1->free_sparse_feature_vector(0);
	s2->free_sparse_feature_vector(0);
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,copy_SPARSE_SCALAR_source_and_target_empty)
{
	SGMatrix<float64_t> a(2,1);
	SGMatrix<float64_t> b(3,1);

	a.set_const(1);
	b.set_const(2);

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseVector<float64_t> vec1=s1->get_sparse_feature_vector(0);
	SGSparseVector<float64_t> vec2=s2->get_sparse_feature_vector(0);
	void* temp1=vec1.features;
	void* temp2=vec2.features;
	vec2.features=NULL;
	vec1.features=NULL;
	vec1.num_feat_entries=0;
	vec2.num_feat_entries=0;

	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &vec1, "", "");
	TParameter* param2=new TParameter(&type, &vec2, "", "");

	param1->copy(param2);

	EXPECT_EQ(vec1.num_feat_entries, vec2.num_feat_entries);
	EXPECT_EQ(vec2.num_feat_entries, 0);

	delete param1;
	delete param2;
	SG_FREE(temp1);
	SG_FREE(temp2);
	s1->free_sparse_feature_vector(0);
	s2->free_sparse_feature_vector(0);
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,copy_SGMATRIX_SPARSE_same_size)
{
	SGMatrix<float64_t> a(2,2);
	SGMatrix<float64_t> b(2,2);

	a.set_const(1);
	b.set_const(2);

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseMatrix<float64_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<float64_t> mat2=s2->get_sparse_feature_matrix();

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	//	float64_t accuracy=0.1;
	//	EXPECT_FALSE(param1->equals(param2, accuracy));
	EXPECT_TRUE(param1->copy(param2));
	//	EXPECT_TRUE(param1->equals(param2, accuracy));

	EXPECT_EQ(mat1[0].num_feat_entries, mat2[0].num_feat_entries);
	EXPECT_EQ(mat1[0].features[0].feat_index, mat2[0].features[0].feat_index);
	EXPECT_EQ(mat1[0].features[1].feat_index, mat2[0].features[1].feat_index);
	EXPECT_EQ(mat1[0].features[0].entry, mat2[0].features[0].entry);
	EXPECT_EQ(mat1[0].features[1].entry, mat2[0].features[1].entry);
	EXPECT_EQ(mat1[1].num_feat_entries, mat2[1].num_feat_entries);
	EXPECT_EQ(mat1[1].features[0].feat_index, mat2[1].features[0].feat_index);
	EXPECT_EQ(mat1[1].features[1].feat_index, mat2[1].features[1].feat_index);
	EXPECT_EQ(mat1[1].features[0].entry, mat2[1].features[0].entry);
	EXPECT_EQ(mat1[1].features[1].entry, mat2[1].features[1].entry);

	delete param1;
	delete param2;
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,copy_SGMATRIX_SPARSE_different_size)
{
	SGMatrix<float64_t> a(2,2);
	SGMatrix<float64_t> b(2,3);

	a.set_const(1);
	b.set_const(2);


	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseMatrix<float64_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<float64_t> mat2=s2->get_sparse_feature_matrix();

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	//	float64_t accuracy=0.1;
	//	EXPECT_FALSE(param1->equals(param2, accuracy));
	EXPECT_TRUE(param1->copy(param2));
	//	EXPECT_TRUE(param1->equals(param2, accuracy));

	EXPECT_EQ(mat1[0].num_feat_entries, mat2[0].num_feat_entries);
	EXPECT_EQ(mat1[0].features[0].feat_index, mat2[0].features[0].feat_index);
	EXPECT_EQ(mat1[0].features[1].feat_index, mat2[0].features[1].feat_index);
	EXPECT_EQ(mat1[0].features[0].entry, mat2[0].features[0].entry);
	EXPECT_EQ(mat1[0].features[1].entry, mat2[0].features[1].entry);
	EXPECT_EQ(mat1[1].num_feat_entries, mat2[1].num_feat_entries);
	EXPECT_EQ(mat1[1].features[0].feat_index, mat2[1].features[0].feat_index);
	EXPECT_EQ(mat1[1].features[1].feat_index, mat2[1].features[1].feat_index);
	EXPECT_EQ(mat1[1].features[0].entry, mat2[1].features[0].entry);
	EXPECT_EQ(mat1[1].features[1].entry, mat2[1].features[1].entry);

	delete param1;
	delete param2;
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,copy_SGMATRIX_SPARSE_target_empty)
{
	SGMatrix<float64_t> a(2,2);

	a.set_const(1);


	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	SGSparseMatrix<float64_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<float64_t> mat2(2,2);

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	//	float64_t accuracy=0.1;
	//	EXPECT_FALSE(param1->equals(param2, accuracy));
	EXPECT_TRUE(param1->copy(param2));
	//	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
	SG_UNREF(s1);
}

TEST(TParameter,copy_SGMATRIX_SPARSE_source_and_target_empty)
{
	SGSparseMatrix<float64_t> mat1(2,2);
	SGSparseMatrix<float64_t> mat2(2,2);
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	//	float64_t accuracy=0.1;
	EXPECT_TRUE(param1->copy(param2));
	//	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}
