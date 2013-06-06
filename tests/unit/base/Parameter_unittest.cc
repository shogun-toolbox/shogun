/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/base/Parameter.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/SparseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(TParameter,equals_null)
{
	TSGDataType type(CT_SCALAR, ST_NONE, PT_BOOL);
	TParameter* param=new TParameter(&type, NULL, "", "");

	EXPECT_FALSE(param->equals(NULL));

	delete param;
}

TEST(TParameter,equals_different_name)
{
	TSGDataType type(CT_SCALAR, ST_NONE, PT_BOOL);
	TParameter* param=new TParameter(&type, NULL, "a", "");
	TParameter* param2=new TParameter(&type, NULL, "b", "");

	EXPECT_FALSE(param->equals(param2));

	delete param;
	delete param2;
}

TEST(TSGDataType,equals_different_PT)
{
	TSGDataType type1(CT_SCALAR, ST_NONE, PT_BOOL);
	TSGDataType type2(CT_SCALAR, ST_NONE, PT_FLOAT64);

	EXPECT_FALSE(type1.equals(type2));
}

TEST(TSGDataType,equals_different_ST)
{
	TSGDataType type1(CT_SCALAR, ST_NONE, PT_BOOL);
	TSGDataType type2(CT_SCALAR, ST_STRING, PT_BOOL);

	EXPECT_FALSE(type1.equals(type2));
}

TEST(TSGDataType,equals_different_CT)
{
	TSGDataType type1(CT_SCALAR, ST_NONE, PT_BOOL);
	TSGDataType type2(CT_SGVECTOR, ST_NONE, PT_BOOL);

	EXPECT_FALSE(type1.equals(type2));
}

TEST(TSGDataType,equals_different_leny_value)
{
	index_t len_a=1;
	index_t len_b=2;

	TSGDataType type1(CT_SCALAR, ST_NONE, PT_BOOL, &len_a);
	TSGDataType type2(CT_SCALAR, ST_NONE, PT_BOOL, &len_b);

	EXPECT_FALSE(type1.equals(type2));
}

TEST(TSGDataType,equals_different_len_y_null1)
{
	index_t len_a=1;

	TSGDataType type1(CT_SCALAR, ST_NONE, PT_BOOL, NULL);
	TSGDataType type2(CT_SCALAR, ST_NONE, PT_BOOL, &len_a);

	EXPECT_FALSE(type1.equals(type2));
}

TEST(TSGDataType,equals_different_len_y_null2)
{
	index_t len_a=1;

	TSGDataType type1(CT_SCALAR, ST_NONE, PT_BOOL, &len_a);
	TSGDataType type2(CT_SCALAR, ST_NONE, PT_BOOL, NULL);

	EXPECT_FALSE(type1.equals(type2));
}

TEST(TSGDataType,equals_different_len_xy_values)
{
	index_t len_y_a=1;
	index_t len_x_a=2;

	index_t len_y_b=1;
	index_t len_x_b=4;

	TSGDataType type1(CT_SCALAR, ST_NONE, PT_BOOL, &len_y_a, &len_x_a);
	TSGDataType type2(CT_SCALAR, ST_NONE, PT_BOOL, &len_y_b, &len_x_b);

	EXPECT_FALSE(type1.equals(type2));
}

TEST(TParameter,compare_ptype_null1)
{
	int a=1;
	EXPECT_FALSE(TParameter::compare_ptype(PT_FLOAT64, NULL, &a));
}

TEST(TParameter,compare_ptype_null2)
{
	int a=1;
	EXPECT_FALSE(TParameter::compare_ptype(PT_FLOAT64, &a, NULL));
}

TEST(TParameter,compare_ptype_null3)
{
	complex64_t a(0.0);
	EXPECT_FALSE(TParameter::compare_ptype(PT_COMPLEX64, &a, NULL));
}

TEST(TParameter,compare_ptype_null4)
{
	complex64_t a(0.0);
	EXPECT_FALSE(TParameter::compare_ptype(PT_COMPLEX64, NULL, &a));
}

TEST(TParameter,compare_ptype_BOOL)
{
	bool a=true;
	bool b=false;
	EXPECT_FALSE(TParameter::compare_ptype(PT_BOOL, &a, &b));
}

TEST(TParameter,compare_ptype_CHAR)
{
	char a='a';
	char b='b';
	EXPECT_FALSE(TParameter::compare_ptype(PT_CHAR, &a, &b));
}

TEST(TParameter,compare_ptype_INT8)
{
	int8_t a=1;
	int8_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_INT8, &a, &b));
}

TEST(TParameter,compare_ptype_UINT8)
{
	uint8_t a=1;
	uint8_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_UINT8, &a, &b));
}

TEST(TParameter,compare_ptype_INT16)
{
	int16_t a=1;
	int16_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_INT16, &a, &b));
}

TEST(TParameter,compare_ptype_UINT16)
{
	uint16_t a=1;
	uint16_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_UINT16, &a, &b));
}

TEST(TParameter,compare_ptype_INT32)
{
	int32_t a=1;
	int32_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_INT32, &a, &b));
}

TEST(TParameter,compare_ptype_UINT32)
{
	uint32_t a=1;
	uint32_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_UINT32, &a, &b));
}

TEST(TParameter,compare_ptype_INT64)
{
	int64_t a=1;
	int64_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_INT64, &a, &b));
}

TEST(TParameter,compare_ptype_UINT64)
{
	uint64_t a=1;
	uint64_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_UINT64, &a, &b));
}

TEST(TParameter,compare_ptype_FLOAT32)
{
	float32_t a=1;
	float32_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_FLOAT32, &a, &b));
}

TEST(TParameter,compare_ptype_FLOAT64)
{
	float64_t a=1;
	float64_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_FLOAT64, &a, &b));
}

TEST(TParameter,compare_ptype_FLOATMAX)
{
	floatmax_t a=1;
	floatmax_t b=2;
	EXPECT_FALSE(TParameter::compare_ptype(PT_FLOATMAX, &a, &b));
}

TEST(TParameter,compare_ptype_COMPLEX64)
{
	complex64_t a(1.0);
	complex64_t b(2.0);
	EXPECT_FALSE(TParameter::compare_ptype(PT_COMPLEX64, &a, &b));
}

TEST(TParameter,compare_ptype_SGOBJECT)
{
	CBinaryLabels* a=new CBinaryLabels(10);
	CRegressionLabels* b=new CRegressionLabels(10);
	
	EXPECT_FALSE(TParameter::compare_ptype(PT_SGOBJECT, &a, &b));

	SG_UNREF(a);
	SG_UNREF(b);
}

TEST(TParameter,equals_scalar_different1)
{
	int32_t a=1;
	int32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_FALSE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_scalar_different2)
{
	float64_t a=1.0;
	float64_t b=1.5;
	float64_t accuracy=0.2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_scalar_different3)
{
	complex64_t a(1.0, 1.0);
	complex64_t b(1.5, 1.5);
	float64_t accuracy=0.2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_COMPLEX64);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_scalar_equal1)
{
	int32_t a=1;
	int32_t b=1;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_scalar_equal2)
{
	float64_t a=1.0;
	float64_t b=1.2;
	float64_t accuracy=0.2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_scalar_equal3)
{
	complex64_t a(1.0, 1.0);
	complex64_t b(1.2, 1.2);
	float64_t accuracy=0.2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_COMPLEX64);
	TParameter* param1=new TParameter(&type, &a, "", "");
	TParameter* param2=new TParameter(&type, &b, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_vector_different1)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);
	float64_t accuracy=0.1;

	a[0]=1;
	b[0]=1;
	a[1]=1;
	b[1]=1.11;

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "", "");
	TParameter* param2=new TParameter(&type, &b.vector, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_vector_different2)
{
	SGVector<complex64_t> a(2);
	SGVector<complex64_t> b(2);
	float64_t accuracy=0.1;

	a[0]=complex64_t(1.0, 1.0);
	b[0]=complex64_t(1.0, 1.0);
	a[1]=complex64_t(1.0, 1.0);
	b[1]=complex64_t(1.11, 1.11);

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_COMPLEX64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "", "");
	TParameter* param2=new TParameter(&type, &b.vector, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_vector_equal1)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);
	float64_t accuracy=0.1;

	a[0]=1;
	b[0]=1;
	a[1]=1;
	b[1]=1.01;

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "", "");
	TParameter* param2=new TParameter(&type, &b.vector, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_vector_equal2)
{
	SGVector<complex64_t> a(2);
	SGVector<complex64_t> b(2);
	float64_t accuracy=0.1;

	a[0]=complex64_t(1.0, 1.0);
	b[0]=complex64_t(1.0, 1.0);
	a[1]=complex64_t(1.0, 1.0);
	b[1]=complex64_t(1.01, 1.01);

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_COMPLEX64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "", "");
	TParameter* param2=new TParameter(&type, &b.vector, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_matrix_different1)
{
	SGMatrix<float64_t> a(2,2);
	SGMatrix<float64_t> b(2,2);
	float64_t accuracy=0.1;

	a.set_const(1);
	b.set_const(1);
	b(1,1)=1.11;

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_matrix_different2)
{
	SGMatrix<complex64_t> a(2,2);
	SGMatrix<complex64_t> b(2,2);
	float64_t accuracy=0.1;

	a.set_const(complex64_t(1.0, 1.0));
	b.set_const(complex64_t(1.0, 1.0));
	b(1,1)=complex64_t(1.11, 1.11);

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_COMPLEX64, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_matrix_equal1)
{
	SGMatrix<float64_t> a(2,2);
	SGMatrix<float64_t> b(2,2);
	float64_t accuracy=0.1;

	a.set_const(1);
	b.set_const(1);
	b(1,1)=1.01;

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_matrix_equal2)
{
	SGMatrix<complex64_t> a(2,2);
	SGMatrix<complex64_t> b(2,2);
	float64_t accuracy=0.1;

	a.set_const(complex64_t(1.0, 1.0));
	b.set_const(complex64_t(1.0, 1.0));
	b(1,1)=complex64_t(1.01, 1.01);

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_COMPLEX64, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_string_scalar_different)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);
	float64_t accuracy=0.1;

	a.set_const(1);
	b.set_const(1);
	b[1]=1.11;

	SGString<float64_t> str1(a);
	SGString<float64_t> str2(b);

	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &str1.string, "", "");
	TParameter* param2=new TParameter(&type, &str2.string, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_string_scalar_equal)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);
	float64_t accuracy=0.1;

	a.set_const(1);
	b.set_const(1);
	b[1]=1.01;

	SGString<float64_t> str1(a);
	SGString<float64_t> str2(b);

	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &str1.string, "", "");
	TParameter* param2=new TParameter(&type, &str2.string, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_sparse_scalar_different)
{
	SGMatrix<float64_t> a(2,2);
	a.set_const(1);
	SGMatrix<float64_t> b(2,2);
	b.set_const(1);
	float64_t accuracy=0.1;

	a.set_const(1);
	b.set_const(1);
	b(1,1)=1.11;

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseVector<float64_t> vec1=s1->get_sparse_feature_vector(1);
	SGSparseVector<float64_t> vec2=s2->get_sparse_feature_vector(1);

	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &vec1, "", "");
	TParameter* param2=new TParameter(&type, &vec2, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
	s1->free_sparse_feature_vector(0);
	s2->free_sparse_feature_vector(0);
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,equals_sparse_scalar_equal)
{
	SGMatrix<float64_t> a(2,2);
	a.set_const(1);
	SGMatrix<float64_t> b(2,2);
	b.set_const(1);
	float64_t accuracy=0.1;

	a.set_const(1);
	b.set_const(1);
	b(1,1)=1.01;

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseVector<float64_t> vec1=s1->get_sparse_feature_vector(1);
	SGSparseVector<float64_t> vec2=s2->get_sparse_feature_vector(1);

	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &vec1, "", "");
	TParameter* param2=new TParameter(&type, &vec2, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
	s1->free_sparse_feature_vector(0);
	s2->free_sparse_feature_vector(0);
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,copy_ptype_BOOL)
{
	bool a=true;
	bool b=false;
	EXPECT_TRUE(TParameter::copy_ptype(PT_BOOL, &a, &b));
	EXPECT_EQ(a, b);
}

TEST(TParameter,copy_ptype_CHAR)
{
	char a=1;
	char b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_CHAR, &a, &b));
	EXPECT_EQ(a, b);
}

TEST(TParameter,copy_ptype_INT8)
{
	int8_t a=1;
	int8_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_INT8, &a, &b));
	EXPECT_EQ(a, b);
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
}

TEST(TParameter,copy_ptype_FLOAT64)
{
	float64_t a=1;
	float64_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_FLOAT64, &a, &b));
	EXPECT_EQ(a, b);
}

TEST(TParameter,copy_ptype_FLOATMAX)
{
	floatmax_t a=1;
	floatmax_t b=2;
	EXPECT_TRUE(TParameter::copy_ptype(PT_FLOATMAX, &a, &b));
	EXPECT_EQ(a, b);
}
