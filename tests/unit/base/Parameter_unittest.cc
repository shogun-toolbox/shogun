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
	complex128_t a(0.0);
	EXPECT_FALSE(TParameter::compare_ptype(PT_COMPLEX128, &a, NULL));
}

TEST(TParameter,compare_ptype_null4)
{
	complex128_t a(0.0);
	EXPECT_FALSE(TParameter::compare_ptype(PT_COMPLEX128, NULL, &a));
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

TEST(TParameter,compare_ptype_COMPLEX128)
{
	complex128_t a(1.0);
	complex128_t b(2.0);
	EXPECT_FALSE(TParameter::compare_ptype(PT_COMPLEX128, &a, &b));
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
	complex128_t a(1.0, 1.0);
	complex128_t b(1.5, 1.5);
	float64_t accuracy=0.2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_COMPLEX128);
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
	complex128_t a(1.0, 1.0);
	complex128_t b(1.2, 1.2);
	float64_t accuracy=0.2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_COMPLEX128);
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
	SGVector<complex128_t> a(2);
	SGVector<complex128_t> b(2);
	float64_t accuracy=0.1;

	a[0]=complex128_t(1.0, 1.0);
	b[0]=complex128_t(1.0, 1.0);
	a[1]=complex128_t(1.0, 1.0);
	b[1]=complex128_t(1.11, 1.11);

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_COMPLEX128, &a.vlen);
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
	SGVector<complex128_t> a(2);
	SGVector<complex128_t> b(2);
	float64_t accuracy=0.1;

	a[0]=complex128_t(1.0, 1.0);
	b[0]=complex128_t(1.0, 1.0);
	a[1]=complex128_t(1.0, 1.0);
	b[1]=complex128_t(1.01, 1.01);

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_COMPLEX128, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "", "");
	TParameter* param2=new TParameter(&type, &b.vector, "", "");

	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(TParameter,equals_VECTOR_STRING_FLOAT64_equal)
{
	// this tests whether equals works on vectors of string of non-char type
	// which is a quite tricky case
	index_t a_len=2;
	index_t b_len=2;

	SGString<float64_t>* a=SG_MALLOC(SGString<float64_t>, a_len);
	SGString<float64_t>* b=SG_MALLOC(SGString<float64_t>, b_len);

	index_t slen=2;
	float64_t* string1=SG_MALLOC(float64_t, slen);
	string1[0]=1.0;
	string1[1]=2.0;

	float64_t* string2=SG_MALLOC(float64_t, slen);
	string2[0]=3.0;
	string2[1]=4.0;

	a[0].string=string1;
	a[0].slen=slen;
	a[0].do_free=false;
	a[1].string=string2;
	a[1].slen=slen;
	a[1].do_free=false;

	b[0].string=string1;
	b[0].slen=slen;
	b[0].do_free=false;
	b[1].string=string2;
	b[1].slen=slen;
	b[1].do_free=false;

	TSGDataType type_a(CT_VECTOR, ST_STRING, PT_FLOAT64, &a_len);
	TSGDataType type_b(CT_VECTOR, ST_STRING, PT_FLOAT64, &b_len);

	TParameter* param1=new TParameter(&type_a, &a, "", "");
	TParameter* param2=new TParameter(&type_b, &b, "", "");

	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;

	SG_FREE(a);
	SG_FREE(b);
	SG_FREE(string1);
	SG_FREE(string2);
}

TEST(TParameter,equals_VECTOR_STRING_FLOAT64_different)
{
	// this tests whether equals works on vectors of string of non-char type
	// which is a quite tricky case
	index_t a_len=2;
	index_t b_len=2;

	SGString<float64_t>* a=SG_MALLOC(SGString<float64_t>, a_len);
	SGString<float64_t>* b=SG_MALLOC(SGString<float64_t>, b_len);

	index_t slen=2;
	float64_t* string1=SG_MALLOC(float64_t, slen);
	string1[0]=1.0;
	string1[1]=2.0;

	float64_t* string2=SG_MALLOC(float64_t, slen);
	string2[0]=3.0;
	string2[1]=4.0;

	a[0].string=string1;
	a[0].slen=slen;
	a[0].do_free=false;
	a[1].string=string1;
	a[1].slen=slen;
	a[1].do_free=false;

	b[0].string=string1;
	b[0].slen=slen;
	b[0].do_free=false;
	b[1].string=string2;
	b[1].slen=slen;
	b[1].do_free=false;

	TSGDataType type_a(CT_VECTOR, ST_STRING, PT_FLOAT64, &a_len);
	TSGDataType type_b(CT_VECTOR, ST_STRING, PT_FLOAT64, &b_len);

	TParameter* param1=new TParameter(&type_a, &a, "", "");
	TParameter* param2=new TParameter(&type_b, &b, "", "");

	EXPECT_FALSE(param1->equals(param2));

	delete param1;
	delete param2;

	SG_FREE(a);
	SG_FREE(b);
	SG_FREE(string1);
	SG_FREE(string2);
}

TEST(TParameter,equals_MATRIX_STRING_FLOAT64_equal)
{
	index_t a_len=2;
	index_t b_len=2;

	SGString<float64_t>* a=SG_MALLOC(SGString<float64_t>, a_len);
	SGString<float64_t>* b=SG_MALLOC(SGString<float64_t>, b_len);

	index_t slen=2;
	float64_t* string1=SG_MALLOC(float64_t, slen);
	string1[0]=1.0;
	string1[1]=2.0;

	float64_t* string2=SG_MALLOC(float64_t, slen);
	string2[0]=3.0;
	string2[1]=4.0;

	a[0].string=string1;
	a[0].slen=slen;
	a[0].do_free=false;
	a[1].string=string2;
	a[1].slen=slen;
	a[1].do_free=false;

	b[0].string=string1;
	b[0].slen=slen;
	b[0].do_free=false;
	b[1].string=string2;
	b[1].slen=slen;
	b[1].do_free=false;

	// mimic a matrix with one row
	index_t len_y=1;

	TSGDataType type_a(CT_MATRIX, ST_STRING, PT_FLOAT64, &len_y, &a_len);
	TSGDataType type_b(CT_MATRIX, ST_STRING, PT_FLOAT64, &len_y, &b_len);

	TParameter* param1=new TParameter(&type_a, &a, "", "");
	TParameter* param2=new TParameter(&type_b, &b, "", "");

	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;

	SG_FREE(a);
	SG_FREE(b);
	SG_FREE(string1);
	SG_FREE(string2);
}

TEST(TParameter,equals_MATRIX_STRING_FLOAT64_different)
{
	index_t a_len=2;
	index_t b_len=2;

	SGString<float64_t>* a=SG_MALLOC(SGString<float64_t>, a_len);
	SGString<float64_t>* b=SG_MALLOC(SGString<float64_t>, b_len);

	index_t slen=2;
	float64_t* string1=SG_MALLOC(float64_t, slen);
	string1[0]=1.0;
	string1[1]=2.0;

	float64_t* string2=SG_MALLOC(float64_t, slen);
	string2[0]=3.0;
	string2[1]=4.0;

	a[0].string=string1;
	a[0].slen=slen;
	a[0].do_free=false;
	a[1].string=string2;
	a[1].slen=slen;
	a[1].do_free=false;

	b[0].string=string1;
	b[0].slen=slen;
	b[0].do_free=false;
	b[1].string=string1;
	b[1].slen=slen;
	b[1].do_free=false;

	// mimic a matrix with one row
	index_t len_y=1;

	TSGDataType type_a(CT_MATRIX, ST_STRING, PT_FLOAT64, &len_y, &a_len);
	TSGDataType type_b(CT_MATRIX, ST_STRING, PT_FLOAT64, &len_y, &b_len);

	TParameter* param1=new TParameter(&type_a, &a, "", "");
	TParameter* param2=new TParameter(&type_b, &b, "", "");

	EXPECT_FALSE(param1->equals(param2));

	delete param1;
	delete param2;

	SG_FREE(a);
	SG_FREE(b);
	SG_FREE(string1);
	SG_FREE(string2);
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
	SGMatrix<complex128_t> a(2,2);
	SGMatrix<complex128_t> b(2,2);
	float64_t accuracy=0.1;

	a.set_const(complex128_t(1.0, 1.0));
	b.set_const(complex128_t(1.0, 1.0));
	b(1,1)=complex128_t(1.11, 1.11);

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_COMPLEX128, &a.num_rows, &a.num_cols);
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
	SGMatrix<complex128_t> a(2,2);
	SGMatrix<complex128_t> b(2,2);
	float64_t accuracy=0.1;

	a.set_const(complex128_t(1.0, 1.0));
	b.set_const(complex128_t(1.0, 1.0));
	b(1,1)=complex128_t(1.01, 1.01);

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_COMPLEX128, &a.num_rows, &a.num_cols);
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

TEST(TParameter,equals_sparse_scalar_equal_different_index)
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
	vec1.features[1].feat_index=2;
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

TEST(TParameter,equals_sparse_matrix_data_different)
{
	SGMatrix<float64_t> a(2,2);
	a.set_const(1);

	SGMatrix<float64_t> b(2,2);
	b.set_const(1);
	b(1,1)=1.11;

	float64_t accuracy=0.1;

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseMatrix<float64_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<float64_t> mat2=s2->get_sparse_feature_matrix();

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
	SG_UNREF(s1);
	SG_UNREF(s2);
}

TEST(TParameter,equals_sparse_matrix_param_null)
{
	SGMatrix<floatmax_t> a(2,2);
	a.set_const(1);

	float64_t accuracy=0.1;

	CSparseFeatures<floatmax_t>* s1=new CSparseFeatures<floatmax_t>(a);

	SGSparseMatrix<floatmax_t> mat1=s1->get_sparse_feature_matrix();

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOATMAX,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, NULL, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
	SG_UNREF(s1);
}

TEST(TParameter,equals_sparse_matrix_one_matrix_null)
{
	SGMatrix<floatmax_t> a(2,2);
	a.set_const(1);

	float64_t accuracy=0.1;

	CSparseFeatures<floatmax_t>* s1=new CSparseFeatures<floatmax_t>(a);
	SGSparseMatrix<floatmax_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<floatmax_t> mat2(2,2);

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOATMAX,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
	SG_UNREF(s1);
}

TEST(TParameter,equals_sparse_matrix_equal)
{
	SGMatrix<uint8_t> a(2,2);
	a.set_const(1);

	SGMatrix<uint8_t> b(2,2);
	b.set_const(1);

	CSparseFeatures<uint8_t>* s1=new CSparseFeatures<uint8_t>(a);
	CSparseFeatures<uint8_t>* s2=new CSparseFeatures<uint8_t>(b);

	SGSparseMatrix<uint8_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<uint8_t> mat2=s2->get_sparse_feature_matrix();

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT8,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
	SG_UNREF(s1);
	SG_UNREF(s2);
}

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

	float64_t accuracy=0.1;

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseMatrix<float64_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<float64_t> mat2=s2->get_sparse_feature_matrix();

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));
	EXPECT_TRUE(param1->copy(param2));
	EXPECT_TRUE(param1->equals(param2, accuracy));

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

	float64_t accuracy=0.1;

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	CSparseFeatures<float64_t>* s2=new CSparseFeatures<float64_t>(b);

	SGSparseMatrix<float64_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<float64_t> mat2=s2->get_sparse_feature_matrix();

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));
	EXPECT_TRUE(param1->copy(param2));
	EXPECT_TRUE(param1->equals(param2, accuracy));

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

	float64_t accuracy=0.1;

	CSparseFeatures<float64_t>* s1=new CSparseFeatures<float64_t>(a);
	SGSparseMatrix<float64_t> mat1=s1->get_sparse_feature_matrix();
	SGSparseMatrix<float64_t> mat2(2,2);

	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	EXPECT_FALSE(param1->equals(param2, accuracy));
	EXPECT_TRUE(param1->copy(param2));
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
	SG_UNREF(s1);
}

TEST(TParameter,copy_SGMATRIX_SPARSE_source_and_target_empty)
{
	float64_t accuracy=0.1;

	SGSparseMatrix<float64_t> mat1(2,2);
	SGSparseMatrix<float64_t> mat2(2,2);
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64,
		&mat1.num_vectors, &mat1.num_features);
	TParameter* param1=new TParameter(&type, &mat1.sparse_matrix, "", "");
	TParameter* param2=new TParameter(&type, &mat2.sparse_matrix, "", "");

	EXPECT_TRUE(param1->copy(param2));
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}
