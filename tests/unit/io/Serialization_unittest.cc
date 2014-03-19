/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/io/SerializableJsonFile.h>
#include <shogun/io/SerializableXmlFile.h>
#include <shogun/io/SerializableHdf5File.h>
#include <gtest/gtest.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;

TEST(Serialization, Ascii_scalar_equal_BOOL)
{
	bool a=true;
	bool b=false;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_BOOL);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="bool_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_CHAR)
{
	char a='a';
	char b='b';

	TSGDataType type(CT_SCALAR, ST_NONE, PT_CHAR);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="char_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_INT8)
{
	int8_t a=1;
	int8_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT8);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int8_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_UINT8)
{
	uint8_t a=1;
	uint8_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT8);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint8_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_INT16)
{
	int16_t a=1;
	int16_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT16);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int16_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_UINT16)
{
	uint16_t a=1;
	uint16_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT16);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint16_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_INT32)
{
	int32_t a=1;
	int32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int32_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_UINT32)
{
	uint32_t a=1;
	uint32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint32_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_INT64)
{
	int64_t a=1;
	int64_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int64_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_UINT64)
{
	uint64_t a=1;
	uint64_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint64_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_FLOAT32)
{
	float32_t a=1.71265;
	float32_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="float32_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_FLOAT64)
{
	float64_t a=1.7126587125;
	float64_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="float64_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_FLOATMAX)
{
	floatmax_t a=1.7126587125;
	floatmax_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOATMAX);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="floatmax_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-15;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_scalar_equal_COMPLEX128)
{
	complex128_t a(1.7126587125, 2.7126587125);
	complex128_t b(0.0, 0.0);

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="complex128_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_vector_equal_FLOAT64)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);

	a.set_const(1.14263158);
	b.zero();

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "param", "");
	TParameter* param2=new TParameter(&type, &b.vector, "param", "");

	const char* filename="float64_sgvec_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_vector_equal_COMPLEX128)
{
	SGVector<complex128_t> a(2);
	SGVector<complex128_t> b(2);

	a.set_const(complex128_t(1.14263158, 2.83645548));
	b.zero();

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_COMPLEX128, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "param", "");
	TParameter* param2=new TParameter(&type, &b.vector, "param", "");

	const char* filename="complex128_sgvec_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_matrix_equal_FLOAT64)
{
	SGMatrix<float64_t> a(2, 2);
	SGMatrix<float64_t> b(2, 2);

	a.set_const(1.14263158);
	b.zero();

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "param", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "param", "");

	const char* filename="float64_sgmat_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Ascii_matrix_equal_COMPLEX128)
{
	SGMatrix<complex128_t> a(2, 2);
	SGMatrix<complex128_t> b(2, 2);

	a.set_const(complex128_t(1.14263158, 2.435754));
	b.zero();

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_COMPLEX128, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "param", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "param", "");

	const char* filename="complex128_sgmat_param.txt";
	// save parameter to an ascii file
	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from an ascii file
	file=new CSerializableAsciiFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

#ifdef HAVE_JSON
TEST(Serialization, Json_scalar_equal_BOOL)
{
	bool a=true;
	bool b=false;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_BOOL);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="bool_param.json";
	// save parameter to a json file
	CSerializableJsonFile *file=new CSerializableJsonFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a json file
	file=new CSerializableJsonFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Json_scalar_equal_FLOAT32)
{
	float32_t a=1.7325;
	float32_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="float32_param.json";
	// save parameter to a json file
	CSerializableJsonFile *file=new CSerializableJsonFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a json file
	file=new CSerializableJsonFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-6;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Json_scalar_equal_FLOAT64)
{
	float64_t a=1.7126587125;
	float64_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="float64_param.json";
	// save parameter to a json file
	CSerializableJsonFile *file=new CSerializableJsonFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a json file
	file=new CSerializableJsonFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-6;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Json_scalar_equal_FLOATMAX)
{
	floatmax_t a=1.7126587125;
	floatmax_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOATMAX);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="floatmax_param.json";
	// save parameter to a json file
	CSerializableJsonFile *file=new CSerializableJsonFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a json file
	file=new CSerializableJsonFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-6;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Json_vector_equal_FLOAT64)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);

	a.set_const(1.14263158);
	b.zero();

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "param", "");
	TParameter* param2=new TParameter(&type, &b.vector, "param", "");

	const char* filename="float64_sgvec_param.json";
	// save parameter to a json file
	CSerializableJsonFile *file=new CSerializableJsonFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a json file
	file=new CSerializableJsonFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-6;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Json_matrix_equal_FLOAT64)
{
	SGMatrix<float64_t> a(2, 2);
	SGMatrix<float64_t> b(2, 2);

	a.set_const(1.14263158);
	b.zero();

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "param", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "param", "");

	const char* filename="float64_sgmat_param.json";
	// save parameter to a json file
	CSerializableJsonFile *file=new CSerializableJsonFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a json file
	file=new CSerializableJsonFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-6;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}
#endif // HAVE_JSON

#ifdef HAVE_XML
TEST(Serialization, Xml_scalar_equal_BOOL)
{
	bool a=true;
	bool b=false;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_BOOL);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="bool_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_CHAR)
{
	char a='a';
	char b='b';

	TSGDataType type(CT_SCALAR, ST_NONE, PT_CHAR);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="char_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_INT8)
{
	int8_t a=1;
	int8_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT8);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int8_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_UINT8)
{
	uint8_t a=1;
	uint8_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT8);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint8_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_INT16)
{
	int16_t a=1;
	int16_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT16);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int16_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_UINT16)
{
	uint16_t a=1;
	uint16_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT16);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint16_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_INT32)
{
	int32_t a=1;
	int32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int32_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_UINT32)
{
	uint32_t a=1;
	uint32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint32_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_INT64)
{
	int64_t a=1;
	int64_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int64_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_UINT64)
{
	uint64_t a=1;
	uint64_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint64_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_FLOAT32)
{
	float32_t a=1.71265;
	float32_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="float32_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-15;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_FLOAT64)
{
	float64_t a=1.7126587125;
	float64_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="float64_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-15;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_FLOATMAX)
{
	floatmax_t a=1.7126587125;
	floatmax_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOATMAX);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="floatmax_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-15;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_scalar_equal_COMPLEX128)
{
	complex128_t a(1.7126587125, 1.7126587125);
	complex128_t b(0.0);

	TSGDataType type(CT_SCALAR, ST_NONE, PT_COMPLEX128);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="complex128_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=1E-15;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_vector_equal_FLOAT64)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);

	a.set_const(1.14263158);
	b.zero();

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "param", "");
	TParameter* param2=new TParameter(&type, &b.vector, "param", "");

	const char* filename="float64_sgvec_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_vector_equal_COMPLEX128)
{
	SGVector<complex128_t> a(2);
	SGVector<complex128_t> b(2);

	a.set_const(complex128_t(1.14263158, 2.83645548));
	b.zero();

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_COMPLEX128, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "param", "");
	TParameter* param2=new TParameter(&type, &b.vector, "param", "");

	const char* filename="complex128_sgvec_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_matrix_equal_FLOAT64)
{
	SGMatrix<float64_t> a(2, 2);
	SGMatrix<float64_t> b(2, 2);

	a.set_const(1.14263158);
	b.zero();

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "param", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "param", "");

	const char* filename="float64_sgmat_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Xml_matrix_equal_COMPLEX128)
{
	SGMatrix<complex128_t> a(2, 2);
	SGMatrix<complex128_t> b(2, 2);

	a.set_const(complex128_t(1.14263158, 2.435754));
	b.zero();

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_COMPLEX128, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "param", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "param", "");

	const char* filename="complex128_sgmat_param.xml";
	// save parameter to a xml file
	CSerializableXmlFile *file=new CSerializableXmlFile(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a xml file
	file=new CSerializableXmlFile(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

#endif // HAVE_XML

#ifdef HAVE_HDF5
TEST(Serialization, Hdf5_scalar_equal_BOOL)
{
	bool a=true;
	bool b=false;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_BOOL);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="bool_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_CHAR)
{
	char a='a';
	char b='b';

	TSGDataType type(CT_SCALAR, ST_NONE, PT_CHAR);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="char_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_INT8)
{
	int8_t a=1;
	int8_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT8);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int8_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_UINT8)
{
	uint8_t a=1;
	uint8_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT8);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint8_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_INT16)
{
	int16_t a=1;
	int16_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT16);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int16_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_UINT16)
{
	uint16_t a=1;
	uint16_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT16);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint16_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_INT32)
{
	int32_t a=1;
	int32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int32_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_UINT32)
{
	uint32_t a=1;
	uint32_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint32_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_INT64)
{
	int64_t a=1;
	int64_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="int64_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_UINT64)
{
	uint64_t a=1;
	uint64_t b=2;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="uint64_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	EXPECT_TRUE(param1->equals(param2));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_FLOAT32)
{
	float64_t a=1.71265;
	float32_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT32);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="float32_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_FLOAT64)
{
	float64_t a=1.7126587125;
	float64_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT64);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="float64_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_scalar_equal_FLOATMAX)
{
	floatmax_t a=1.7126587125;
	floatmax_t b=0.0;

	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOATMAX);
	TParameter* param1=new TParameter(&type, &a, "param", "");
	TParameter* param2=new TParameter(&type, &b, "param", "");

	const char* filename="floatmax_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Hd5f_vector_equal_FLOAT64)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);

	a.set_const(1.14263158);
	b.zero();

	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &a.vlen);
	TParameter* param1=new TParameter(&type, &a.vector, "param", "");
	TParameter* param2=new TParameter(&type, &b.vector, "param", "");

	const char* filename="float64_sgvec_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

TEST(Serialization, Hdf5_matrix_equal_FLOAT64)
{
	SGMatrix<float64_t> a(2, 2);
	SGMatrix<float64_t> b(2, 2);

	a.set_const(1.14263158);
	b.zero();

	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &a.num_rows, &a.num_cols);
	TParameter* param1=new TParameter(&type, &a.matrix, "param", "");
	TParameter* param2=new TParameter(&type, &b.matrix, "param", "");

	const char* filename="float64_sgmat_param.hdf5";
	// save parameter to a hdf5 file
	CSerializableHdf5File *file=new CSerializableHdf5File(filename, 'w');
	param1->save(file);
	file->close();
	SG_UNREF(file);

	// load parameter from a hdf5 file
	file=new CSerializableHdf5File(filename, 'r');
	param2->load(file);
	file->close();
	SG_UNREF(file);

	// check for equality
	float64_t accuracy=0.0;
	EXPECT_TRUE(param1->equals(param2, accuracy));

	delete param1;
	delete param2;
}

#endif // HAVE_HDF5
