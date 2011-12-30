/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/base/Parameter.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/base/ParameterMap.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

class CTestClassInt : public CSGObject
{
public:
	CTestClassInt()
	{
		m_number=10;
		m_parameters->add(&m_number, "number", "Test number");

		m_vector_length=3;
		m_vector=SG_MALLOC(int32_t, m_vector_length);
		CMath::fill_vector(m_vector, m_vector_length, 10);
		m_parameters->add_vector(&m_vector, &m_vector_length, "vector",
				"Test vector");

		m_matrix_rows=2;
		m_matrix_cols=3;
		m_matrix=SG_MALLOC(int32_t, m_matrix_rows*m_matrix_cols);
		CMath::range_fill_vector(m_matrix, m_matrix_rows*m_matrix_cols);
		m_parameters->add_matrix(&m_matrix, &m_matrix_rows, &m_matrix_cols,
				"matrix", "Test matrix");
	}

	virtual ~CTestClassInt()
	{
		SG_FREE(m_vector);
		SG_FREE(m_matrix);
	}

	int32_t m_number;
	int32_t* m_vector;
	int32_t m_vector_length;
	int32_t* m_matrix;
	int32_t m_matrix_rows;
	int32_t m_matrix_cols;


	virtual const char* get_name() const { return "TestClassInt"; }
};

class CTestClassFloat : public CSGObject
{
public:
	CTestClassFloat()
	{
		m_number=3.2;
		m_vector=SGVector<float64_t>(10);
		m_matrix=SGMatrix<float64_t>(2, 3);

		m_parameters->add(&m_number, "number", "Test number");
		m_parameters->add(&m_vector, "vector", "Test vector");
		m_parameters->add(&m_matrix, "matrix", "Test matrix");

		/* add some parameter mappings for number, here: type changes */
		m_parameter_map->put(
				new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_FLOAT64, 1),
				new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT8, 0)
		);

		m_parameter_map->put(
				new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT8, 0),
				new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT32, -1)
		);

		/* changes for vector: from int32_t vector to float64_t SG_VECTOR */
		m_parameter_map->put(
				new SGParamInfo("vector", CT_SGVECTOR, ST_NONE, PT_FLOAT64, 1),
				new SGParamInfo("vector", CT_SGVECTOR, ST_NONE, PT_INT32, 0)
		);

		/* from normal vector to SG_VECTOR of same type */
		m_parameter_map->put(
				new SGParamInfo("vector", CT_SGVECTOR, ST_NONE, PT_INT32, 0),
				new SGParamInfo("vector", CT_VECTOR, ST_NONE, PT_INT32, -1)
		);

		/* changes for vector: from int32_t vector to float64_t SG_VECTOR */
		m_parameter_map->put(
				new SGParamInfo("matrix", CT_SGMATRIX, ST_NONE, PT_FLOAT64, 1),
				new SGParamInfo("matrix", CT_SGMATRIX, ST_NONE, PT_INT32, 0)
		);

		/* from normal vector to SG_VECTOR of same type */
		m_parameter_map->put(
				new SGParamInfo("matrix", CT_SGMATRIX, ST_NONE, PT_INT32, 0),
				new SGParamInfo("matrix", CT_MATRIX, ST_NONE, PT_INT32, -1)
		);

		m_parameter_map->finalize_map();
	}

	virtual ~CTestClassFloat()
	{
		m_vector.destroy_vector();
		m_matrix.destroy_matrix();
	}

	float64_t m_number;
	SGVector<float64_t> m_vector;
	SGMatrix<float64_t> m_matrix;

	virtual const char* get_name() const { return "TestClassFloat"; }
};

const char* filename="test.txt";

void test_load_file_parameter()
{
	/* create one instance of each class */
	CTestClassInt* int_instance=new CTestClassInt();
	CTestClassFloat* float_instance=new CTestClassFloat();

	CSerializableAsciiFile* file;

	/* serialize int instance */
	file=new CSerializableAsciiFile(filename, 'w');
	int_instance->save_serializable(file);
	file->close();
	SG_UNREF(file);

	/* reopen file for reading */
	file=new CSerializableAsciiFile(filename, 'r');

	/* build parameter info for parameter of the OTHER instance, start from
	 * version 1 */
	SGParamInfo param_info_number(
			float_instance->m_parameters->get_parameter(0), 1);

	SGParamInfo param_info_vector(
			float_instance->m_parameters->get_parameter(1), 1);

	SGParamInfo param_info_matrix(
			float_instance->m_parameters->get_parameter(2), 1);

	int32_t file_version=-1;

	/* now, here the magic happens, the parameter info of the float instance is
	 * mapped backwards (see its parameter map above) until the parameter
	 * info of the file is found. Then the parameter with the file version
	 * is loaded into memory. This will be used for migration */
	TParameter* file_loaded_number=float_instance->load_file_parameter(
			&param_info_number, file_version, file);

	TParameter* file_loaded_vector=float_instance->load_file_parameter(
			&param_info_vector, file_version, file);

	TParameter* file_loaded_matrix=float_instance->load_file_parameter(
			&param_info_matrix, file_version, file);

	/* ensure that its he same as of the instance */
	int32_t value_number=*((int32_t*)file_loaded_number->m_parameter);
	SG_SPRINT("%i\n", value_number);
	ASSERT(value_number=int_instance->m_number);

	/* same for the vector */
	int32_t* value_vector=*((int32_t**)file_loaded_vector->m_parameter);
	CMath::display_vector(value_vector, int_instance->m_vector_length);
	for (index_t i=0; i<int_instance->m_vector_length; ++i)
		ASSERT(value_vector[i]=int_instance->m_vector[i]);

	/* and for the vector */
	int32_t* value_matrix=*((int32_t**)file_loaded_matrix->m_parameter);
	CMath::display_matrix(value_matrix, int_instance->m_matrix_rows,
			int_instance->m_matrix_cols);
	for (index_t i=0; i<int_instance->m_matrix_rows*int_instance->m_matrix_cols;
			++i)
	{
		ASSERT(value_matrix[i]==int_instance->m_matrix[i]);
	}


	/* only the TParameter instances have to be deleted, data, data pointer,
	 * and possible length variables are deleted automatically */
	delete file_loaded_number;
	delete file_loaded_vector;
	delete file_loaded_matrix;

	file->close();
	SG_UNREF(file);
	SG_UNREF(int_instance);
	SG_UNREF(float_instance);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_load_file_parameter();

	exit_shogun();

	return 0;
}

