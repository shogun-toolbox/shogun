/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <base/init.h>
#include <base/Parameter.h>
#include <io/SGIO.h>
#include <io/SerializableAsciiFile.h>
#include <base/ParameterMap.h>
#include <features/DenseFeatures.h>
#include <unistd.h>

using namespace shogun;

class CTestClassInt : public CSGObject
{
public:
	CTestClassInt()
	{
		m_number=10;
		m_parameters->add(&m_number, "number", "Test number");

		m_vector_length=3;
		m_vector=SG_MALLOC(int32_t, m_vector_length);
		SGVector<int32_t>::fill_vector(m_vector, m_vector_length, 10);
		m_parameters->add_vector(&m_vector, &m_vector_length, "vector",
				"Test vector");

		m_matrix_rows=2;
		m_matrix_cols=3;
		m_matrix=SG_MALLOC(int32_t, m_matrix_rows*m_matrix_cols);
		SGVector<int32_t>::range_fill_vector(m_matrix, m_matrix_rows*m_matrix_cols);
		m_parameters->add_matrix(&m_matrix, &m_matrix_rows, &m_matrix_cols,
				"matrix", "Test matrix");

		SGMatrix<int32_t> features=SGMatrix<int32_t>(2, 3);
		SGVector<int32_t>::range_fill_vector(features.matrix,
				features.num_rows*features.num_cols, 3);
		m_features=new CDenseFeatures<int32_t>(features);
		SG_REF(m_features);
		m_parameters->add((CSGObject**)&m_features, "int_features",
				"Test features");
	}

	virtual ~CTestClassInt()
	{
		SG_FREE(m_vector);
		SG_FREE(m_matrix);
		SG_UNREF(m_features);
	}

	int32_t m_number;
	int32_t* m_vector;
	int32_t m_vector_length;

	int32_t* m_matrix;
	int32_t m_matrix_rows;
	int32_t m_matrix_cols;

	CDenseFeatures<int32_t>* m_features;

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

		SGMatrix<float64_t> features=SGMatrix<float64_t>(2, 3);
		SGVector<float64_t>::range_fill_vector(features.matrix,
				features.num_rows*features.num_cols, 3.0);
		m_features=new CDenseFeatures<float64_t>(features);
		SG_REF(m_features);
		m_parameters->add((CSGObject**)&m_features, "float_features",
				"Test features");

		/* add some parameter mappings for number, here: type changes */
		m_parameter_map->put(
				new const SGParamInfo("number", CT_SCALAR, ST_NONE, PT_FLOAT64, 1),
				new const SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT8, 0)
		);

		m_parameter_map->put(
				new const SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT8, 0),
				new const SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT32, -1)
		);

		/* changes for vector: from int32_t vector to float64_t SG_VECTOR */
		m_parameter_map->put(
				new const SGParamInfo("vector", CT_SGVECTOR, ST_NONE, PT_FLOAT64, 1),
				new const SGParamInfo("vector", CT_SGVECTOR, ST_NONE, PT_INT32, 0)
		);

		/* from normal vector to SG_VECTOR of same type */
		m_parameter_map->put(
				new const SGParamInfo("vector", CT_SGVECTOR, ST_NONE, PT_INT32, 0),
				new const SGParamInfo("vector", CT_VECTOR, ST_NONE, PT_INT32, -1)
		);

		/* changes for vector: from int32_t vector to float64_t SG_VECTOR */
		m_parameter_map->put(
				new const SGParamInfo("matrix", CT_SGMATRIX, ST_NONE, PT_FLOAT64, 1),
				new const SGParamInfo("matrix", CT_SGMATRIX, ST_NONE, PT_INT32, 0)
		);

		/* from normal vector to SG_VECTOR of same type */
		m_parameter_map->put(
				new const SGParamInfo("matrix", CT_SGMATRIX, ST_NONE, PT_INT32, 0),
				new const SGParamInfo("matrix", CT_MATRIX, ST_NONE, PT_INT32, -1)
		);

		/* name change for sgobject */
		m_parameter_map->put(
				new const SGParamInfo("float_features", CT_SCALAR, ST_NONE,
						PT_SGOBJECT, 1),
				new const SGParamInfo("int_features", CT_SCALAR, ST_NONE, PT_SGOBJECT,
						0)
		);

		m_parameter_map->finalize_map();
	}

	virtual ~CTestClassFloat()
	{
		SG_UNREF(m_features);
	}

	float64_t m_number;
	SGVector<float64_t> m_vector;
	SGMatrix<float64_t> m_matrix;
	CDenseFeatures<float64_t>* m_features;

	virtual const char* get_name() const { return "TestClassFloat"; }
};

void test_load_file_parameters()
{
	char filename_tmp[] = "/tmp/file_params_test.XXXXXX";
	char* filename = mktemp(filename_tmp);

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
	const SGParamInfo param_info_number(
			float_instance->m_parameters->get_parameter(0), 1);

	const SGParamInfo param_info_vector(
			float_instance->m_parameters->get_parameter(1), 1);

	const SGParamInfo param_info_matrix(
			float_instance->m_parameters->get_parameter(2), 1);

	const SGParamInfo param_info_sgobject(
			float_instance->m_parameters->get_parameter(3), 1);

	int32_t file_version=-1;

	/* now, here the magic happens, the parameter info of the float instance is
	 * mapped backwards (see its parameter map above) until the parameter
	 * info of the file is found. Then the parameters with the file version
	 * are loaded into memory. This will be used for migration.
	 * Note that only one parameter is in the array here for testing */
	DynArray<TParameter*>* file_loaded_number=
			float_instance->load_file_parameters(&param_info_number,
					file_version, file);

	DynArray<TParameter*>* file_loaded_vector=
			float_instance->load_file_parameters(&param_info_vector,
					file_version, file);

	DynArray<TParameter*>* file_loaded_matrix=
			float_instance->load_file_parameters(&param_info_matrix,
					file_version, file);

	DynArray<TParameter*>* file_loaded_sgobject=
			float_instance->load_file_parameters(&param_info_sgobject,
					file_version, file);

	/* Note that there is only ONE element in array here (old test) */
	TParameter* current;

	/* ensure that its the same as of the instance */
	current=file_loaded_number->get_element(0);
	int32_t value_number=*((int32_t*)current->m_parameter);
	SG_SPRINT("%i\n", value_number);
	ASSERT(value_number=int_instance->m_number);

	/* same for the vector */
	current=file_loaded_vector->get_element(0);
	int32_t* value_vector=*((int32_t**)current->m_parameter);
	SGVector<int32_t>::display_vector(value_vector, int_instance->m_vector_length);
	for (index_t i=0; i<int_instance->m_vector_length; ++i)
		ASSERT(value_vector[i]=int_instance->m_vector[i]);

	/* and for the matrix */
	current=file_loaded_matrix->get_element(0);
	int32_t* value_matrix=*((int32_t**)current->m_parameter);
	SGMatrix<int32_t>::display_matrix(value_matrix, int_instance->m_matrix_rows,
			int_instance->m_matrix_cols);
	for (index_t i=0; i<int_instance->m_matrix_rows*int_instance->m_matrix_cols;
			++i)
	{
		ASSERT(value_matrix[i]==int_instance->m_matrix[i]);
	}

	/* and for the feature object */
	current=file_loaded_sgobject->get_element(0);
	CDenseFeatures<int32_t>* features=
			*((CDenseFeatures<int32_t>**)current->m_parameter);
	SGMatrix<int32_t> feature_matrix_loaded=
			features->get_feature_matrix();
	SGMatrix<int32_t> feature_matrix_original=
			int_instance->m_features->get_feature_matrix();

	SGMatrix<int32_t>::display_matrix(feature_matrix_loaded.matrix,
			feature_matrix_loaded.num_rows,
			feature_matrix_loaded.num_cols,
			"features");
	for (index_t i=0;
			i<int_instance->m_matrix_rows*int_instance->m_matrix_cols;
			++i)
	{
		ASSERT(feature_matrix_original.matrix[i]==
				feature_matrix_loaded.matrix[i]);
	}

	/* only the TParameter instances have to be deleted, data, data pointer,
	 * and possible length variables are deleted automatically */
	for (index_t i=0; i<file_loaded_number->get_num_elements(); ++i)
		delete file_loaded_number->get_element(i);

	for (index_t i=0; i<file_loaded_vector->get_num_elements(); ++i)
		delete file_loaded_vector->get_element(i);

	for (index_t i=0; i<file_loaded_matrix->get_num_elements(); ++i)
		delete file_loaded_matrix->get_element(i);

	for (index_t i=0; i<file_loaded_sgobject->get_num_elements(); ++i)
		delete file_loaded_sgobject->get_element(i);

	/* also delete arrays */
	delete file_loaded_number;
	delete file_loaded_vector;
	delete file_loaded_matrix;
	delete file_loaded_sgobject;


	file->close();
	SG_UNREF(file);
	SG_UNREF(int_instance);
	SG_UNREF(float_instance);
	unlink(filename);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();
	sg_io->set_loglevel(MSG_DEBUG);

	test_load_file_parameters();

	exit_shogun();

	return 0;
}

