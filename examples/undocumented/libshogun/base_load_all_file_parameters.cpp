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
#include <shogun/features/DenseFeatures.h>
#include <unistd.h>

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

		/* name change for sgobject */
		m_parameter_map->put(
				new SGParamInfo("float_features", CT_SCALAR, ST_NONE,
						PT_SGOBJECT, 1),
				new SGParamInfo("int_features", CT_SCALAR, ST_NONE, PT_SGOBJECT,
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

void test_load_file_parameter()
{
	char filename_tmp[] = "load_all_file_test.XXXXXX";
	char* filename=mktemp(filename_tmp);

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

	int32_t file_version=-1;

	/* load all parameter data, current version is set to 1 here */
	DynArray<TParameter*>* params=
			float_instance->load_all_file_parameters(file_version, 1, file, "");

	/* test the result */
	for (index_t i=0; i<params->get_num_elements(); ++i)
	{
		TParameter* current=params->get_element(i);
		current->m_delete_data = true; // TODO: This shouldn't be necessary!

		/* ensure that data is same as of the instance for all parameters */
		if (!strcmp(current->m_name, "number"))
		{
			int32_t value_number=*((int32_t*)current->m_parameter);
			SG_SPRINT("%i\n", value_number);
			ASSERT(value_number=int_instance->m_number);
		}
		else if (!strcmp(current->m_name, "vector"))
		{
			int32_t* value_vector=*((int32_t**)current->m_parameter);
			SGVector<int32_t>::display_vector(value_vector, int_instance->m_vector_length);

			for (index_t j=0; j<int_instance->m_vector_length; ++j)
				ASSERT(value_vector[j]=int_instance->m_vector[j]);
		}
		else if (!strcmp(current->m_name, "matrix"))
		{
			int32_t* value_matrix=*((int32_t**)current->m_parameter);
			SGMatrix<int32_t>::display_matrix(value_matrix, int_instance->m_matrix_rows,
					int_instance->m_matrix_cols);

			for (index_t j=0; j<int_instance->m_matrix_rows*int_instance->m_matrix_cols;
					++j)
			{
				ASSERT(value_matrix[j]==int_instance->m_matrix[j]);
			}
		}
		else if (!strcmp(current->m_name, "int_features"))
		{
			CDenseFeatures<int32_t>* features=
					*((CDenseFeatures<int32_t>**)
							current->m_parameter);
			SGMatrix<int32_t> feature_matrix_loaded=
					features->get_feature_matrix();
			SGMatrix<int32_t> feature_matrix_original=
					int_instance->m_features->get_feature_matrix();

			SGMatrix<int32_t>::display_matrix(feature_matrix_loaded.matrix,
					feature_matrix_loaded.num_rows,
					feature_matrix_loaded.num_cols,
					"features");

			for (index_t j=0;
					j<int_instance->m_matrix_rows*int_instance->m_matrix_cols;
					++j)
			{
				ASSERT(feature_matrix_original.matrix[j]==
						feature_matrix_loaded.matrix[j]);
			}
		}
	}

	/* assert that parameter data is sorted */
	for (index_t i=1; i<params->get_num_elements(); ++i)
	{
		/* assert via TParameter < and == operator */
		TParameter* t1=params->get_element(i-1);
		TParameter* t2=params->get_element(i);
		ASSERT((*t1)<(*t2) || (*t1)==(*t2));

		/* assert via name (which is used in the operator, but to be sure */
		const char* s1=t1->m_name;
		const char* s2=t2->m_name;
		SG_SPRINT("param \"%s\" <= \"%s\" ? ... ", s1, s2);
		ASSERT(strcmp(s1, s2)<=0);
		SG_SPRINT("yes\n");
	}


	/* clean up */
	for (index_t i=0; i<params->get_num_elements(); ++i)
		delete params->get_element(i);

	delete params;

	file->close();
	SG_UNREF(file);
	SG_UNREF(int_instance);
	SG_UNREF(float_instance);
	unlink(filename);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_load_file_parameter();

	exit_shogun();

	return 0;
}

