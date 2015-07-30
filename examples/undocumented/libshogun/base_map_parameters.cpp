/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/base/Parameter.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/base/ParameterMap.h>
#include <shogun/features/DenseFeatures.h>
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
		m_features=new CDenseFeatures<int32_t>(10);
		m_features->set_feature_matrix(features);
		m_features->set_combined_feature_weight(5.0);
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

		SGMatrix<int32_t> features=SGMatrix<int32_t>(2, 3);
		SGVector<int32_t>::range_fill_vector(features.matrix,
				features.num_rows*features.num_cols, 3);
		m_features=new CDenseFeatures<int32_t>(features);
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

		/* CSGObject mapping is not yet done */
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

	/* no type change here */
	CDenseFeatures<int32_t>* m_features;

	virtual const char* get_name() const { return "TestClassFloat"; }

	virtual TParameter* migrate(DynArray<TParameter*>* param_base,
				const SGParamInfo* target)
	{
		TSGDataType type(target->m_ctype, target->m_stype,
				target->m_ptype);

		TParameter* result=NULL;
		TParameter* to_migrate=NULL;

		if (*target==SGParamInfo("number", CT_SCALAR, ST_NONE, PT_FLOAT64, 1))
		{
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate);

			/* here: simply copy (and cast) data because nothing has changed */
			*((float64_t*)result->m_parameter)=
					*((int8_t*)to_migrate->m_parameter);
		}
		else if (*target==SGParamInfo("number", CT_SCALAR, ST_NONE,
				PT_INT8, 0))
		{
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate);

			/* here: simply copy (and cast) data because nothing has changed */
			*((int8_t*)result->m_parameter)=
					*((int32_t*)to_migrate->m_parameter);
		}
		else if (*target==SGParamInfo("vector", CT_SGVECTOR, ST_NONE,
				PT_FLOAT64, 1))
		{
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate);

			/* here: copy data element wise because type changes */
			float64_t* array_to=*((float64_t**)result->m_parameter);
			int32_t* array_from=*((int32_t**)to_migrate->m_parameter);
			for (index_t i=0; i<*to_migrate->m_datatype.m_length_y; ++i)
				array_to[i]=array_from[i];
		}
		else if (*target==SGParamInfo("vector", CT_SGVECTOR, ST_NONE,
				PT_INT32, 0))
		{
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate);

			/* here: copy data complete because its just wrapper type change */
			int32_t* array_to=*((int32_t**)result->m_parameter);
			int32_t* array_from=*((int32_t**)to_migrate->m_parameter);
			memcpy(array_to, array_from, to_migrate->m_datatype.get_size());
		}
		else if (*target==SGParamInfo("matrix", CT_SGMATRIX, ST_NONE,
				PT_INT32, 0))
		{
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate);

			/* here: copy data complete because its just wrapper type change */
			int32_t* array_to=*((int32_t**)result->m_parameter);
			int32_t* array_from=*((int32_t**)to_migrate->m_parameter);
			memcpy(array_to, array_from, to_migrate->m_datatype.get_size());
		}
		else if (*target==SGParamInfo("matrix", CT_SGMATRIX, ST_NONE,
				PT_FLOAT64, 1))
		{
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate);

			/* here: copy data element wise because type changes */
			float64_t* array_to=*((float64_t**)result->m_parameter);
			int32_t* array_from=*((int32_t**)to_migrate->m_parameter);
			for (index_t i=0; i<to_migrate->m_datatype.get_num_elements(); ++i)
				array_to[i]=array_from[i];
		}
		else if (*target==SGParamInfo("float_features", CT_SCALAR, ST_NONE,
				PT_SGOBJECT, 1))
		{
			/* specify name change and thats it */
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate, (char*) "int_features");
		}

		if (result)
			return result;
		else
			return CSGObject::migrate(param_base, target);
	}
};

void test_load_file_parameter()
{
	char filename_tmp[] = "map_params_test.XXXXXX";
    int fd = mkstemp(filename_tmp);
    ASSERT(fd != -1);
	char* filename = filename_tmp;

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

	/* versions that are used in this example */
	int32_t file_version=-1;
	int32_t current_version=1;

	/* load all parameter data, current version is set to 1 here */
	DynArray<TParameter*>* params=
			float_instance->load_all_file_parameters(file_version,
					current_version, file, "");

	/* create an array of param infos from float instance parameters */
	DynArray<const SGParamInfo*>* param_infos=
			new DynArray<const SGParamInfo*>();
	for (index_t i=0; i<float_instance->m_parameters->get_num_parameters(); ++i)
	{
		param_infos->append_element(
				new SGParamInfo(float_instance->m_parameters->get_parameter(i),
						current_version));
	}

	/* here the magic mapping happens */
	float_instance->map_parameters(params, file_version, param_infos);

	/* assert equalness of all parameters
	 * alphabetical order is "float_features", "matrix", "number", "vector" */
	TParameter* current=NULL;

	/* "float_features" (no type change here) */
	current=params->get_element(0);
	SG_SPRINT("checking \"float_features\":\n");
	ASSERT(!strcmp(current->m_name, "float_features"));
	/* cast to simple features */
	CDenseFeatures<int32_t>* features=
			*((CDenseFeatures<int32_t>**)current->m_parameter);
	SG_SPRINT("checking address (mapped!=original): %p!=%p\n", features,
			int_instance->m_features);
	ASSERT((void*)features!=(void*)int_instance->m_features);
	SG_SPRINT("checking cache size: %d==%d\n", features->get_cache_size(),
				int_instance->m_features->get_cache_size());
	ASSERT(features->get_cache_size()==
			int_instance->m_features->get_cache_size());
	SG_SPRINT("checking combined feature weight: %f==%f\n",
			features->get_combined_feature_weight(),
			int_instance->m_features->get_combined_feature_weight());
	ASSERT(features->get_combined_feature_weight()==
			int_instance->m_features->get_combined_feature_weight());
	SG_SPRINT("checking feature matrix:\n");
	SGMatrix<int32_t> int_matrix=int_instance->m_features->get_feature_matrix();
	SGMatrix<int32_t> float_matrix=features->get_feature_matrix();
	SG_SPRINT("number of rows: %d==%d\n", int_matrix.num_rows,
			float_matrix.num_rows);
	ASSERT(int_matrix.num_rows==float_matrix.num_rows);
	SG_SPRINT("number of cols: %d==%d\n", int_matrix.num_cols,
			float_matrix.num_cols);
	ASSERT(int_matrix.num_cols==float_matrix.num_cols);
	SGMatrix<int32_t>::display_matrix(float_matrix.matrix, float_matrix.num_rows,
			float_matrix.num_cols, "mapped");
	SGMatrix<int32_t>::display_matrix(int_matrix.matrix, int_matrix.num_rows,
			int_matrix.num_cols, "original");
	for (index_t i=0; i<int_matrix.num_rows*int_matrix.num_cols; ++i)
		ASSERT(int_matrix.matrix[i]==float_matrix.matrix[i]);

	/* "matrix" */
	current=params->get_element(1);
	ASSERT(!strcmp(current->m_name, "matrix"));
	SGMatrix<float64_t> matrix(*(float64_t**)current->m_parameter,
			*current->m_datatype.m_length_y, *current->m_datatype.m_length_x, false);
	SG_SPRINT("checking \"matrix:\n");
	SG_SPRINT("number of rows: %d==%d\n", *current->m_datatype.m_length_y,
			int_instance->m_matrix_rows);
	ASSERT(*current->m_datatype.m_length_y==int_instance->m_matrix_rows);
	SGMatrix<float64_t>::display_matrix(matrix.matrix, matrix.num_rows, matrix.num_cols,
			"mapped");
	SGMatrix<int32_t>::display_matrix(int_instance->m_matrix, int_instance->m_matrix_rows,
			int_instance->m_matrix_cols, "original");
	for (index_t i=0; i<int_instance->m_matrix_rows*int_instance->m_matrix_cols;
			++i)
	{
		ASSERT(matrix.matrix[i]==int_instance->m_matrix[i]);
	}

	/* "number" */
	current=params->get_element(2);
	ASSERT(!strcmp(current->m_name, "number"));
	float64_t number=*((float64_t*)current->m_parameter);
	SG_SPRINT("checking \"number\": %f == %d\n", number,
			int_instance->m_number);
	ASSERT(number==int_instance->m_number);

	/* "vector" */
	current=params->get_element(3);
	ASSERT(!strcmp(current->m_name, "vector"));
	SGVector<float64_t> vector(*(float64_t**)current->m_parameter,
			*current->m_datatype.m_length_y, false);
	SG_SPRINT("checking \"vector:\n");
	SG_SPRINT("length: %d==%d\n", *current->m_datatype.m_length_y,
			int_instance->m_vector_length);
	ASSERT(*current->m_datatype.m_length_y==int_instance->m_vector_length);
	SGVector<float64_t>::display_vector(vector.vector, vector.vlen, "mapped");
	SGVector<int32_t>::display_vector(int_instance->m_vector, int_instance->m_vector_length,
			"original");
	for (index_t i=0; i<int_instance->m_vector_length; ++i)
		ASSERT(vector.vector[i]==int_instance->m_vector[i]);

	/* clean up */
	for (index_t i=0; i<param_infos->get_num_elements(); ++i)
		delete param_infos->get_element(i);

	delete param_infos;

	for (index_t i=0; i<params->get_num_elements(); ++i)
	{
		/* delete data of TParameters because they were mapped */
		params->get_element(i)->m_delete_data=true;
		delete params->get_element(i);
	}

	delete params;

	file->close();
	SG_UNREF(file);
	SG_UNREF(int_instance);
	SG_UNREF(float_instance);
	unlink(filename);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test_load_file_parameter();

	exit_shogun();

	return 0;
}
