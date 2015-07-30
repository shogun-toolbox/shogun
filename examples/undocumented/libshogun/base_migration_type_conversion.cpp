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
#include <shogun/io/SGIO.h>
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
		SGVector<float64_t>::fill_vector(m_vector.vector, m_vector.vlen, 0.0);

		m_matrix=SGMatrix<float64_t>(3, 3);
		SGVector<float64_t>::range_fill_vector(m_matrix.matrix,
				m_matrix.num_rows*m_matrix.num_cols, 0.0);

		m_parameters->add(&m_number, "number", "Test number");
		m_parameters->add(&m_vector, "vector", "Test vector");
		m_parameters->add(&m_matrix, "matrix", "Test matrix");

		SGMatrix<int32_t> features=SGMatrix<int32_t>(2, 3);
		SGVector<int32_t>::range_fill_vector(features.matrix,
				features.num_rows*features.num_cols, 0);
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
			char* new_name=(char*) "int_features";
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate, new_name);
		}

		if (result)
			return result;
		else
			return CSGObject::migrate(param_base, target);
	}
};

void check_equalness(CTestClassInt* int_instance,
		CTestClassFloat* float_instance)
{
	/* number */
	SG_SPRINT("checking \"number\":\n");
	SG_SPRINT("\t%d==%f\n", int_instance->m_number, float_instance->m_number);
	ASSERT(int_instance->m_number==float_instance->m_number);

	/* "vector" */
	SG_SPRINT("checking \"vector\":\n");
	SG_SPRINT("\tlength: %d==%d\n", int_instance->m_vector_length,
			float_instance->m_vector.vlen);
	ASSERT(int_instance->m_vector_length==float_instance->m_vector.vlen);
	SGVector<int32_t>::display_vector(int_instance->m_vector, int_instance->m_vector_length,
			"oiginal", "\t");
	SGVector<float64_t>::display_vector(float_instance->m_vector.vector,
			float_instance->m_vector.vlen, "migrated", "\t");
	for (index_t i=0; i<int_instance->m_vector_length; ++i)
		ASSERT(int_instance->m_vector[i]==float_instance->m_vector.vector[i]);

	/* "matrix" */
	SG_SPRINT("checking \"matrix\":\n");
	SG_SPRINT("\trows: %d==%d\n", int_instance->m_matrix_rows,
			float_instance->m_matrix.num_rows);
	ASSERT(int_instance->m_matrix_rows==float_instance->m_matrix.num_rows);
	SG_SPRINT("\tcols: %d==%d\n", int_instance->m_matrix_cols,
			float_instance->m_matrix.num_cols);
	ASSERT(int_instance->m_matrix_cols==float_instance->m_matrix.num_cols);
	SGMatrix<int32_t>::display_matrix(int_instance->m_matrix, int_instance->m_matrix_rows,
			int_instance->m_matrix_cols, "original", "\t");
	SGMatrix<float64_t>::display_matrix(float_instance->m_matrix.matrix,
			float_instance->m_matrix.num_rows,
			float_instance->m_matrix.num_cols, "migrated", "\t");
	for (index_t i=0; i<int_instance->m_matrix_rows*int_instance->m_matrix_cols;
			++i)
	{
		ASSERT(int_instance->m_matrix[i]==float_instance->m_matrix.matrix[i]);
	}

	/* "features" */
	SG_SPRINT("checking \"features\":\n");
	SG_SPRINT("\tchecking \"feature matrix\":\n");
	SGMatrix<int32_t> original_matrix=
			int_instance->m_features->get_feature_matrix();
	SGMatrix<int32_t> migrated_matrix=
			float_instance->m_features->get_feature_matrix();

	SG_SPRINT("\t\trows: %d==%d\n", original_matrix.num_rows,
			migrated_matrix.num_rows);
	ASSERT(original_matrix.num_rows==migrated_matrix.num_rows);
	SG_SPRINT("\t\tcols: %d==%d\n", original_matrix.num_cols,
			migrated_matrix.num_cols);
	ASSERT(original_matrix.num_cols==migrated_matrix.num_cols);
	SGMatrix<int32_t>::display_matrix(original_matrix.matrix, original_matrix.num_rows,
			original_matrix.num_cols, "original", "\t\t");
	SGMatrix<int32_t>::display_matrix(migrated_matrix.matrix, migrated_matrix.num_rows,
			migrated_matrix.num_cols, "migrated", "\t\t");
	for (index_t i=0; i<int_instance->m_matrix_rows*int_instance->m_matrix_cols;
			++i)
	{
		ASSERT(original_matrix.matrix[i]==migrated_matrix.matrix[i]);
	}
}

void test_migration()
{
	char filename_tmp[] = "migration_type_conv_test.XXXXXX";
    int fd = mkstemp(filename_tmp);
    ASSERT(fd != -1);
	char* filename = filename_tmp;

	/* create one instance of each class */
	CTestClassInt* int_instance=new CTestClassInt();
	CTestClassFloat* float_instance=new CTestClassFloat();

	CSerializableAsciiFile* file;

	/* serialize int instance, use custom parameter version */
	file=new CSerializableAsciiFile(filename, 'w');
	int_instance->save_serializable(file, "", -1);
	file->close();
	SG_UNREF(file);

	/* now the magic happens, the float instance is derserialized from file.
	 * Note that the parameter types are different. they will all be mapped.
	 * See migration methods. Everything is just converted, value is kept.
	 * The float instance has different initial values for all members, however,
	 * after de-serializing it from the int_instance file, the values should be
	 * the same
	 *
	 * The parameter mappings are chosen in such way that CTestClassInt could
	 * be seen as an old version of CTestClassFloat. */

	/* de-serialize float instance, use custom parameter version
	 * Note that a warning will appear, complaining that there is no parameter
	 * version in file. This is not true, the version is -1, which is used here
	 * as custom version. Normally numbers >=0 are used. */
	file=new CSerializableAsciiFile(filename, 'r');
	// mute the warning so we don't have a false positive on the buildbot
	float_instance->io->set_loglevel(MSG_ERROR);
	float_instance->load_serializable(file, "", 1);
	float_instance->io->set_loglevel(MSG_WARN);
	file->close();
	SG_UNREF(file);

	/* assert that content is equal */
	check_equalness(int_instance, float_instance);

	SG_UNREF(int_instance);
	SG_UNREF(float_instance);
	SG_UNREF(file);
	unlink(filename);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_migration();

	exit_shogun();

	return 0;
}
