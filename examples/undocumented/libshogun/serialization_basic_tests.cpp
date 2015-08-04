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
#include <shogun/features/DenseFeatures.h>
#include <unistd.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

class CTestClass : public CSGObject
{
public:
	CTestClass() {}
	CTestClass(float64_t number, float64_t vec_start, int32_t features_start)
	{
		m_number=number;
		m_vec=SGVector<float64_t>(10);
		SGVector<float64_t>::range_fill_vector(m_vec.vector, m_vec.vlen, vec_start);

		m_mat=SGMatrix<float64_t>(3,3);
		SGVector<float64_t>::range_fill_vector(m_mat.matrix, m_mat.num_cols*m_mat.num_rows,
				vec_start);

		SGMatrix<int32_t> data=SGMatrix<int32_t>(3, 2);
		SGVector<int32_t>::range_fill_vector(data.matrix, data.num_rows*data.num_cols,
				features_start);
		m_features=new CDenseFeatures<int32_t>(data);
		SG_REF(m_features);

		m_parameters->add(&m_number, "number", "Test variable");
		m_parameters->add(&m_mat, "mat", "Test variable");
		m_parameters->add(&m_vec, "vec", "Test variable");
		m_parameters->add((CSGObject**)&m_features, "features", "Test variable");
	}

	virtual ~CTestClass()
	{
		SG_UNREF(m_features);
	}


	void print()
	{
		SG_PRINT("m_number=%f\n", m_number);
		SGVector<float64_t>::display_vector(m_vec.vector, m_vec.vlen, "m_vec");
		SGVector<float64_t>::display_vector(m_mat.matrix, m_mat.num_cols*m_mat.num_rows,
				"m_mat");

		SGMatrix<int32_t> features=m_features->get_feature_matrix();
		SGMatrix<int32_t>::display_matrix(features.matrix, features.num_rows,
				features.num_cols, "m_features");
	}

	inline virtual const char* get_name() const { return "TestClass"; }

public:
	float64_t m_number;
	SGVector<float64_t> m_vec;
	SGMatrix<float64_t> m_mat;
	CDenseFeatures<int32_t>* m_features;
};


void test_test_class_serial()
{
	char filename_tmp[] = "serialization_test.XXXXXX";
	int fd = mkstemp(filename_tmp);
	ASSERT(fd != -1);
	int retval = close(fd);
	ASSERT(retval != -1);
	char* filename = filename_tmp;

	CTestClass* to_save=new CTestClass(10, 0, 0);
	CTestClass* to_load=new CTestClass(20, 10, 66);

	SG_SPRINT("original instance 1:\n");
	to_save->print();
	SG_SPRINT("original instance 2:\n");
	to_load->print();

	CSerializableAsciiFile* file;

	file=new CSerializableAsciiFile(filename, 'w');
	to_save->save_serializable(file);
	file->close();
	SG_UNREF(file);


	file=new CSerializableAsciiFile(filename, 'r');
	to_load->load_serializable(file);
	file->close();
	SG_UNREF(file);

	SG_SPRINT("deserialized instance 1 into instance 2: (should be equal to "
			"first instance)\n");
	to_load->print();

	/* assert that variable is equal */
	ASSERT(to_load->m_number==to_save->m_number);


	/* assert that vector is equal */
	for (index_t i=0; i<to_load->m_vec.vlen; ++i)
	{
		ASSERT(to_load->m_vec[i]==to_save->m_vec[i]);
	}

	/* assert that matrix is equal */
	for (index_t i=0; i<to_load->m_mat.num_cols*to_load->m_mat.num_rows; ++i)
	{
		ASSERT(to_load->m_mat[i]==to_save->m_mat[i]);
	}

	/* assert that features object is equal */
	SGMatrix<int32_t> features_loaded=to_load->m_features->get_feature_matrix();
	SGMatrix<int32_t> features_saved=to_save->m_features->get_feature_matrix();
	for (index_t i=0; i<features_loaded.num_rows*features_loaded.num_cols; ++i)
	{
		ASSERT(features_loaded[i]==features_saved[i]);
	}

	SG_UNREF(to_save);
	SG_UNREF(to_load);
	unlink(filename);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_test_class_serial();

	exit_shogun();

	return 0;
}

