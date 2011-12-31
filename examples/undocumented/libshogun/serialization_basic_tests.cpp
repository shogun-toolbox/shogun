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

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

class CTestClass : public CSGObject
{
public:
	CTestClass() {}
	CTestClass(float64_t number, float64_t vec_start)
	{
		m_number=number;
		m_vec=SGVector<float64_t>(10);
		CMath::range_fill_vector(m_vec.vector, m_vec.vlen, vec_start);

		m_mat=SGMatrix<float64_t>(3,3);
		CMath::range_fill_vector(m_mat.matrix, m_mat.num_cols*m_mat.num_rows,
				vec_start);

		m_parameters->add(&m_number, "number", "Test variable");
		m_parameters->add(&m_mat, "mat", "Test variable");
		m_parameters->add(&m_vec, "vec", "Test variable");
	}

	virtual ~CTestClass()
	{
		m_vec.destroy_vector();
		m_mat.destroy_matrix();
	}


	void print()
	{
		SG_PRINT("m_number=%f\n", m_number);
		CMath::display_vector(m_vec.vector, m_vec.vlen, "m_vec");
		CMath::display_vector(m_mat.matrix, m_mat.num_cols*m_mat.num_rows,
				"m_mat");
	}

	inline virtual const char* get_name() const { return "TestClass"; }

public:
	float64_t m_number;
	SGVector<float64_t> m_vec;
	SGMatrix<float64_t> m_mat;

};


const char* filename="test.txt";

void test_test_class_serial()
{
	CTestClass* to_save=new CTestClass(10, 0);
	CTestClass* to_load=new CTestClass(20, 10);

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

	SG_SPRINT("deserialized instance 1 into instance 2:\n");
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

	SG_UNREF(to_save);
	SG_UNREF(to_load);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_test_class_serial();

	exit_shogun();

	return 0;
}
