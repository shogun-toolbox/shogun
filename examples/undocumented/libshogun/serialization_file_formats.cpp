/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/base/Parameter.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/io/SerializableHdf5File.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

const char* filename="filename.txt";

void print(Parameter* p)
{
	TParameter* param=p->get_parameter(0);

	SGVector<float64_t>* v=(SGVector<float64_t>*)param->m_parameter;
	CMath::display_vector(v->vector, v->vlen, "vector:");

	param=p->get_parameter(1);
	SGMatrix<float64_t>* m=(SGMatrix<float64_t>*)param->m_parameter;
	CMath::display_matrix(m->matrix, m->num_rows, m->num_cols, "matrix:");
}

void check_content_equal(Parameter* save_param, Parameter* load_param)
{
	TParameter* p;

	p=save_param->get_parameter(0);
	SGVector<float64_t>* sv=(SGVector<float64_t>*)p->m_parameter;
	p=save_param->get_parameter(1);
	SGMatrix<float64_t>* sm=(SGMatrix<float64_t>*)p->m_parameter;

	p=load_param->get_parameter(0);
	SGVector<float64_t>* lv=(SGVector<float64_t>*)p->m_parameter;
	p=load_param->get_parameter(1);
	SGMatrix<float64_t>* lm=(SGMatrix<float64_t>*)p->m_parameter;

	ASSERT(sv->vlen==lv->vlen);
	ASSERT(sm->num_rows==lm->num_rows);
	ASSERT(sm->num_cols==lm->num_cols);

	for (index_t i=0; i<sv->vlen; ++i)
		ASSERT(sv->vector[i]==lv->vector[i]);

	for (index_t i=0; i<sm->num_cols*sm->num_rows; ++i)
		ASSERT(sm->matrix[i]==lm->matrix[i]);
}

void test_ascii(Parameter* save_param, Parameter* load_param)
{
	SG_SPRINT("testing ascii serialization\n");
	SG_SPRINT("to save:\n");
	print(save_param);
	SG_SPRINT("loaded before:\n");
	print(load_param);

	CSerializableAsciiFile* file;

	file=new CSerializableAsciiFile(filename, 'w');
	save_param->save(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename, 'r');
	load_param->load(file);
	file->close();
	SG_UNREF(file);

	SG_SPRINT("loaded after:\n");
	print(load_param);

	check_content_equal(save_param, load_param);
}

void test_hdf5(Parameter* save_param, Parameter* load_param)
{
	/* TODO, HDF5 file leaks memory */
	SG_SPRINT("testing hdf5 serialization\n");
	SG_SPRINT("to save:\n");
	print(save_param);
	SG_SPRINT("loaded before:\n");
	print(load_param);

	CSerializableHdf5File* file;

	file=new CSerializableHdf5File(filename, 'w');
	save_param->save(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableHdf5File(filename, 'r');
	load_param->load(file);
	file->close();
	SG_UNREF(file);

	SG_SPRINT("loaded after:\n");
	print(load_param);

	check_content_equal(save_param, load_param);
}

void reset_values(Parameter* save_param, Parameter* load_param)
{
	TParameter* p;

	p=save_param->get_parameter(0);
	SGVector<float64_t>* sv=(SGVector<float64_t>*)p->m_parameter;
	p=save_param->get_parameter(1);
	SGMatrix<float64_t>* sm=(SGMatrix<float64_t>*)p->m_parameter;

	p=load_param->get_parameter(0);
	SGVector<float64_t>* lv=(SGVector<float64_t>*)p->m_parameter;
	p=load_param->get_parameter(1);
	SGMatrix<float64_t>* lm=(SGMatrix<float64_t>*)p->m_parameter;

	sv->destroy_vector();
	lv->destroy_vector();
	sm->destroy_matrix();
	lm->destroy_matrix();

	*sv=SGVector<float64_t>(9);
	*lv=SGVector<float64_t>(3);
	*sm=SGMatrix<float64_t>(3, 3);
	*lm=SGMatrix<float64_t>(4, 4);

	CMath::range_fill_vector(sv->vector, sv->vlen);
	CMath::range_fill_vector(sm->matrix, sm->num_rows*sm->num_cols);
	CMath::fill_vector(lv->vector, lv->vlen, 0.0);
	CMath::fill_vector(lm->matrix, lm->num_rows*lm->num_cols, 0.0);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	/* for serialization */
	SGVector<float64_t> sv;
	SGMatrix<float64_t> sm;
	Parameter* sp=new Parameter();
	sp->add(&sv, "vector", "description");
	sp->add(&sm, "matrix", "description");

	/* for deserialization */
	SGVector<float64_t> lv;
	SGMatrix<float64_t> lm;
	Parameter* lp=new Parameter();
	lp->add(&lv, "vector", "description");
	lp->add(&lm, "matrix", "description");

	reset_values(sp, lp);
	test_ascii(sp, lp);

	/* still leaks memory TODO */
	reset_values(sp, lp);
	test_hdf5(sp, lp);

	/* clean up */
	sv.destroy_vector();
	sm.destroy_matrix();
	lv.destroy_vector();
	lm.destroy_matrix();
	delete sp;
	delete lp;

	exit_shogun();

	return 0;
}

