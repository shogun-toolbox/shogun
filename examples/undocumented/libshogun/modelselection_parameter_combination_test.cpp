/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Thoralf Klein, Roman Votyakov
 */

#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/SGVector.h>

#include <stdlib.h>

using namespace std;
using namespace shogun;

void test_parameter_set_multiplication()
{
	SG_SPRINT("\ntest_parameter_set_multiplication()\n");

	DynArray<Parameter*> set1;
	DynArray<Parameter*> set2;

	SGVector<float64_t> param_vector(8);
	SGVector<float64_t>::range_fill_vector(param_vector.vector, param_vector.vlen);

	Parameter parameters[4];

	parameters[0].add(&param_vector.vector[0], "0");
	parameters[0].add(&param_vector.vector[1], "1");
	set1.append_element(&parameters[0]);

	parameters[1].add(&param_vector.vector[2], "2");
	parameters[1].add(&param_vector.vector[3], "3");
	set1.append_element(&parameters[1]);

	parameters[2].add(&param_vector.vector[4], "4");
	parameters[2].add(&param_vector.vector[5], "5");
	set2.append_element(&parameters[2]);

	parameters[3].add(&param_vector.vector[6], "6");
	parameters[3].add(&param_vector.vector[7], "7");
	set2.append_element(&parameters[3]);

	DynArray<Parameter*>* result=new DynArray<Parameter*>();//CParameterCombination::parameter_set_multiplication(set1, set2);

	for (index_t i=0; i<result->get_num_elements(); ++i)
	{
		Parameter* p=result->get_element(i);
		for (index_t j=0; j<p->get_num_parameters(); ++j)
			SG_SPRINT("%s ", p->get_parameter(j)->m_name);

		SG_SPRINT("\n");
		delete p;
	}
	delete result;
}

void test_leaf_sets_multiplication()
{
	SG_SPRINT("\ntest_leaf_sets_multiplication()\n");
	SGVector<float64_t> param_vector(6);
	SGVector<float64_t>::range_fill_vector(param_vector.vector, param_vector.vlen);

	DynamicObjectArray sets;
	CParameterCombination* new_root=new CParameterCombination();

	DynamicObjectArray* current=new DynamicObjectArray();
	sets.append_element(current);
	Parameter* p=new Parameter();
	p->add(&param_vector.vector[0], "0");
	CParameterCombination* pc=new CParameterCombination(p);
	current->append_element(pc);

	p=new Parameter();
	p->add(&param_vector.vector[1], "1");
	pc=new CParameterCombination(p);
	current->append_element(pc);

	/* first case: one element */
	DynamicObjectArray* result_simple=
			CParameterCombination::leaf_sets_multiplication(sets, new_root);

	SG_SPRINT("one set\n");
	for (index_t i=0; i<result_simple->get_num_elements(); ++i)
	{
		CParameterCombination* tpc=(CParameterCombination*)
				result_simple->get_element(i);
		tpc->print_tree();
	}

	/* now more elements are created */

	current=new DynamicObjectArray();
	sets.append_element(current);
	p=new Parameter();
	p->add(&param_vector.vector[2], "2");
	pc=new CParameterCombination(p);
	current->append_element(pc);

	p=new Parameter();
	p->add(&param_vector.vector[3], "3");
	pc=new CParameterCombination(p);
	current->append_element(pc);

	current=new DynamicObjectArray();
	sets.append_element(current);
	p=new Parameter();
	p->add(&param_vector.vector[4], "4");
	pc=new CParameterCombination(p);
	current->append_element(pc);

	p=new Parameter();
	p->add(&param_vector.vector[5], "5");
	pc=new CParameterCombination(p);
	current->append_element(pc);

	/* second case: more element */
	DynamicObjectArray* result_complex=
			CParameterCombination::leaf_sets_multiplication(sets, new_root);

	SG_SPRINT("more sets\n");
	for (index_t i=0; i<result_complex->get_num_elements(); ++i)
	{
		CParameterCombination* tpc=(CParameterCombination*)
				result_complex->get_element(i);
		tpc->print_tree();
	}

}

int main(int argc, char **argv)
{
	test_parameter_set_multiplication();
	test_leaf_sets_multiplication();

	return 0;
}
