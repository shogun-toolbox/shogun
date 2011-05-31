/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/ParameterCombination.h"
#include "base/Parameter.h"

using namespace shogun;

CParameterCombination::CParameterCombination()
{
	m_param=NULL;
	m_node_name=NULL;
	m_child_nodes=new DynArray<CParameterCombination*> ();
}

CParameterCombination::CParameterCombination(const char* name) :
	m_node_name(name)
{
	m_param=NULL;
	m_child_nodes=new DynArray<CParameterCombination*> ();

}

CParameterCombination::CParameterCombination(Parameter* param) :
	m_param(param)
{
	m_node_name=NULL;
	m_child_nodes=new DynArray<CParameterCombination*> ();

}

CParameterCombination::~CParameterCombination()
{
	delete m_child_nodes;
}

void CParameterCombination::append_child(CParameterCombination* child)
{
	m_child_nodes->append_element(child);
}

void CParameterCombination::print(int prefix_num)
{
	/* prefix is enlarged */
	char* prefix=new char[prefix_num+1];
	for (index_t i=0; i<prefix_num; ++i)
		prefix[i]='\t';

	prefix[prefix_num]='\0';

	/* cases:
	 * -node with only a name and children
	 * -node with a Parameter instance and a possible children
	 * -root node with children
	 */

	if (m_node_name)
		SG_PRINT("%s\"%s\" ", prefix, m_node_name);
	else if (m_param)
	{
		SG_PRINT("%s", prefix);
		for (index_t i=0; i<m_param->get_num_parameters(); ++i)
		{
			/* distinction between sgobject and values */
			if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_SGOBJECT)
				SG_PRINT("CSGObject:%s ", m_param->get_parameter(i)->m_name);
			else
				SG_PRINT("\"%s\"=%f ", m_param->get_parameter(i)->m_name,
						*((float64_t*)m_param->get_parameter(i)->m_parameter));
		}

	}
	else
		SG_PRINT("%sroot", prefix);

	SG_PRINT("\n");

	for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		m_child_nodes->get_element(i)->print(prefix_num+1);

	delete[] prefix;
}

void CParameterCombination::parameter_set_multiplication(
		DynArray<Parameter*>& set_1, DynArray<Parameter*>& set_2,
		DynArray<Parameter*>& result)
{
	for (index_t i=0; i<set_1.get_num_elements(); ++i)
	{
		for (index_t j=0; j<set_2.get_num_elements(); ++j)
		{
			Parameter* p=new Parameter();
			p->add_parameters(set_1[i]);
			p->add_parameters(set_2[j]);
			result.append_element(p);
		}

		/* delete input sets */
		delete set_1[i];
	}

	/* delete input sets */
	for (index_t i=0; i<set_2.get_num_elements(); ++i)
		delete set_2[i];
}

void CParameterCombination::leaf_sets_multiplication(
		DynArray<DynArray<CParameterCombination*>*>& sets,
		CParameterCombination* new_root,
		DynArray<CParameterCombination*>& result)
{
	/* check marginal cases */
	if (sets.get_num_elements()==1)
	{
		/* just copy the only element into result array.
		 * put root node before all combinations*/
		result=*sets[0];

		/* delete input set */
		delete sets[0];

		for (index_t i=0; i<result.get_num_elements(); ++i)
		{
			/* put new root as root into the tree and replace tree */
			CParameterCombination* root=new_root->copy_tree();
			root->append_child(result[i]);
			result.set_element(root, i);
		}
	}
	else if (sets.get_num_elements()>1)
	{
		/* now the case where at least two sets are given */

		/* first, extract Parameter instances of given sets */
		DynArray<DynArray<Parameter*>*> param_sets;

		for (index_t set_nr=0; set_nr<sets.get_num_elements(); ++set_nr)
		{
			DynArray<CParameterCombination*>* current_set=sets[set_nr];
			param_sets.append_element(new DynArray<Parameter*> ());

			for (index_t i=0; i<current_set->get_num_elements(); ++i)
			{
				CParameterCombination* current_node=current_set->get_element(i);

				if (current_node->m_child_nodes->get_num_elements())
				{
					SG_SERROR("leaf sets multiplication only possible if all "
							"trees are leafs");
				}

				Parameter* current_param=current_node->m_param;

				if (current_param)
					param_sets[set_nr]->append_element(current_param);
				else
				{
					SG_SERROR("leaf sets multiplication only possible if all "
							"leafs have non-NULL Parameter instances\n");
				}
			}
		}

		/* second, build products of all parameter sets */
		DynArray<Parameter*>* param_product=param_sets[0];

		/* build product of all remaining sets and collect results. delete all
		 * old parameter instances */
		for (index_t i=1; i<param_sets.get_num_elements(); ++i)
		{
			DynArray<Parameter*>* temp_result=new DynArray<Parameter*> ();
			parameter_set_multiplication(*param_product, *param_sets[i],
					*temp_result);

			/* these two deletes cover all DynArray instances that were yet
			 * created in this method (in first loop and two lines above) */
			delete param_product;
			delete param_sets[i];

			param_product=temp_result;
		}
		/* at this point there is only one DynArray instance remaining:
		 * param_product. contains all combinations of parameters of all given
		 * sets */

		/* third, build tree sets with the given root and the parameter product
		 * elements */
		for (index_t i=0; i<param_product->get_num_elements(); ++i)
		{
			/* build parameter node from parameter product to append to root */
			CParameterCombination* param_node=new CParameterCombination(
					param_product->get_element(i));

			/* copy new root node, has to be a new one each time */
			CParameterCombination* root=new_root->copy_tree();

			/* append both and add them to result set */
			root->append_child(param_node);

			result.append_element(root);
		}

		/* delete old trees. note that the Parameter instances in the
		 * tree have been deleted with the above call of
		 * parameter_set_multiplication. also delete elements of input sets
		 * array */
		for (index_t set_nr=0; set_nr<sets.get_num_elements(); ++set_nr)
		{
			DynArray<CParameterCombination*>* current_set=sets[set_nr];
			for (index_t i=0; i<current_set->get_num_elements(); ++i)
				delete current_set->get_element(i);

			delete current_set;
		}

		/* this is not needed anymore, because the Parameter instances are now
		 * in the resulting tree sets */
		delete param_product;
	}

	/* delete input new root node with parameter data (was copied) in any
	 * case */
	new_root->destroy(true, true);
}

void CParameterCombination::destroy(bool recursive, bool destroy_data)
{
	if (destroy_data)
		delete m_param;

	if (recursive)
	{
		for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
			m_child_nodes->get_element(i)->destroy(recursive, destroy_data);
	}

	delete this;
}

CParameterCombination* CParameterCombination::copy_tree()
{
	CParameterCombination* copy=new CParameterCombination();

	/* use name of original */
	copy->m_node_name=m_node_name;

	/* but build new Parameter instance */

	/* only call add_parameters() argument is non-null */
	if (m_param)
	{
		copy->m_param=new Parameter();
		copy->m_param->add_parameters(m_param);
	} else
		copy->m_param=NULL;

	/* recursively copy all children */
	for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		copy->m_child_nodes->append_element(
				m_child_nodes->get_element(i)->copy_tree());

	return copy;
}
