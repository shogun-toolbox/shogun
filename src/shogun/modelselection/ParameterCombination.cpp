/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/base/Parameter.h>
#include <shogun/machine/Machine.h>

using namespace shogun;

CParameterCombination::CParameterCombination()
{
	init();
}

CParameterCombination::CParameterCombination(Parameter* param)
{
	init();

	m_param=param;
}

void CParameterCombination::init()
{
	m_param=NULL;
	m_child_nodes=new CDynamicObjectArray<CParameterCombination> ();
	SG_REF(m_child_nodes);

	m_parameters->add((CSGObject**)m_child_nodes, "child nodes",
			"children of this node");
}

CParameterCombination::~CParameterCombination()
{
	delete m_param;
	SG_UNREF(m_child_nodes);
}

void CParameterCombination::append_child(CParameterCombination* child)
{
	m_child_nodes->append_element(child);
}

void CParameterCombination::print_tree(int prefix_num) const
{
	/* prefix is enlarged */
	char* prefix=new char[prefix_num+1];
	for (index_t i=0; i<prefix_num; ++i)
		prefix[i]='\t';

	prefix[prefix_num]='\0';

	/* cases:
	 * -node with a Parameter instance and a possible children
	 * -root node with children
	 */

	if (m_param)
	{
		SG_SPRINT("%s", prefix);
		for (index_t i=0; i<m_param->get_num_parameters(); ++i)
		{
			/* distinction between sgobject and values */
			if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_SGOBJECT)
			{
				TParameter* param=m_param->get_parameter(i);
				CSGObject* current_sgobject=*((CSGObject**) param->m_parameter);
				SG_SPRINT("\"%s\":%s at %p ", param->m_name,
						current_sgobject->get_name(), current_sgobject);
			}
			else
			{
				SG_SPRINT("\"%s\"=%f ", m_param->get_parameter(i)->m_name,
						*((float64_t*)m_param->get_parameter(i)->m_parameter));
			}
		}

	}
	else
		SG_SPRINT("%sroot", prefix);

	SG_SPRINT("\n");

	for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
	{
		CParameterCombination* child=m_child_nodes->get_element(i);
		child->print_tree(prefix_num+1);
		SG_UNREF(child);
	}

	SG_FREE(prefix);
}

DynArray<Parameter*>* CParameterCombination::parameter_set_multiplication(
		const DynArray<Parameter*>& set_1, const DynArray<Parameter*>& set_2)
{
	DynArray<Parameter*>* result=new DynArray<Parameter*>();

	for (index_t i=0; i<set_1.get_num_elements(); ++i)
	{
		for (index_t j=0; j<set_2.get_num_elements(); ++j)
		{
			Parameter* p=new Parameter();
			p->add_parameters(set_1[i]);
			p->add_parameters(set_2[j]);
			result->append_element(p);
		}
	}

	return result;
}

CDynamicObjectArray<CParameterCombination>* CParameterCombination::leaf_sets_multiplication(
		const CDynamicObjectArray<CDynamicObjectArray<CParameterCombination> >& sets,
		const CParameterCombination* new_root)
{
	CDynamicObjectArray<CParameterCombination>* result=new CDynamicObjectArray<
			CParameterCombination>();

	/* check marginal cases */
	if (sets.get_num_elements()==1)
	{
		CDynamicObjectArray<CParameterCombination>* current_set=
				sets.get_element(0);

		/* just use the only element into result array.
		 * put root node before all combinations*/
		*result=*current_set;

		SG_UNREF(current_set);

		for (index_t i=0; i<result->get_num_elements(); ++i)
		{
			/* put new root as root into the tree and replace tree */
			CParameterCombination* current=result->get_element(i);
			CParameterCombination* root=new_root->copy_tree();
			root->append_child(current);
			result->set_element(root, i);
			SG_UNREF(current);
		}
	}
	else if (sets.get_num_elements()>1)
	{
		/* now the case where at least two sets are given */

		/* first, extract Parameter instances of given sets */
		DynArray<DynArray<Parameter*>*> param_sets;

		for (index_t set_nr=0; set_nr<sets.get_num_elements(); ++set_nr)
		{
			CDynamicObjectArray<CParameterCombination>* current_set=
					sets.get_element(set_nr);
			DynArray<Parameter*>* new_param_set=new DynArray<Parameter*> ();
			param_sets.append_element(new_param_set);

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
					new_param_set->append_element(current_param);
				else
				{
					SG_SERROR("leaf sets multiplication only possible if all "
							"leafs have non-NULL Parameter instances\n");
				}

				SG_UNREF(current_node);
			}

			SG_UNREF(current_set);
		}

		/* second, build products of all parameter sets */
		DynArray<Parameter*>* param_product=parameter_set_multiplication(
				*param_sets[0], *param_sets[1]);

		delete param_sets[0];
		delete param_sets[1];

		/* build product of all remaining sets and collect results. delete all
		 * parameter instances of interim products*/
		for (index_t i=2; i<param_sets.get_num_elements(); ++i)
		{
			DynArray<Parameter*>* old_temp_result=param_product;
			param_product=parameter_set_multiplication(*param_product,
					*param_sets[i]);

			/* delete interim result parameter instances */
			for (index_t j=0; j<old_temp_result->get_num_elements(); ++j)
				delete old_temp_result->get_element(j);

			/* and dyn arrays of interim result and of param_sets */
			delete old_temp_result;
			delete param_sets[i];
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
			result->append_element(root);
		}

		/* this is not needed anymore, because the Parameter instances are now
		 * in the resulting tree sets */
		delete param_product;
	}

	return result;
}

CParameterCombination* CParameterCombination::copy_tree() const
{
	CParameterCombination* copy=new CParameterCombination();

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
	{
		CParameterCombination* child=m_child_nodes->get_element(i);
		copy->m_child_nodes->append_element(child->copy_tree());
		SG_UNREF(child);
	}

	return copy;
}

void CParameterCombination::apply_to_machine(CMachine* machine) const
{
	apply_to_parameter(machine->m_parameters);
}

void CParameterCombination::apply_to_parameter(Parameter* parameter) const
{
	/* case root node or name node */
	if (!m_param)
	{
		/* iterate over all children and recursively set parameters from
		 * their values to the current parameter input (its just handed one
		 * recursion level downwards) */
		for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		{
			CParameterCombination* child=m_child_nodes->get_element(i);
			child->apply_to_parameter(parameter);
			SG_UNREF(child);
		}
	}
	/* case parameter node */
	else if (m_param)
	{
		/* does this node has sub parameters? */
		if (has_children())
		{
			/* if a parameter node has children, it has to have ONE CSGObject as
			 * parameter */
			if (m_param->get_num_parameters()>1 ||
					m_param->get_parameter(0)->m_datatype.m_ptype!=PT_SGOBJECT)
			{
				SG_SERROR("invalid CParameterCombination node type, has children"
						" and more than one parameter or is not a "
						"CSGObject.\n");
			}

			/* cast is now safe */
			CSGObject* current_sgobject=
					*((CSGObject**)(m_param->get_parameter(0)->m_parameter));

			/* set parameters */
			parameter->set_from_parameters(m_param);

			/* iterate over all children and recursively set parameters from
			 * their values */
			for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
			{
				CParameterCombination* child=m_child_nodes->get_element(i);
				child->apply_to_parameter(current_sgobject->m_parameters);
				SG_UNREF(child);
			}
		}
		else
			parameter->set_from_parameters(m_param);
	}
	else
		SG_SERROR("CParameterCombination node has illegal type.\n");

}
