/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Written (W) 2012 Jacob Walker
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/base/Parameter.h>
#include <shogun/machine/Machine.h>
#include <set>
#include <string>

using namespace shogun;
using namespace std;

CParameterCombination::CParameterCombination()
{
	init();
}

CParameterCombination::CParameterCombination(Parameter* param)
{
	init();

	m_param=param;
}

CParameterCombination::CParameterCombination(CSGObject* obj)
{
	init();

	Parameter* gradient_params=obj->m_gradient_parameters;

	for (index_t i=0; i<gradient_params->get_num_parameters(); i++)
	{
		TParameter* param=gradient_params->get_parameter(i);
		TSGDataType type=param->m_datatype;

		if (type.m_ptype==PT_FLOAT64 || type.m_ptype==PT_FLOAT32 ||
			type.m_ptype==PT_FLOATMAX)
		{
			if ((type.m_ctype==CT_SGVECTOR || type.m_ctype==CT_VECTOR))
			{
				Parameter* p=new Parameter();
				p->add_vector((float64_t**)param->m_parameter, type.m_length_y,
						param->m_name);

				m_child_nodes->append_element(new CParameterCombination(p));
				m_parameters_length+=*(type.m_length_y);
			}
			else if (type.m_ctype==CT_SCALAR)
			{
				Parameter* p=new Parameter();
				p->add((float64_t*)param->m_parameter, param->m_name);

				m_child_nodes->append_element(new CParameterCombination(p));
				m_parameters_length++;
			}
		}
		else
		{
			SG_WARNING("Parameter %s.%s was not added to parameter combination, "
					"since it isn't of floating point type\n", obj->get_name(),
					param->m_name);
		}
	}

	Parameter* modsel_params=obj->m_model_selection_parameters;

	for (index_t i=0; i<modsel_params->get_num_parameters(); i++)
	{
		TParameter* param=modsel_params->get_parameter(i);
		TSGDataType type=param->m_datatype;

		if (type.m_ptype==PT_SGOBJECT)
		{
			if (type.m_ctype==CT_SCALAR)
			{
				CSGObject* child=*((CSGObject**)(param->m_parameter));

				if (child->m_gradient_parameters->get_num_parameters()>0)
				{
					CParameterCombination* comb=new CParameterCombination(child);

					comb->m_param=new Parameter();
					comb->m_param->add((CSGObject**)(param->m_parameter),
							param->m_name);

					m_child_nodes->append_element(comb);
					m_parameters_length+=comb->m_parameters_length;
				}
			}
			else
			{
				SG_NOTIMPLEMENTED
			}
		}
	}
}

void CParameterCombination::init()
{
	m_parameters_length=0;
	m_param=NULL;
	m_child_nodes=new CDynamicObjectArray();
	SG_REF(m_child_nodes);

	SG_ADD((CSGObject**)&m_child_nodes, "child_nodes", "Children of this node",
			MS_NOT_AVAILABLE);
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

bool CParameterCombination::set_parameter_helper(
		const char* name, bool value, index_t index)
{
	if (m_param)
	{
		for (index_t i = 0; i < m_param->get_num_parameters(); ++i)
		{
			void* param = m_param->get_parameter(i)->m_parameter;

			if (!strcmp(m_param->get_parameter(i)->m_name, name))
			{
				if (m_param->get_parameter(i)->m_datatype.m_ptype
						!= PT_BOOL)
					SG_ERROR("Parameter %s not a boolean parameter", name)

				if (index < 0)
					*((bool*)(param)) = value;

				else
					(*((bool**)(param)))[index] = value;

				return true;
			}
		}

	}

	return false;
}

bool CParameterCombination::set_parameter_helper(
		const char* name, int32_t value, index_t index)
{
	if (m_param)
	{
		for (index_t i = 0; i < m_param->get_num_parameters(); ++i)
		{
			void* param = m_param->get_parameter(i)->m_parameter;

			if (!strcmp(m_param->get_parameter(i)->m_name, name))
			{
				if (m_param->get_parameter(i)->m_datatype.m_ptype
						!= PT_INT32)
					SG_ERROR("Parameter %s not a integer parameter", name)

				if (index < 0)
					*((int32_t*)(param)) = value;

				else
					(*((int32_t**)(param)))[index] = value;

				return true;
			}
		}
	}

	return false;
}

bool CParameterCombination::set_parameter_helper(
		const char* name, float64_t value, index_t index)
{
	if (m_param)
	{
		for (index_t i = 0; i < m_param->get_num_parameters(); ++i)
		{
			void* param = m_param->get_parameter(i)->m_parameter;

			if (!strcmp(m_param->get_parameter(i)->m_name, name))
			{
				if (m_param->get_parameter(i)->m_datatype.m_ptype
						!= PT_FLOAT64)
					SG_ERROR("Parameter %s not a double parameter", name)

				if (index < 0)
					*((float64_t*)(param)) = value;

				else
					(*((float64_t**)(param)))[index] = value;

				return true;
			}
		}

	}

	return false;
}


TParameter* CParameterCombination::get_parameter_helper(const char* name)
{
	if (m_param)
	{
		for (index_t i = 0; i < m_param->get_num_parameters(); i++)
		{
			if (!strcmp(m_param->get_parameter(i)->m_name, name))
					return m_param->get_parameter(i);
		}

	}

	return NULL;
}


TParameter* CParameterCombination::get_parameter(const char* name,
		CSGObject* parent)
{
	bool match = false;

	if (m_param)
	{
		for (index_t i = 0; i < m_param->get_num_parameters(); i++)
		{
			if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_SGOBJECT)
			{
				CSGObject* obj =
						(*((CSGObject**)m_param->get_parameter(i)->m_parameter));
				if (parent == obj)
					match = true;
			}
		}

	}

	for (index_t i = 0; i < m_child_nodes->get_num_elements(); ++i)
	{
		CParameterCombination* child = (CParameterCombination*)
				m_child_nodes->get_element(i);

		TParameter* p;

		if (!match)
			 p = child->get_parameter(name, parent);

		else
			 p = child->get_parameter_helper(name);

		if (p)
		{
			SG_UNREF(child);
			return p;
		}

		SG_UNREF(child);
	}

	return NULL;
}


void CParameterCombination::merge_with(CParameterCombination* node)
{
	for (index_t i=0; i<node->m_child_nodes->get_num_elements(); ++i)
	{
		CParameterCombination* child=
				(CParameterCombination*)node->m_child_nodes->get_element(i);
		append_child(child->copy_tree());
		SG_UNREF(child);
	}
}

void CParameterCombination::print_tree(int prefix_num) const
{
	/* prefix is enlarged */
	char* prefix=SG_MALLOC(char, prefix_num+1);
	for (index_t i=0; i<prefix_num; ++i)
		prefix[i]='\t';

	prefix[prefix_num]='\0';

	/* cases:
	 * -node with a Parameter instance and a possible children
	 * -root node with children
	 */

	if (m_param)
	{
		SG_SPRINT("%s", prefix)
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

		    else if (m_param->get_parameter(i)->m_datatype.m_ctype == CT_SGVECTOR)
		    {
				SG_SPRINT("\"%s\"=", m_param->get_parameter(i)->m_name)
			float64_t** param = (float64_t**)(m_param->
					get_parameter(i)->m_parameter);
			if (!m_param->get_parameter(i)->m_datatype.m_length_y)
			{
				SG_ERROR("Parameter vector %s has no length\n",
						m_param->get_parameter(i)->m_name);
			}

			index_t length = *(m_param->get_parameter(i)->m_datatype.m_length_y);

			for (index_t j = 0; j < length; j++)
				SG_SPRINT("%f ", (*param)[j])
		    }

			else
			{
				SG_SPRINT("\"%s\"=", m_param->get_parameter(i)->m_name)
				void* param=m_param->get_parameter(i)->m_parameter;

				if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_FLOAT64)
					SG_SPRINT("%f ", *((float64_t*)param))
				else if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_INT32)
					SG_SPRINT("%i ", *((int32_t*)param))
				else if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_BOOL)
					SG_SPRINT("%s ", *((bool*)param ? "true" : "false"))
				else
					SG_NOTIMPLEMENTED
			}

		}

	}
	else
		SG_SPRINT("%sroot", prefix)

	SG_SPRINT("\n")

	for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
	{
		CParameterCombination* child=(CParameterCombination*)
				m_child_nodes->get_element(i);
		child->print_tree(prefix_num+1);
		SG_UNREF(child);
	}

	SG_FREE(prefix);
}

DynArray<Parameter*>* CParameterCombination::parameter_set_multiplication(
		const DynArray<Parameter*>& set_1, const DynArray<Parameter*>& set_2)
{
	SG_SDEBUG("entering CParameterCombination::parameter_set_multiplication()\n")

	SG_SDEBUG("set 1:\n")
	for (index_t i=0; i<set_1.get_num_elements(); ++i)
	{
		for (index_t j=0; j<set_1.get_element(i)->get_num_parameters(); ++j)
			SG_SDEBUG("\t%s\n", set_1.get_element(i)->get_parameter(j)->m_name)
	}

	SG_SDEBUG("set 2:\n")
	for (index_t i=0; i<set_2.get_num_elements(); ++i)
	{
		for (index_t j=0; j<set_2.get_element(i)->get_num_parameters(); ++j)
			SG_SDEBUG("\t%s\n", set_2.get_element(i)->get_parameter(j)->m_name)
	}

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

	SG_SDEBUG("leaving CParameterCombination::parameter_set_multiplication()\n")
	return result;
}

CDynamicObjectArray* CParameterCombination::leaf_sets_multiplication(
		const CDynamicObjectArray& sets, const CParameterCombination* new_root)
{
	CDynamicObjectArray* result=new CDynamicObjectArray();

	/* check marginal cases */
	if (sets.get_num_elements()==1)
	{
		CDynamicObjectArray* current_set=
				(CDynamicObjectArray*)sets.get_element(0);

		/* just use the only element into result array.
		 * put root node before all combinations*/
		*result=*current_set;

		SG_UNREF(current_set);

		for (index_t i=0; i<result->get_num_elements(); ++i)
		{
			/* put new root as root into the tree and replace tree */
			CParameterCombination* current=(CParameterCombination*)
					result->get_element(i);
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
			CDynamicObjectArray* current_set=(CDynamicObjectArray*)
					sets.get_element(set_nr);
			DynArray<Parameter*>* new_param_set=new DynArray<Parameter*> ();
			param_sets.append_element(new_param_set);

			for (index_t i=0; i<current_set->get_num_elements(); ++i)
			{
				CParameterCombination* current_node=(CParameterCombination*)
						current_set->get_element(i);

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

CDynamicObjectArray* CParameterCombination::non_value_tree_multiplication(
				const CDynamicObjectArray* sets,
				const CParameterCombination* new_root)
{
	SG_SDEBUG("entering CParameterCombination::non_value_tree_multiplication()\n")
	CDynamicObjectArray* result=new CDynamicObjectArray();

	/* first step: get all names in the sets */
	set<string> names;

	for (index_t j=0;
			j<sets->get_num_elements(); ++j)
	{
		CDynamicObjectArray* current_set=
				(CDynamicObjectArray*)
				sets->get_element(j);

		for (index_t k=0; k
				<current_set->get_num_elements(); ++k)
		{
			CParameterCombination* current_tree=(CParameterCombination*)
					current_set->get_element(k);

			names.insert(string(current_tree->m_param->get_parameter(0)->m_name));

			SG_UNREF(current_tree);
		}

		SG_UNREF(current_set);
	}

	SG_SDEBUG("all names\n")
	for (set<string>::iterator it=names.begin(); it!=names.end(); ++it)
		SG_SDEBUG("\"%s\"\n", (*it).c_str())

	/* only do stuff if there are names */
	if (!names.empty())
	{
		/* next step, build temporary structure where all elements with first
		 * name are put. Elements of that structure will be extend iteratively
		 * per name */


		/* extract all trees with first name */
		const char* first_name=(*(names.begin())).c_str();
		CDynamicObjectArray* trees=
				CParameterCombination::extract_trees_with_name(sets, first_name);

		SG_SDEBUG("adding trees for first name \"%s\":\n", first_name)
		for (index_t i=0; i<trees->get_num_elements(); ++i)
		{
			CParameterCombination* current_tree=
					(CParameterCombination*)trees->get_element(i);

			CParameterCombination* current_root=new_root->copy_tree();
			current_root->append_child(current_tree);
			result->append_element(current_root);

			// current_tree->print_tree(1);
			SG_UNREF(current_tree);
		}
		SG_UNREF(trees);

		/* now iterate over the remaining names and build products */
		SG_SDEBUG("building products with remaining trees:\n")
		set<string>::iterator it=names.begin();
		for (++it; it!=names.end(); ++it)
		{
			SG_SDEBUG("processing \"%s\"\n", (*it).c_str())

			/* extract all trees with current name */
			const char* current_name=(*it).c_str();
			trees=CParameterCombination::extract_trees_with_name(sets,
					current_name);

			/* create new set of trees where each element is put once for each
			 * of the just generated trees */
			CDynamicObjectArray* new_result=new CDynamicObjectArray();
			for (index_t i=0; i<result->get_num_elements(); ++i)
			{
				for (index_t j=0; j<trees->get_num_elements(); ++j)
				{
					CParameterCombination* to_copy=
							(CParameterCombination*)result->get_element(i);

					/* create a copy of current element */
					CParameterCombination* new_element=to_copy->copy_tree();
					SG_UNREF(to_copy);

					CParameterCombination* to_add=
							(CParameterCombination*)trees->get_element(j);
					new_element->append_child(to_add);
					SG_UNREF(to_add);
					new_result->append_element(new_element);
					// SG_SDEBUG("added:\n")
					// new_element->print_tree();
				}
			}

			/* clean up */
			SG_UNREF(trees);

			/* replace result by new_result */
			SG_UNREF(result);
			result=new_result;
		}
	}

	SG_SDEBUG("leaving CParameterCombination::non_value_tree_multiplication()\n")
	return result;
}

CDynamicObjectArray* CParameterCombination::extract_trees_with_name(
		const CDynamicObjectArray* sets, const char* desired_name)
{
	CDynamicObjectArray* result=new CDynamicObjectArray();

	for (index_t j=0;
			j<sets->get_num_elements(); ++j)
	{
		CDynamicObjectArray* current_set=
				(CDynamicObjectArray*) sets->get_element(j);

		for (index_t k=0; k<current_set->get_num_elements(); ++k)
		{
			CParameterCombination* current_tree=(CParameterCombination*)
					current_set->get_element(k);

			char* current_name=current_tree->m_param->get_parameter(0)->m_name;

			if (!strcmp(current_name, desired_name))
				result->append_element(current_tree);

			SG_UNREF(current_tree);
		}

		SG_UNREF(current_set);
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
		CParameterCombination* child=(CParameterCombination*)
				m_child_nodes->get_element(i);
		copy->m_child_nodes->append_element(child->copy_tree());
		SG_UNREF(child);
	}

	return copy;
}

void CParameterCombination::apply_to_machine(CMachine* machine) const
{
	apply_to_modsel_parameter(machine->m_model_selection_parameters);
}

void CParameterCombination::apply_to_modsel_parameter(
		Parameter* parameter) const
{
	/* case root node */
	if (!m_param)
	{
		/* iterate over all children and recursively set parameters from
		 * their values to the current parameter input (its just handed one
		 * recursion level downwards) */
		for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		{
			CParameterCombination* child=(CParameterCombination*)
					m_child_nodes->get_element(i);
			child->apply_to_modsel_parameter(parameter);
			SG_UNREF(child);
		}
	}
	/* case parameter node */
	else if (m_param)
	{
		/* set parameters */
		parameter->set_from_parameters(m_param);

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

			/* iterate over all children and recursively set parameters from
			 * their values */
			for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
			{
				CParameterCombination* child=(CParameterCombination*)
						m_child_nodes->get_element(i);
				child->apply_to_modsel_parameter(
						current_sgobject->m_model_selection_parameters);
				SG_UNREF(child);
			}
		}
	}
	else
		SG_SERROR("CParameterCombination node has illegal type.\n")
}

void CParameterCombination::build_parameter_values_map(
		CMap<TParameter*, SGVector<float64_t> >* dict)
{
	if (m_param)
	{
		for (index_t i=0; i<m_param->get_num_parameters(); i++)
		{
			TParameter* param=m_param->get_parameter(i);
			TSGDataType type=param->m_datatype;

			if (type.m_ptype==PT_FLOAT64 || type.m_ptype==PT_FLOAT32 ||
					type.m_ptype==PT_FLOATMAX)
			{
				if ((type.m_ctype==CT_SGVECTOR || type.m_ctype==CT_VECTOR))
				{
					SGVector<float64_t> value(*((float64_t **)param->m_parameter),
							(*type.m_length_y));
					dict->add(param, value);
				}
				else if (type.m_ctype==CT_SCALAR)
				{
					SGVector<float64_t> value(1);
					value.set_const(*((float64_t *)param->m_parameter));
					dict->add(param, value);
				}
			}
		}
	}

	for (index_t i=0; i<m_child_nodes->get_num_elements(); i++)
	{
		CParameterCombination* child=(CParameterCombination*)
			m_child_nodes->get_element(i);
		child->build_parameter_values_map(dict);
		SG_UNREF(child);
	}
}

void CParameterCombination::build_parameter_parent_map(
		CMap<TParameter*, CSGObject*>* dict)
{
	CSGObject* parent=NULL;

	if (m_param)
	{
		for (index_t i=0; i<m_param->get_num_parameters(); i++)
		{
			TParameter* param=m_param->get_parameter(i);
			TSGDataType type=param->m_datatype;

			if (type.m_ptype==PT_SGOBJECT)
			{
				if (type.m_ctype==CT_SCALAR)
				{
					parent=(*(CSGObject**)param->m_parameter);
					break;
				}
				else
				{
					SG_NOTIMPLEMENTED
				}
			}
		}
	}

	for (index_t i=0; i<m_child_nodes->get_num_elements(); i++)
	{
		CParameterCombination* child=(CParameterCombination*)
			m_child_nodes->get_element(i);

		for (index_t j=0; j<child->m_param->get_num_parameters(); j++)
		{
			TParameter* param=child->m_param->get_parameter(j);
			TSGDataType type=param->m_datatype;

			if (type.m_ptype==PT_SGOBJECT)
			{
				if (type.m_ctype==CT_SCALAR)
				{
					child->build_parameter_parent_map(dict);
				}
				else
				{
					SG_NOTIMPLEMENTED
				}
			}
			else
			{
				dict->add(param, parent);
			}
		}
		SG_UNREF(child);
	}
}
