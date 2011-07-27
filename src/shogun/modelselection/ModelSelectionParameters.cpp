/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/lib/DataType.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/DynArray.h>

using namespace shogun;

CModelSelectionParameters::CModelSelectionParameters()
{
	init();
}

CModelSelectionParameters::CModelSelectionParameters(const char* node_name)
{
	init();

	m_node_name=node_name;
}

CModelSelectionParameters::CModelSelectionParameters(const char* node_name,
		CSGObject* sgobject)
{
	init();

	m_node_name=node_name;
	m_sgobject=sgobject;
	SG_REF(sgobject);
}

void CModelSelectionParameters::init()
{
	m_node_name=NULL;
	m_sgobject=NULL;
	m_child_nodes=new CDynamicObjectArray<CModelSelectionParameters>();
	SG_REF(m_child_nodes);
	m_value_type=MSPT_NONE;

	m_parameters->add((char*)m_node_name, "node_name", "Name of node");
	m_parameters->add((CSGObject**)&m_sgobject, "sgobject",
			"CSGObject of this node");
	m_parameters->add((CSGObject**)m_child_nodes, "child nodes",
			"children of this node");
//	m_parameters->add(&m_value_type, "value_type",
//				"type of the values of this node");
}

CModelSelectionParameters::~CModelSelectionParameters()
{
	SG_UNREF(m_child_nodes);
	SG_UNREF(m_sgobject);

	delete_values();
}

void CModelSelectionParameters::append_child(CModelSelectionParameters* child)
{
	/* only possible if there are no values set */
	if (m_values.vector)
		SG_ERROR("not possible to append child: there already is a range\n");

	/* do a basic check if the add is possible */
	if (m_sgobject)
	{
		/* (does this node's CSGObject contain a parameter with the name of the
		 * child?) to prevent problems when trying to set parameters that do not
		 * exist */
		if (child->m_node_name)
		{
			if (!m_sgobject->m_parameters->contains_parameter(child->m_node_name))
			{
				SG_ERROR("Not possible to add child, node with CSGObject \"%s\""
						" does not contain a parameter called \"%s\"\n",
						m_sgobject->get_name(), child->m_node_name);
			}
		}
		else
		{
			SG_ERROR("Not possible to add child which has no name.\n");
		}
	}

	m_child_nodes->append_element(child);
}

void CModelSelectionParameters::set_values(SGVector<void> values)
{
	/* possibly delete old range values */
	delete_values();
	m_values=values;
}

void CModelSelectionParameters::build_values(float64_t min, float64_t max,
		ERangeType type, float64_t step, float64_t type_base)
{
	build_values(MSPT_FLOAT64, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
}

void CModelSelectionParameters::build_values(int32_t min, int32_t max,
		ERangeType type, int32_t step, int32_t type_base)
{
	build_values(MSPT_INT32, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
}

void CModelSelectionParameters::build_values(EMSParamType value_type, void* min,
		void* max, ERangeType type, void* step, void* type_base)
{
	if (m_sgobject || has_children())
	{
		SG_ERROR("unable to set range for an CSGObject model selection "
				"parameter\n");
	}

	/* possibly delete old range values */
	delete_values();

	/* save new type */
	m_value_type=value_type;

	if (value_type==MSPT_FLOAT64)
	{
		SGVector<float64_t> values=create_range_array<float64_t>(
				*((float64_t*)min),
				*((float64_t*)max),
				type,
				*((float64_t*)step),
				*((float64_t*)type_base));

		m_values.vector=(void*)values.vector;
		m_values.vlen=values.vlen;
	}
	else if (value_type==MSPT_INT32)
	{
		SGVector<int32_t> values=create_range_array<int32_t>(
				*((int32_t*)min),
				*((int32_t*)max),
				type,
				*((int32_t*)step),
				*((int32_t*)type_base));

		m_values.vector=(void*)values.vector;
		m_values.vlen=values.vlen;
	}
	else if (value_type==MSPT_NONE)
	{
		SG_ERROR("Value node has no type!\n");
	}
	else
	{
		SG_ERROR("Unknown type for model selection parameter!\n");
	}
}

CDynamicObjectArray<CParameterCombination>* CModelSelectionParameters::get_combinations()
{
	CDynamicObjectArray<CParameterCombination>* result=new CDynamicObjectArray<
			CParameterCombination>();

	/* value case: node with values and no children.
	 * build trees of Parameter instances which each contain one value
	 */
	if (m_values.vector)
	{
		for (index_t i=0; i<m_values.vlen; ++i)
		{
			/* create tree with only one parameter element */
			Parameter* p=new Parameter();

			switch (m_value_type)
			{
			case MSPT_FLOAT64:
				p->add(&((float64_t*)m_values.vector)[i], m_node_name);
				break;
			case MSPT_INT32:
				p->add(&((int32_t*)m_values.vector)[i], m_node_name);;
				break;
			case MSPT_NONE:
				SG_ERROR("Value node has no type!\n");
				break;
			default:
				SG_ERROR("Unknown type for model selection parameter!\n");
				break;
			}

			result->append_element(new CParameterCombination(p));
		}
	}
	/* two cases here, similar
	 * -case CSGObject:
	 * -case root node (no name, no values, but children
	 * build all permutations of the result trees of children with values and
	 * combine them iteratively children which are something different
	 */
	else if ((m_sgobject && m_node_name) ||
			(!m_node_name && !m_sgobject && !m_values.vector))
	{
		/* only consider combinations if this node has children */
		if (m_child_nodes->get_num_elements())
		{
			/* split value and non-value child combinations */
			CDynamicObjectArray<CModelSelectionParameters> value_children;
			CDynamicObjectArray<CModelSelectionParameters> non_value_children;

			for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
			{
				CModelSelectionParameters* current=m_child_nodes->get_element(i);

				/* split children with values and children with other */
				if (current->m_values.vector)
					value_children.append_element(current);
				else
					non_value_children.append_element(current);

				SG_UNREF(current);
			}

			/* extract all tree sets of all value children */
			CDynamicObjectArray<CDynamicObjectArray<CParameterCombination> > value_node_sets;
			for (index_t i=0; i<value_children.get_num_elements(); ++i)
			{
				/* recursively get all combinations in a new array */
				CModelSelectionParameters* value_child=
						value_children.get_element(i);
				value_node_sets.append_element(value_child->get_combinations());
				SG_UNREF(value_child);
			}

			/* build product of all these tree sets */

			/* new root node is needed for new trees, depends on current case */
			CParameterCombination* new_root=NULL;
			if (m_sgobject)
			{
				Parameter* p=new Parameter();
				p->add(&m_sgobject, m_node_name);
				new_root=new CParameterCombination(p);
			}
			else
				new_root=new CParameterCombination();

			SG_REF(new_root);

			CDynamicObjectArray<CParameterCombination>* value_combinations=
					CParameterCombination::leaf_sets_multiplication(value_node_sets,
							new_root);

			SG_UNREF(new_root);

			/* if there are no non-value sets, just use the above result */
			if (!non_value_children.get_num_elements())
				*result=*value_combinations;
			/* in the other case, the non-values have also to be treated, but
			 * combined iteratively */
			else
			{
				/* extract all tree sets of non-value nodes */
				CDynamicObjectArray<CDynamicObjectArray<CParameterCombination> >
						non_value_combinations;
				for (index_t i=0; i<non_value_children.get_num_elements(); ++i)
				{
					/* recursively get all combinations in a new array */
					CModelSelectionParameters* non_value_child=
							non_value_children.get_element(i);
					non_value_combinations.append_element(
							non_value_child->get_combinations());
					SG_UNREF(non_value_child);
				}

				/* combine combinations of value and non-value nodes */

				/* if there are only non-value children, nothing is combined */
				if (!value_combinations->get_num_elements())
				{
					/* non-value children are only pasted together. However, the
					 * new root node is to put as root in front of all trees.
					 * If there were value children before, this is done by
					 * value_node_sets_multiplication. In this case it has to be done
					 * by hand. */

					for (index_t j=0;
							j<non_value_combinations.get_num_elements(); ++j)
					{
						CDynamicObjectArray<CParameterCombination>* current_non_value_set=
								non_value_combinations.get_element(j);

						for (index_t k=0; k
								<current_non_value_set->get_num_elements(); ++k)
						{
							CParameterCombination* current_non_value_tree=
									current_non_value_set->get_element(k);

							/* append new root with rest of tree to current
							 * tree. re-use of new_root variable, safe here */
							new_root=new CParameterCombination();
							new_root->append_child(current_non_value_tree);
							result->append_element(new_root);

							SG_UNREF(current_non_value_tree);
						}

						SG_UNREF(current_non_value_set);
					}
				}
				else
				{
					for (index_t i=0; i<value_combinations->get_num_elements(); ++i)
					{
						CParameterCombination* current_value_tree=
								value_combinations->get_element(i);

						for (index_t j=0; j
								<non_value_combinations.get_num_elements(); ++j)
						{
							CDynamicObjectArray<CParameterCombination> * current_non_value_set=
									non_value_combinations.get_element(j);

							for (index_t k=0; k
									<current_non_value_set->get_num_elements(); ++k)
							{
								CParameterCombination* current_non_value_tree=
										current_non_value_set->get_element(k);

								/* copy the current trees and append non-value
								 * tree to value tree. Note that the root in the
								 * non-value tree is already the current
								 * CSGObject and therefore the non-value tree
								 * copy may just be appended as child */
								CParameterCombination* value_copy=
										current_value_tree->copy_tree();
								CParameterCombination* non_value_copy=
										current_non_value_tree->copy_tree();

								value_copy->append_child(non_value_copy);
								result->append_element(value_copy);

								SG_UNREF(current_non_value_tree);
							}

							SG_UNREF(current_non_value_set);
						}

						SG_UNREF(current_value_tree);
					}
				}
			}

			SG_UNREF(value_combinations);
		}
		else
		{
			/* if there are no children of a sgobject or root node, result is
			 * only one element (sgobject node) or empty (root node)
			 */
			if (m_sgobject)
			{
				Parameter* p=new Parameter();
				p->add(&m_sgobject, m_node_name);
				result->append_element(new CParameterCombination(p));
			}
		}

	}
	else
		SG_ERROR("Illegal CModelSelectionParameters node type.\n");

	return result;
}

void CModelSelectionParameters::print_tree(int prefix_num)
{
	/* prefix is enlarged */
	char* prefix=SG_MALLOC(char, prefix_num+1);
	for (index_t i=0; i<prefix_num; ++i)
		prefix[i]='\t';

	prefix[prefix_num]='\0';

	if (has_children())
	{
		if (m_sgobject)
			SG_PRINT("%s%s:\"%s\"\n", prefix, m_node_name, m_sgobject->get_name());
		else
			SG_PRINT("%s%s with\n", prefix, m_node_name ? m_node_name : "root");

		/* now recursively print successors */

		/* cast safe because only CModelSelectionParameters are added to list */
		for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		{
			CModelSelectionParameters* child=m_child_nodes->get_element(i);
			child->print_tree(prefix_num+1);
			SG_UNREF(child);
		}
	}
	else
	{
		/* has to be a node with name and a numeric range or a single sg_object
		 * without children*/
		if (m_sgobject)
		{
			SG_PRINT("%s%s:\"%s\"\n", prefix, m_node_name, m_sgobject->get_name());
		}
		else
		{

			if (m_values.vector)
			{
				/* value node */
				SG_PRINT("%s%s with values: ", prefix, m_node_name);

				switch (m_value_type)
				{
				case MSPT_FLOAT64:
					CMath::display_vector((float64_t*)m_values.vector, m_values.vlen);
					break;
				case MSPT_INT32:
					CMath::display_vector((int32_t*)m_values.vector, m_values.vlen);;
					break;
				case MSPT_NONE:
					SG_ERROR("Value node has no type!\n");
					break;
				default:
					SG_ERROR("Unknown type for model selection parameter!\n");
					break;
				}
			}
			else
				SG_PRINT("root\n");
		}
	}

	SG_FREE(prefix);
}

void CModelSelectionParameters::delete_values()
{
	if (m_values.vector)
	{
		switch (m_value_type)
		{
		case MSPT_FLOAT64:
			SG_FREE((float64_t*) m_values.vector);
			break;
		case MSPT_INT32:
			SG_FREE((int32_t*) m_values.vector);
			break;
		case MSPT_NONE:
			SG_ERROR("Value node has no type!\n");
			break;
		default:
			SG_ERROR("Unknown type for model selection parameter!\n");
			break;
		}
	}
}
