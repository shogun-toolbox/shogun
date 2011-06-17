/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/ModelSelectionParameters.h"
#include "lib/DataType.h"
#include "base/DynArray.h"
#include "base/Parameter.h"

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
	m_destroy_tree=false;

	m_parameters->add((char*)m_node_name, "node_name", "Name of node");
	m_parameters->add((CSGObject**)&m_sgobject, "sgobject", "CSGObject of this node");
	m_parameters->add(&m_destroy_tree, "destroy_tree",
			"Whether tree of this node is destroyed in destructor");
}

CModelSelectionParameters::~CModelSelectionParameters()
{
	if (m_destroy_tree)
	{
		for (index_t i=0; i<m_child_nodes.get_num_elements(); ++i)
			m_child_nodes[i]->destroy();
	}
	else
	{
		delete[] m_values.vector;
		SG_UNREF(m_sgobject);
	}
}

void CModelSelectionParameters::destroy()
{
	for (index_t i=0; i<m_child_nodes.get_num_elements(); ++i)
		m_child_nodes[i]->destroy();

	delete this;
}

void CModelSelectionParameters::append_child(CModelSelectionParameters* child)
{
	/* only possible if there are no values set */
	if (m_values.vector)
		SG_ERROR("not possible to append child: there already is a range\n");

	/* do a basic check if the add is possible (does this node's CSGObject
	 * contain a parameter with the name of the child?) to prevent problems
	 * when trying to set parameters that do not exist */
	if (m_sgobject)
	{
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
	m_child_nodes.append_element(child);
}

void CModelSelectionParameters::set_values(SGVector<float64_t> values)
{
	/* possibly delete old range values */
	delete[] m_values.vector;

	m_values.vector=values.vector;
	m_values.length=values.length;
}

void CModelSelectionParameters::set_range(float64_t min, float64_t max,
		ERangeType type, float64_t step, float64_t type_base)
{
	if (m_sgobject || has_children())
	{
		SG_ERROR("unable to set range for an CSGObject model selection "
				"parameter\n");
	}

	/* possibly delete old range values */
	delete[] m_values.vector;

	if (max<min)
		SG_ERROR("unable to set range: maximum=%f < minimum=%f\n", max, min);

	/* create value vector */
	index_t num_values=CMath::round(max-min)/step+1;
	m_values.length=num_values;
	m_values.vector=new float64_t[num_values];

	/* fill array */
	for (index_t i=0; i<num_values; ++i)
	{
		float64_t current=min+i*step;

		switch (type)
		{
		case R_LINEAR:
			m_values.vector[i]=current;
			break;
		case R_EXP:
			m_values.vector[i]=CMath::pow(type_base, current);
			break;
		case R_LOG:
			if (current<=0)
				SG_ERROR("log(x) with x=%f\n", current);

			/* custom base b: log_b(i*step)=log_2(i*step)/log_2(b) */
			m_values.vector[i]=CMath::log2(current)/CMath::log2(type_base);
			break;
		default:
			SG_ERROR("unknown range type!\n");
		}
	}
}

void CModelSelectionParameters::get_combinations(
		DynArray<ParameterCombination*>& result)
{
	/* leaf case: node with values and no children.
	 * build trees of Parameter instances which each contain one value
	 */
	if (m_values.vector)
	{
		for (index_t i=0; i<m_values.length; ++i)
		{
			/* create tree with only one parameter element */
			Parameter* p=new Parameter();
			p->add(&m_values.vector[i], m_node_name);

			result.append_element(new ParameterCombination(p));
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
		/* only consider combinations if this CSGObject has children */
		if (m_child_nodes.get_num_elements())
		{
			/* split leaf and no-leaf child combinations */
			DynArray<CModelSelectionParameters*> leaf_children;
			DynArray<CModelSelectionParameters*> non_leaf_children;

			for (index_t i=0; i<m_child_nodes.get_num_elements(); ++i)
			{
				CModelSelectionParameters* current=m_child_nodes[i];

				/* split children with values (leafs) and children with other */
				if (current->m_values.vector)
					leaf_children.append_element(current);
				else
					non_leaf_children.append_element(current);
			}

			/* extract all tree sets of all leaf children */
			DynArray<DynArray<ParameterCombination*>*> leaf_sets;
			for (index_t i=0; i<leaf_children.get_num_elements(); ++i)
			{
				/* temporary DynArray instance */
				leaf_sets.append_element(
						new DynArray<ParameterCombination*> ());

				/* recursively get all combinations and put into above array */
				leaf_children[i]->get_combinations(*leaf_sets[i]);
			}

			/* build product of all these tree sets */
			DynArray<ParameterCombination*> leaf_combinations;

			/* new root node is needed for new trees, depends on current case */
			ParameterCombination* new_root=NULL;
			if (m_sgobject)
			{
				Parameter* p=new Parameter();
				p->add(&m_sgobject, m_node_name);
				new_root=new ParameterCombination(p);
			}
			else
				new_root=new ParameterCombination();

			/* above created DynArray instances are deleted by this call */
			ParameterCombination::leaf_sets_multiplication(leaf_sets,
					new_root, leaf_combinations);

			/* if there are no non-leaf sets, just use the above result */
			if (!non_leaf_children.get_num_elements())
				result=leaf_combinations;
			/* in the other case, the non-leafs have also to be treated, but
			 * combined iteratively */
			else
			{
				/* extract all tree sets of non-leaf nodes */
				DynArray<DynArray<ParameterCombination*>*>
						non_leaf_combinations;
				for (index_t i=0; i<non_leaf_children.get_num_elements(); ++i)
				{
					/* temporary DynArray instance */
					non_leaf_combinations.append_element(
							new DynArray<ParameterCombination*> ());

					/* recursively get all combinations and put into above
					 * array */
					non_leaf_children[i]->get_combinations(
							*non_leaf_combinations[i]);
				}

				/* combine combinations of leafs and non-leafs */

				/* if there are only non-leaf children, nothing is combined */
				if (!leaf_combinations.get_num_elements())
				{
					/* non-leaf children are only pasted together. However, the
					 * new root node is to put as root in front of all trees.
					 * If there were leaf children before, this is done by
					 * leaf_sets_multiplication. In this case it has to be done
					 * by hand. */

					for (index_t j=0; j
							<non_leaf_combinations.get_num_elements(); ++j)
					{
						DynArray<ParameterCombination*>* current_non_leaf_set=
								non_leaf_combinations[j];

						for (index_t k=0; k
								<current_non_leaf_set->get_num_elements(); ++k)
						{
							ParameterCombination* current_non_leaf_tree=
									current_non_leaf_set->get_element(k);

							/* append new root with rest of tree to current
							 * tree. re-use of new_root variable, safe here */
							new_root=new ParameterCombination();
							new_root->append_child(current_non_leaf_tree);
							result.append_element(new_root);
						}
					}

					/* since there were no non-leaf sets, they do not have to be
					 * deleted here */
				}
				else
				{
					for (index_t i=0; i<leaf_combinations.get_num_elements(); ++i)
					{
						ParameterCombination* current_leaf_tree=
								leaf_combinations[i];
						for (index_t j=0; j
								<non_leaf_combinations.get_num_elements(); ++j)
						{
							DynArray<ParameterCombination*>
									* current_non_leaf_set=
											non_leaf_combinations[j];

							for (index_t k=0; k
									<current_non_leaf_set->get_num_elements(); ++k)
							{
								ParameterCombination* current_non_leaf_tree=
										current_non_leaf_set->get_element(k);

								/* copy the current trees and append non-leaf
								 * tree to leaf tree. Note that the root in the
								 * non-leaf tree is already the current
								 * CSGObject and therefore the non-leaf tree
								 * copy may just be appended as child */
								ParameterCombination* leaf_copy=
										current_leaf_tree->copy_tree();
								ParameterCombination* non_leaf_copy=
										current_non_leaf_tree->copy_tree();

								leaf_copy->append_child(non_leaf_copy);
								result.append_element(leaf_copy);
							}
						}
					}

					/* delete non-leaf combination trees */
					for (index_t i=0; i
						<leaf_combinations.get_num_elements(); ++i)
							leaf_combinations[i]->destroy(true, true);

					for (index_t i=0; i
							<non_leaf_combinations.get_num_elements(); ++i)
					{
						DynArray<ParameterCombination*>* current_non_leaf_set=
								non_leaf_combinations[i];

						for (index_t j=0; j
								<current_non_leaf_set->get_num_elements(); ++j)
							current_non_leaf_set->get_element(j)->destroy(true,
									true);

					}

					/* the arrays of the non-leaf sets have to be deleted in
					 * both cases: if there were leaf children or not */
					for (index_t i=0; i
							<non_leaf_combinations.get_num_elements(); ++i)
						delete non_leaf_combinations[i];
				}
			}
		}
	}

	/* case name placeholder node: a node which contains a (parameter) name and
	 * one (or more) CSGObject nodes which are to be substituted into the
	 * parameter with the above name. The parameter name is one of the learning
	 * machine, like "kernel". basically all combinations of all children have to
	 * be appended to the result and a new root is to be added to all trees
	 */
	else if (m_node_name && !m_sgobject && !m_values.vector)
	{
		if (!m_child_nodes.get_num_elements())
		{
			SG_ERROR("ModelSelectionParameter node with name but no children or "
					"values.\n");
		}

		for (index_t i=0; i<m_child_nodes.get_num_elements(); ++i)
		{
			/* recursively get all combinations of the current child */
			DynArray<ParameterCombination*> child_combinations;
			m_child_nodes[i]->get_combinations(child_combinations);

			/* and process them each */
			for (index_t j=0; j<child_combinations.get_num_elements(); ++j)
			{
				/* append new root node with the name */
				ParameterCombination* new_root=new ParameterCombination(
						m_node_name);
				new_root->append_child(child_combinations[j]);
				child_combinations.set_element(new_root, j);

				/* append them to the result */
				result.append_element(child_combinations[j]);
			}
		}
	}
	else
		SG_ERROR("Illegal CModelSelectionParameters node type.\n");
}

void CModelSelectionParameters::print(int prefix_num)
{
	/* prefix is enlarged */
	char* prefix=new char[prefix_num+1];
	for (index_t i=0; i<prefix_num; ++i)
		prefix[i]='\t';

	prefix[prefix_num]='\0';

	if (has_children())
	{
		/* this node might also be a parameter */
		SG_PRINT("%s%s with\n", prefix, m_node_name);

		/* now recursively print successors */

		/* cast safe because only CModelSelectionParameters are added to list */
		for (index_t i=0; i<m_child_nodes.get_num_elements(); ++i)
			m_child_nodes[i]->print(prefix_num+1);
	}
	else
	{
		/* has to be a node with name and a numeric range or a single sg_object
		 * without children*/
		if (m_sgobject)
		{
			SG_PRINT("%s%s\n", prefix, m_node_name);
		}
		else
		{
			SG_PRINT("%s%s with values: ", prefix, m_node_name);
			CMath::display_vector(m_values.vector, m_values.length);
		}
	}

	delete[] prefix;
}

