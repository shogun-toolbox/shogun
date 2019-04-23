/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Jacob Walker, Heiko Strathmann, Evgeniy Andreev,
 *          Bjoern Esser, Leon Kuchenbecker
 */

#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/lib/DataType.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/DynArray.h>
#include <shogun/lib/Set.h>
#include <shogun/mathematics/UniformIntDistribution.h>

using namespace shogun;

ModelSelectionParameters::ModelSelectionParameters()
{
	init();
}

ModelSelectionParameters::ModelSelectionParameters(const char* node_name)
{
	init();

	m_node_name=node_name;
}

ModelSelectionParameters::ModelSelectionParameters(const char* node_name,
		std::shared_ptr<SGObject> sgobject)
{
	init();

	m_node_name=node_name;
	m_sgobject=sgobject;

}

void ModelSelectionParameters::init()
{
	m_node_name=NULL;
	m_sgobject=NULL;
	m_child_nodes=std::make_shared<DynamicObjectArray>();

	m_value_type=MSPT_NONE;
	m_values=NULL;
	m_values_length=0;

	/* no parameter registering. These parameter nodes will not be serialized */
}

ModelSelectionParameters::~ModelSelectionParameters()
{



	delete_values();
}

void ModelSelectionParameters::append_child(std::shared_ptr<ModelSelectionParameters> child)
{
	/* only possible if there are no values set */
	if (m_values)
		error("not possible to append child: there already is a range");

	/* do a basic check if the add is possible */
	if (m_sgobject)
	{
		/* (does this node's SGObject contain a parameter with the name of the
		 * child?) to prevent problems when trying to set parameters that do not
		 * exist */
		if (child->m_node_name)
		{
			if (!m_sgobject->m_parameters->contains_parameter(child->m_node_name))
			{
				error("Not possible to add child, node with SGObject \"{}\""
						" does not contain a parameter called \"{}\"",
						m_sgobject->get_name(), child->m_node_name);
			}
		}
		else
		{
			error("Not possible to add child which has no name.");
		}
	}

	m_child_nodes->append_element(child);
}

void ModelSelectionParameters::build_values(float64_t min, float64_t max,
		ERangeType type, float64_t step, float64_t type_base)
{
	build_values(MSPT_FLOAT64, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
}

void ModelSelectionParameters::build_values_vector(float64_t min, float64_t max,
		ERangeType type, void* vector, index_t* size, float64_t step, float64_t type_base)
{
	build_values(MSPT_FLOAT64_VECTOR, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
	m_vector_length = size;
	m_vector = vector;
}

void ModelSelectionParameters::build_values_sgvector(float64_t min, float64_t max,
		ERangeType type, void* vector, float64_t step, float64_t type_base)
{
	build_values(MSPT_FLOAT64_SGVECTOR, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
	m_vector = vector;
}

void ModelSelectionParameters::build_values(int32_t min, int32_t max,
		ERangeType type, int32_t step, int32_t type_base)
{
	build_values(MSPT_INT32, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
}

void ModelSelectionParameters::build_values_vector(int32_t min, int32_t max,
		ERangeType type, void* vector, index_t* size, int32_t step, int32_t type_base)
{
	build_values(MSPT_INT32_VECTOR, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
	m_vector_length = size;
	m_vector = vector;
}

void ModelSelectionParameters::build_values_sgvector(int32_t min, int32_t max,
		ERangeType type, void* vector, int32_t step, int32_t type_base)
{
	build_values(MSPT_INT32_SGVECTOR, (void*)&min, (void*)&max, type, (void*)&step,
			(void*)&type_base);
	m_vector = vector;
}

void ModelSelectionParameters::build_values(EMSParamType value_type, void* min,
		void* max, ERangeType type, void* step, void* type_base)
{
	if (m_sgobject || has_children())
	{
		error("unable to set range for an SGObject model selection "
				"parameter");
	}

	/* possibly delete old range values */
	delete_values();

	/* save new type */
	m_value_type=value_type;

	if (value_type==MSPT_FLOAT64 ||
			value_type==MSPT_FLOAT64_VECTOR
			|| value_type==MSPT_FLOAT64_SGVECTOR)
	{
		SGVector<float64_t> values=create_range_array<float64_t>(
				*((float64_t*)min),
				*((float64_t*)max),
				type,
				*((float64_t*)step),
				*((float64_t*)type_base));

		m_values=values.vector;
		m_values_length=values.vlen;
	}
	else if (value_type==MSPT_INT32 ||
			value_type==MSPT_INT32_VECTOR
			|| value_type==MSPT_INT32_SGVECTOR)
	{
		SGVector<int32_t> values=create_range_array<int32_t>(
				*((int32_t*)min),
				*((int32_t*)max),
				type,
				*((int32_t*)step),
				*((int32_t*)type_base));

		m_values=values.vector;
		m_values_length=values.vlen;
	}
	else if (value_type==MSPT_NONE)
	{
		error("Value node has no type!");
	}
	else
	{
		error("Unknown type for model selection parameter!");
	}
}

std::shared_ptr<ParameterCombination> ModelSelectionParameters::get_single_combination(
		bool is_rand)
{
	/* If this is a value node, then randomly pick a value from the built
	 * range */
	if (m_values)
	{
		UniformIntDistribution<int32_t> uniform_int_dist(0, m_values_length-1);
		index_t i = 0;

		if (is_rand)
			i = uniform_int_dist(m_prng);

		Parameter* p=new Parameter();

		switch (m_value_type)
		{
		case MSPT_FLOAT64_SGVECTOR:
		{
			SGVector<float64_t>* param_vect = (SGVector<float64_t>*)m_vector;

			for (index_t j = 0; j < param_vect->vlen; j++)
			{
				if (is_rand)
					i = uniform_int_dist(m_prng);
				(*param_vect)[j] = ((float64_t*)m_values)[i];
			}
			p->add(param_vect, m_node_name);
			break;
		}
		case MSPT_FLOAT64_VECTOR:
		{
			float64_t* param_vect = (float64_t*)m_vector;

			for (index_t j = 0; j < *m_vector_length; j++)
			{
				if (is_rand)
					i = uniform_int_dist(m_prng);
				(param_vect)[j] = ((float64_t*)m_values)[i];
			}
			p->add_vector(&param_vect, m_vector_length, m_node_name);
			/*watch_param(m_node_name, &param_vect, m_vector_length)*/;

			break;
		}
		case MSPT_INT32_SGVECTOR:
		{
			SGVector<int32_t>* param_vect = (SGVector<int32_t>*)m_vector;

			for (index_t j = 0; j < param_vect->vlen; j++)
			{
				if (is_rand)
					i = uniform_int_dist(m_prng);
				(*param_vect)[j] = ((int32_t*)m_values)[i];
			}
			p->add(param_vect, m_node_name);
			break;
		}
		case MSPT_INT32_VECTOR:
		{
			int32_t* param_vect = (int32_t*)m_vector;

			for (index_t j = 0; j < *m_vector_length; j++)
			{
				if (is_rand)
					i = uniform_int_dist(m_prng);
				(param_vect)[j] = ((int32_t*)m_values)[i];
			}
			p->add_vector(&param_vect, m_vector_length, m_node_name);
			/*watch_param(m_node_name, &param_vect, m_vector_length)*/;

			break;
		}
		case MSPT_FLOAT64:
			p->add(&((float64_t*)m_values)[i], m_node_name);
			break;
		case MSPT_INT32:
			p->add(&((int32_t*)m_values)[i], m_node_name);;
			break;
		case MSPT_NONE:
			error("Value node has no type!");
			break;
		default:
			error("Unknown type for model selection parameter!");
			break;
		}

		return std::make_shared<ParameterCombination>(p);
	}

	std::shared_ptr<ParameterCombination> new_root=NULL;

	/*Complain if we have a bad node*/
	if (!((m_sgobject && m_node_name) || (!m_node_name && !m_sgobject)))
		error("Illegal ModelSelectionParameters node type.");

	/* Incorporate SGObject and root nodes with children*/
	if (m_child_nodes->get_num_elements())
	{

		if (m_sgobject)
		{
			Parameter* p=new Parameter();
			//FIXME
			//p->add(&m_sgobject, m_node_name);
			new_root = std::make_shared<ParameterCombination>(p);
		}

		else
			new_root = std::make_shared<ParameterCombination>();

		for (index_t i = 0; i < m_child_nodes->get_num_elements(); ++i)
		{
			auto current =
				m_child_nodes->get_element<ModelSelectionParameters>(i);

			auto c = current->get_single_combination(is_rand);

			new_root->append_child(c);


		}

		return new_root;
	}

	/*Incorporate childless nodes*/
	else
	{

		if (m_sgobject)
		{
			Parameter* p = new Parameter();
			//FIXME
			//p->add(&m_sgobject, m_node_name);
			return std::make_shared<ParameterCombination>(p);
		}

		else
		{
			new_root = std::make_shared<ParameterCombination>();
			return new_root;
		}
	}

}



std::shared_ptr<DynamicObjectArray> ModelSelectionParameters::get_combinations(
		index_t num_prefix)
{
	char* prefix=SG_MALLOC(char, num_prefix+1);
	prefix[num_prefix]='\0';
	for (index_t i=0; i<num_prefix; ++i)
		prefix[i]='\t';

	SG_DEBUG("{}------>entering ModelSelectionParameters::get_combinations() "
			"for \"{}\"", prefix, m_node_name ? m_node_name : "root");
	auto result=std::make_shared<DynamicObjectArray>();

	/* value case: node with values and no children.
	 * build trees of Parameter instances which each contain one value
	 */

	if (m_values)
	{
		for (index_t i=0; i<m_values_length; ++i)
		{
			// create tree with only one parameter element //
			Parameter* p=new Parameter();

			switch (m_value_type)
			{
			case MSPT_FLOAT64:
				p->add(&((float64_t*)m_values)[i], m_node_name);
				break;
			case MSPT_INT32:
				p->add(&((int32_t*)m_values)[i], m_node_name);;
				break;
			case MSPT_NONE:
				error("{}Value node has no type!", prefix);
				break;
			default:
				error("{}Unknown type for model selection parameter!",
						prefix);
				break;
			}

			result->append_element(std::make_shared<ParameterCombination>(p));
		}

		SG_DEBUG("{}------>leaving ModelSelectionParameters::get_combinations()"
				"for \"{}\"", prefix, m_node_name ? m_node_name : "root");
		SG_FREE(prefix);
		return result;
	}


	/* two cases here, similar
	 * -case SGObject:
	 * -case root node (no name, no values, but children
	 * build all permutations of the result trees of children with values and
	 * combine them iteratively children which are something different
	 */
	if (!((m_sgobject && m_node_name) || (!m_node_name && !m_sgobject)))
		error("{}Illegal ModelSelectionParameters node type.", prefix);

	/* only consider combinations if this node has children */
	if (m_child_nodes->get_num_elements())
	{
		/* split value and non-value child combinations */
		DynamicObjectArray value_children;
		DynamicObjectArray non_value_children;

		for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		{
			auto current=
					m_child_nodes->get_element<ModelSelectionParameters>(i);

			/* split children with values and children with other */
			if (current->m_values)
				value_children.append_element(current);
			else
				non_value_children.append_element(current);


		}

		/* extract all tree sets of all value children */
		DynamicObjectArray value_node_sets;
		for (index_t i=0; i<value_children.get_num_elements(); ++i)
		{
			/* recursively get all combinations in a new array */
			auto value_child=
					value_children.get_element<ModelSelectionParameters>(i);
			value_node_sets.append_element(value_child->get_combinations(
					num_prefix+1));

		}

		/* build product of all these tree sets */

		/* new root node is needed for new trees, depends on current case */
		std::shared_ptr<ParameterCombination> new_root=NULL;
		if (m_sgobject)
		{
			Parameter* p=new Parameter();
			//FIXME
			//p->add(&m_sgobject, m_node_name);
			new_root=std::make_shared<ParameterCombination>(p);
		}
		else
			new_root=std::make_shared<ParameterCombination>();



		auto value_combinations=
				ParameterCombination::leaf_sets_multiplication(value_node_sets,
						new_root);



		if (!non_value_children.get_num_elements())
			*result=*value_combinations;
		/* in the other case, the non-values have also to be treated, but
		 * combined iteratively */
		else
		{
			/* extract all tree sets of non-value nodes */
//			io::print("{}extracting combinations of non-value nodes\n", prefix);
			auto non_value_combinations=
					std::make_shared<DynamicObjectArray>();
			for (index_t i=0; i<non_value_children.get_num_elements(); ++i)
			{
				/* recursively get all combinations in a new array */
				auto non_value_child=
						non_value_children.get_element<ModelSelectionParameters>(i);

//				io::print("{}\tcurrent non-value child\n", prefix);
//				non_value_child->print_tree(num_prefix+1);

				auto current_combination=
						non_value_child->get_combinations(num_prefix+2);
				non_value_combinations->append_element(current_combination);


//				io::print("{}\tcombinations of non-value nodes:\n", prefix);
//				for (index_t j=0; j<current_combination->get_num_elements(); ++j)
//				{
//					CParameterCombination* c=(CParameterCombination*)
//							current_combination->get_element(j);
//					c->print_tree(num_prefix+2);
//					SG_UNREF(c);
//				}
			}
//			io::print("{}done extracting combinations of non-value nodes\n",
//					prefix);

			/* Now, combine combinations of value and non-value nodes */

			/* if there are only non-value children, nothing is combined */
			if (!value_combinations->get_num_elements())
			{
				/* non-value children have to be multipied first, then, all
				 * these products are just appended */

				/* temporary new root is needed to put fron all product trees */
				if (m_sgobject)
				{
					Parameter* p=new Parameter();
					//FIXME
					//p->add(&(ms_sgobject.get()), m_node_name);
					new_root=std::make_shared<ParameterCombination>(p);
				}
				else
					new_root=std::make_shared<ParameterCombination>();

				auto non_value_products=
						ParameterCombination::non_value_tree_multiplication(
								non_value_combinations, new_root);




				non_value_combinations=non_value_products;

				/* append all non-value combinations to result */
				for (index_t i=0; i<non_value_combinations->get_num_elements(); ++i)
				{
					auto current=
							non_value_combinations->get_element<ParameterCombination>(i);
					result->append_element(current);

				}
			}
			else
			{
				/* before combinations are built, produce products of non-value
				 * combinations. new root is temporarily needed to put front
				 * all new trees */
				if (m_sgobject)
				{
					Parameter* p=new Parameter();
					//FIXME
					//p->add(&(ms_sgobject.get()), m_node_name);
					new_root=std::make_shared<ParameterCombination>(p);
				}
				else
					new_root=std::make_shared<ParameterCombination>();

				auto non_value_products=
						ParameterCombination::non_value_tree_multiplication(
								non_value_combinations, new_root);




				non_value_combinations=non_value_products;

				for (index_t i=0; i<value_combinations->get_num_elements(); ++i)
				{
					auto current_value_tree=
							value_combinations->get_element<ParameterCombination>(i);

					for (index_t j=0; j
							<non_value_combinations->get_num_elements(); ++j)
				{
						auto current_non_value_tree=
								non_value_combinations->get_element<ParameterCombination>(j);

						/* copy current value tree and add all childs of non-
						 * value combination. Then add new node to result */
						auto value_copy=
								current_value_tree->copy_tree();

						value_copy->merge_with(current_non_value_tree);
						result->append_element(value_copy);


					}


				}
			}

			/* clean up*/

		}


	}
	else
	{
		/* if there are no children of a sgobject or root node, result is
		 * only one element (sgobject node) or empty (root node)
		 */
		if (m_sgobject)
		{
			Parameter* p=new Parameter();
			//FIXME
			//p->add(&(ms_sgobject.get()), m_node_name);
			result->append_element(std::make_shared<ParameterCombination>(p));
		}
	}

//	io::print("{}result is a set of {} elements:\n", prefix,
//			result->get_num_elements());
//	for (index_t i=0; i<result->get_num_elements(); ++i)
//	{
//		CParameterCombination* current=(CParameterCombination*)
//				result->get_element(i);
//		current->print_tree(num_prefix+1);
//		SG_UNREF(current);
//	}

	SG_DEBUG("{}------>leaving ModelSelectionParameters::get_combinations()"
			"for \"{}\"", prefix, m_node_name ? m_node_name : "root");
	SG_FREE(prefix);
	return result;
}

void ModelSelectionParameters::print_tree(int prefix_num)
{
	/* prefix is enlarged */
	char* prefix=SG_MALLOC(char, prefix_num+1);
	for (index_t i=0; i<prefix_num; ++i)
		prefix[i]='\t';

	prefix[prefix_num]='\0';

	if (has_children())
	{
		if (m_sgobject)
			io::print("{}{}:\"{}\"\n", prefix, m_node_name, m_sgobject->get_name());
		else
			io::print("{}{} with\n", prefix, m_node_name ? m_node_name : "root");

		/* now recursively print successors */

		/* cast safe because only ModelSelectionParameters are added to list */
		for (index_t i=0; i<m_child_nodes->get_num_elements(); ++i)
		{
			auto child=
				m_child_nodes->get_element<ModelSelectionParameters>(i);
			child->print_tree(prefix_num+1);

		}
	}
	else
	{
		/* has to be a node with name and a numeric range or a single sg_object
		 * without children*/
		if (m_sgobject)
		{
			io::print("{}{}:\"{}\"\n", prefix, m_node_name, m_sgobject->get_name());
		}
		else
		{
			if (m_values)
			{
				// value node
				io::print("{}{} with values: ", prefix, m_node_name);

				switch (m_value_type)
				{
				case MSPT_FLOAT64: case MSPT_FLOAT64_VECTOR:
					case MSPT_FLOAT64_SGVECTOR:

					SGVector<float64_t>::display_vector((float64_t*)m_values,
							m_values_length);
					break;
				case MSPT_INT32: case MSPT_INT32_VECTOR:
					case MSPT_INT32_SGVECTOR:

					SGVector<int32_t>::display_vector((int32_t*)m_values,
							m_values_length);;
					break;
				case MSPT_NONE:
					error("Value node has no type!");
					break;
				default:
					error("Unknown type for model selection parameter!");
					break;
				}
			}
			else
				io::print("root\n");
		}
	}

	SG_FREE(prefix);
}

void ModelSelectionParameters::delete_values()
{
	if (m_values)
	{
		switch (m_value_type)
		{
		case MSPT_FLOAT64: case MSPT_FLOAT64_VECTOR:
			case MSPT_FLOAT64_SGVECTOR:

			SG_FREE((float64_t*)m_values);
			break;
		case MSPT_INT32: case MSPT_INT32_VECTOR:
			case MSPT_INT32_SGVECTOR:

			SG_FREE((int32_t*)m_values);
			break;
		case MSPT_NONE:
			error("Value node has no type!");
			break;
		default:
			error("Unknown type for model selection parameter!");
			break;
		}
	}
}
