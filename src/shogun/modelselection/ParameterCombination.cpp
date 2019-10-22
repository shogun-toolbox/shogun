/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Soeren Sonnenburg, Heiko Strathmann, Wu Lin,
 *          Roman Votyakov, Bjoern Esser, Esben Sorig, Sanuj Sharma
 */

#ifdef ENABLE_MODEL_SELECTION
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/base/Parameter.h>
#include <shogun/machine/Machine.h>
#include <set>
#include <string>

using namespace shogun;
using namespace std;

ParameterCombination::ParameterCombination()
{
	init();
}

ParameterCombination::ParameterCombination(Parameter* param)
{
	init();

	m_param=param;
}

ParameterCombination::ParameterCombination(std::shared_ptr<SGObject> obj)
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
			if (type.m_ctype==CT_SGVECTOR || type.m_ctype==CT_VECTOR)
			{
				Parameter* p=new Parameter();
				p->add_vector((float64_t**)param->m_parameter, type.m_length_y,
						param->m_name);

				m_child_nodes.push_back(std::make_shared<ParameterCombination>(p));
				m_parameters_length+=*(type.m_length_y);
			}
			else if (type.m_ctype==CT_SGMATRIX || type.m_ctype==CT_MATRIX)
			{
				Parameter* p=new Parameter();
				p->add_matrix((float64_t**)param->m_parameter, type.m_length_y,
					type.m_length_x, param->m_name);

				m_child_nodes.push_back(std::make_shared<ParameterCombination>(p));
				m_parameters_length+=type.get_num_elements();
			}
			else if (type.m_ctype==CT_SCALAR)
			{
				Parameter* p=new Parameter();
				p->add((float64_t*)param->m_parameter, param->m_name);

				m_child_nodes.push_back(std::make_shared<ParameterCombination>(p));
				m_parameters_length++;
			}
			else
			{
				io::warn("Parameter {}.{} was not added to parameter combination, "
					"since it isn't a type currently supported", obj->get_name(),
					param->m_name);
			}
		}
		else
		{
			io::warn("Parameter {}.{} was not added to parameter combination, "
					"since it isn't of floating point type", obj->get_name(),
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
				auto child=*((SGObject**)(param->m_parameter));

				if (child->m_gradient_parameters->get_num_parameters()>0)
				{
					//FIXME
					//auto comb=std::make_shared<CParameterCombination>(child);
					std::shared_ptr<ParameterCombination> comb;
					comb->m_param=new Parameter();
					comb->m_param->add((SGObject**)(param->m_parameter),
							param->m_name);

					m_child_nodes.push_back(comb);
					m_parameters_length+=comb->m_parameters_length;
				}
			}
			else
			{
				not_implemented(SOURCE_LOCATION);
			}
		}
	}
}

void ParameterCombination::init()
{
	m_parameters_length=0;
	m_param=NULL;
	m_child_nodes.clear();


	/*SG_ADD((SGObject**)&m_child_nodes, "child_nodes", "Children of this node")*/;
}

ParameterCombination::~ParameterCombination()
{
	delete m_param;

}

void ParameterCombination::append_child(std::shared_ptr<ParameterCombination> child)
{
	m_child_nodes.push_back(child);
}

bool ParameterCombination::set_parameter_helper(
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
					error("Parameter {} not a boolean parameter", name);

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

bool ParameterCombination::set_parameter_helper(
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
					error("Parameter {} not a integer parameter", name);

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

bool ParameterCombination::set_parameter_helper(
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
					error("Parameter {} not a double parameter", name);

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


TParameter* ParameterCombination::get_parameter_helper(const char* name)
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


TParameter* ParameterCombination::get_parameter(const char* name,
		SGObject* parent)
{
	bool match = false;

	if (m_param)
	{
		for (index_t i = 0; i < m_param->get_num_parameters(); i++)
		{
			if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_SGOBJECT)
			{
				SGObject* obj =
						(*((SGObject**)m_param->get_parameter(i)->m_parameter));
				if (parent == obj)
					match = true;
			}
		}

	}

	for (index_t i = 0; i < m_child_nodes.size(); ++i)
	{
		auto child =
				m_child_nodes[i];

		TParameter* p;

		if (!match)
			 p = child->get_parameter(name, parent);

		else
			 p = child->get_parameter_helper(name);

		if (p)
		{

			return p;
		}


	}

	return NULL;
}


void ParameterCombination::merge_with(std::shared_ptr<ParameterCombination> node)
{
	for (index_t i=0; i<node->m_child_nodes.size(); ++i)
	{
		auto child=
				node->m_child_nodes[i];
		append_child(child->copy_tree());

	}
}

void ParameterCombination::print_tree(int prefix_num) const
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
		io::print("{}", prefix);
		for (index_t i=0; i<m_param->get_num_parameters(); ++i)
		{
			EContainerType ctype = m_param->get_parameter(i)->m_datatype.m_ctype;

			/* distinction between sgobject and values */
			if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_SGOBJECT)
			{
				TParameter* param=m_param->get_parameter(i);
				SGObject* current_sgobject=*((SGObject**) param->m_parameter);
				io::print("\"{}\":{} at {} ", param->m_name,
						current_sgobject->get_name(), fmt::ptr(current_sgobject));
			}
			else if (ctype==CT_SGVECTOR || ctype==CT_VECTOR || ctype==CT_SGMATRIX || ctype==CT_MATRIX)
			{
				io::print("\"{}\"=", m_param->get_parameter(i)->m_name);
				float64_t** param = (float64_t**)(m_param->
						get_parameter(i)->m_parameter);

				index_t length = m_param->get_parameter(i)->m_datatype.get_num_elements();

				for (index_t j = 0; j < length; j++)
					io::print("{} ", (*param)[j]);
			}

			else
			{
				io::print("\"{}\"=", m_param->get_parameter(i)->m_name);
				void* param=m_param->get_parameter(i)->m_parameter;

				if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_FLOAT64)
					io::print("{} ", *((float64_t*)param));
				else if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_INT32)
					io::print("{} ", *((int32_t*)param));
				else if (m_param->get_parameter(i)->m_datatype.m_ptype==PT_BOOL)
					io::print("{} ", *((bool*)param ? "true" : "false"));
				else
					not_implemented(SOURCE_LOCATION);
			}

		}

	}
	else
		io::print("{}root", prefix);

	io::print("\n");

	for (index_t i=0; i<m_child_nodes.size(); ++i)
	{
		auto child=
				m_child_nodes[i];
		child->print_tree(prefix_num+1);

	}

	SG_FREE(prefix);
}

std::vector<Parameter*> ParameterCombination::parameter_set_multiplication(
	const std::vector<Parameter*>& set_1,
	const std::vector<Parameter*>& set_2)
{
	SG_DEBUG("entering CParameterCombination::parameter_set_multiplication()")

	SG_DEBUG("set 1:")
	for (index_t i=0; i<set_1.size(); ++i)
	{
		for (index_t j=0; j<set_1[i]->get_num_parameters(); ++j)
			SG_DEBUG("\t{}", set_1[i]->get_parameter(j)->m_name)
	}

	SG_DEBUG("set 2:")
	for (index_t i=0; i<set_2.size(); ++i)
	{
		for (index_t j=0; j<set_2[i]->get_num_parameters(); ++j)
			SG_DEBUG("\t{}", set_2[i]->get_parameter(j)->m_name)
	}

	std::vector<Parameter*> result;

	for (index_t i=0; i<set_1.size(); ++i)
	{
		for (index_t j=0; j<set_2.size(); ++j)
		{
			Parameter* p=new Parameter();
			p->add_parameters(set_1[i]);
			p->add_parameters(set_2[j]);
			result.push_back(p);
		}
	}

	SG_DEBUG("leaving CParameterCombination::parameter_set_multiplication()")
	return result;
}


std::vector<std::shared_ptr<ParameterCombination>>
ParameterCombination::leaf_sets_multiplication(
		const std::vector<std::vector<std::shared_ptr<ParameterCombination>>>& sets,
		std::shared_ptr<const ParameterCombination> new_root)
{
	std::vector<std::shared_ptr<ParameterCombination>> result;

	/* check marginal cases */
	if (sets.size()==1)
	{
		auto& current_set = sets[0];

		/* just use the only element into result array.
		 * put root node before all combinations*/
		result=current_set;



		for (index_t i=0; i<result.size(); ++i)
		{
			/* put new root as root into the tree and replace tree */
			auto current=
					result[i];
			auto root=new_root->copy_tree();
			root->append_child(current);
			result[i] = root;

		}
	}
	else if (sets.size()>1)
	{
		/* now the case where at least two sets are given */

		/* first, extract Parameter instances of given sets */
		std::vector<std::vector<Parameter*>> param_sets;

		for (index_t set_nr=0; set_nr<sets.size(); ++set_nr)
		{
			auto current_set = sets[set_nr];
			std::vector<Parameter*> new_param_set;

			for (index_t i=0; i<current_set.size(); ++i)
			{
				auto current_node=
						current_set[i];

				if (current_node->m_child_nodes.size())
				{
					error("leaf sets multiplication only possible if all "
							"trees are leafs");
				}

				Parameter* current_param=current_node->m_param;

				if (current_param)
					new_param_set.push_back(current_param);
				else
				{
					error("leaf sets multiplication only possible if all "
							"leafs have non-NULL Parameter instances");
				}


			}
			param_sets.push_back(new_param_set);

		}

		/* second, build products of all parameter sets */
		std::vector<Parameter*> param_product=parameter_set_multiplication(
				param_sets[0], param_sets[1]);

		std::for_each(
			param_sets[0].begin(), param_sets[0].end(), [](auto* p) { delete p; });
		std::for_each(
			param_sets[1].begin(), param_sets[1].end(), [](auto* p) { delete p; });
		
		/* build product of all remaining sets and collect results. delete all
		 * parameter instances of interim products*/
		for (index_t i=2; i<param_sets.size(); ++i)
		{
			auto old_temp_result=param_product;
			param_product=parameter_set_multiplication(param_product,
					param_sets[i]);

			/* delete interim result parameter instances */
			for (index_t j=0; j<old_temp_result.size(); ++j)
				delete old_temp_result[j];
		}

		/* at this point there is only one DynArray instance remaining:
		 * param_product. contains all combinations of parameters of all given
		 * sets */

		/* third, build tree sets with the given root and the parameter product
		 * elements */
		for (index_t i=0; i<param_product.size(); ++i)
		{
			/* build parameter node from parameter product to append to root */
			auto param_node=std::make_shared<ParameterCombination>(
					param_product[i]);

			/* copy new root node, has to be a new one each time */
			auto root=new_root->copy_tree();

			/* append both and add them to result set */
			root->append_child(param_node);
			result.push_back(root);
		}
	}

	return result;
}

std::vector<std::shared_ptr<ParameterCombination>>
ParameterCombination::non_value_tree_multiplication(
	const std::vector<std::vector<std::shared_ptr<ParameterCombination>>>& sets,
	std::shared_ptr<const ParameterCombination> new_root)
{
	SG_DEBUG("entering ParameterCombination::non_value_tree_multiplication()")
	std::vector<std::shared_ptr<ParameterCombination>> result;

	/* first step: get all names in the sets */
	std::set<string> names;

	for (index_t j=0;
			j<sets.size(); ++j)
	{
		auto& current_set = sets[j];

		for (index_t k=0; k	<current_set.size(); ++k)
		{
			auto& current_tree = current_set[k];

			names.insert(string(current_tree->m_param->get_parameter(0)->m_name));


		}


	}

	SG_DEBUG("all names")
	for (std::set<string>::iterator it=names.begin(); it!=names.end(); ++it)
		SG_DEBUG("\"{}\"", (*it).c_str())

	/* only do stuff if there are names */
	if (!names.empty())
	{
		/* next step, build temporary structure where all elements with first
		 * name are put. Elements of that structure will be extend iteratively
		 * per name */


		/* extract all trees with first name */
		const char* first_name=(*(names.begin())).c_str();
		auto trees=
				ParameterCombination::extract_trees_with_name(sets, first_name);

		SG_DEBUG("adding trees for first name \"{}\":", first_name)
		for (index_t i=0; i<trees.size(); ++i)
		{
			auto current_tree=
					trees[i];

			auto current_root=new_root->copy_tree();
			current_root->append_child(current_tree);
			result.push_back(current_root);

			// current_tree->print_tree(1);

		}


		/* now iterate over the remaining names and build products */
		SG_DEBUG("building products with remaining trees:")
		std::set<string>::iterator it=names.begin();
		for (++it; it!=names.end(); ++it)
		{
			SG_DEBUG("processing \"{}\"", (*it).c_str())

			/* extract all trees with current name */
			const char* current_name=(*it).c_str();
			trees=ParameterCombination::extract_trees_with_name(sets,
					current_name);

			/* create new set of trees where each element is put once for each
			 * of the just generated trees */
			std::vector<std::shared_ptr<ParameterCombination>> new_result;
			for (index_t i=0; i<result.size(); ++i)
			{
				for (index_t j=0; j<trees.size(); ++j)
				{
					auto to_copy=
							result[i];

					/* create a copy of current element */
					auto new_element=to_copy->copy_tree();


					auto& to_add = trees[j];
					new_element->append_child(to_add);

					new_result.push_back(new_element);
					// SG_DEBUG("added:")
					// new_element->print_tree();
				}
			}

			/* clean up */


			/* replace result by new_result */

			result=new_result;
		}
	}

	SG_DEBUG("leaving CParameterCombination::non_value_tree_multiplication()")
	return result;
}

std::vector<std::shared_ptr<ParameterCombination>>
ParameterCombination::extract_trees_with_name(
	const std::vector<std::vector<std::shared_ptr<ParameterCombination>>>& sets,
	const char* desired_name)
{
	std::vector<std::shared_ptr<ParameterCombination>> result;

	for (index_t j=0; j<sets.size(); ++j)
	{
		auto& current_set = sets[j];

		for (index_t k=0; k<current_set.size(); ++k)
		{
			auto& current_tree = current_set[k];

			char* current_name=current_tree->m_param->get_parameter(0)->m_name;

			if (!strcmp(current_name, desired_name))
				result.push_back(current_tree);


		}


	}

	return result;
}

std::shared_ptr<ParameterCombination> ParameterCombination::copy_tree() const
{
	auto copy=std::make_shared<ParameterCombination>();

	/* but build new Parameter instance */

	/* only call add_parameters() argument is non-null */
	if (m_param)
	{
		copy->m_param=new Parameter();
		copy->m_param->add_parameters(m_param);
	} else
		copy->m_param=NULL;

	/* recursively copy all children */
	for (index_t i=0; i<m_child_nodes.size(); ++i)
	{
		auto child = m_child_nodes[i];
		copy->m_child_nodes.push_back(child->copy_tree());

	}

	return copy;
}

void ParameterCombination::apply_to_machine(std::shared_ptr<Machine> machine) const
{
	apply_to_modsel_parameter(machine->m_model_selection_parameters);
}

void ParameterCombination::apply_to_modsel_parameter(
		Parameter* parameter) const
{
	/* case root node */
	if (!m_param)
	{
		/* iterate over all children and recursively set parameters from
		 * their values to the current parameter input (its just handed one
		 * recursion level downwards) */
		for (index_t i=0; i<m_child_nodes.size(); ++i)
		{
			auto child = m_child_nodes[i];
			child->apply_to_modsel_parameter(parameter);

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
			/* if a parameter node has children, it has to have ONE SGObject as
			 * parameter */
			if (m_param->get_num_parameters()>1 ||
					m_param->get_parameter(0)->m_datatype.m_ptype!=PT_SGOBJECT)
			{
				error("invalid CParameterCombination node type, has children"
						" and more than one parameter or is not a "
						"SGObject.");
			}

			/* cast is now safe */
			SGObject* current_sgobject=
					*((SGObject**)(m_param->get_parameter(0)->m_parameter));

			/* iterate over all children and recursively set parameters from
			 * their values */
			for (index_t i=0; i<m_child_nodes.size(); ++i)
			{
				auto child=
						m_child_nodes[i];
				child->apply_to_modsel_parameter(
						current_sgobject->m_model_selection_parameters);

			}
		}
	}
	else
		error("CParameterCombination node has illegal type.");
}

void ParameterCombination::build_parameter_values_map(
		std::shared_ptr<CMap<TParameter*, SGVector<float64_t> >> dict)
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
				if (type.m_ctype==CT_SGVECTOR || type.m_ctype==CT_VECTOR ||
					type.m_ctype==CT_SGMATRIX || type.m_ctype==CT_MATRIX)
				{
					SGVector<float64_t> value(*((float64_t **)param->m_parameter),
							type.get_num_elements(), false);
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

	for (index_t i=0; i<m_child_nodes.size(); i++)
	{
		auto child = m_child_nodes[i];
		child->build_parameter_values_map(dict);

	}
}

void ParameterCombination::build_parameter_parent_map(
		std::shared_ptr<CMap<TParameter*, SGObject*>> dict)
{
	SGObject* parent=NULL;

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
					parent=(*(SGObject**)param->m_parameter);
					break;
				}
				else
				{
					not_implemented(SOURCE_LOCATION);
				}
			}
		}
	}

	for (index_t i=0; i<m_child_nodes.size(); i++)
	{
		auto child=
			m_child_nodes[i];

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
					not_implemented(SOURCE_LOCATION);
				}
			}
			else
			{
				dict->add(param, parent);
			}
		}

	}
}
#endif
