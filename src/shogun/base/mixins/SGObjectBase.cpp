/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein,
 *          Giovanni De Toni, Jacob Walker, Fernando Iglesias, Roman Votyakov,
 *          Soumyajit De, Evgeniy Andreev, Evangelos Anagnostopoulos,
 *          Leon Kuchenbecker, Sanuj Sharma, Wu Lin
 */

#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Version.h>
#include <shogun/base/class_list.h>
#include <shogun/base/mixins/SGObjectBase.h>
#include <shogun/base/range.h>
#include <shogun/io/SerializableFile.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/RefCount.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/observers/ParameterObserver.h>

#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>

using namespace shogun;

template <typename Derived>
CSGObjectBase<Derived>::CSGObjectBase(const CSGObjectBase<Derived>& orig)
    : CSGObjectBase()
{
}

template <typename Derived>
CSGObjectBase<Derived>::CSGObjectBase()
    : house_keeper((HouseKeeper<Derived>&)(*(Derived*)this)), // FIXME
      param_handler((ParameterHandler<Derived>&)(*(Derived*)this)),
      io(house_keeper.io)
{
	init();
	SG_SGCDEBUG("SGObject created (%p)\n", this)
}

template <typename Derived>
CSGObjectBase<Derived>::~CSGObjectBase()
{
	delete m_parameters;
	delete m_model_selection_parameters;
	delete m_gradient_parameters;
}

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;

template <typename Derived>
void CSGObjectBase<Derived>::list_memory_allocs()
{
	shogun::list_memory_allocs();
}
#endif

template <typename Derived>
void CSGObjectBase<Derived>::update_parameter_hash()
{
	SG_DEBUG("entering\n")

	uint32_t carry = 0;
	uint32_t length = 0;

	m_hash = 0;
	get_parameter_incremental_hash(m_hash, carry, length);
	m_hash = CHash::FinalizeIncrementalMurmurHash3(m_hash, carry, length);

	SG_DEBUG("leaving\n")
}

template <typename Derived>
bool CSGObjectBase<Derived>::parameter_hash_changed()
{
	SG_DEBUG("entering\n")

	uint32_t hash = 0;
	uint32_t carry = 0;
	uint32_t length = 0;

	get_parameter_incremental_hash(hash, carry, length);
	hash = CHash::FinalizeIncrementalMurmurHash3(hash, carry, length);

	SG_DEBUG("leaving\n")
	return (m_hash != hash);
}

template <typename Derived>
void CSGObjectBase<Derived>::print_serializable(const char* prefix)
{
	SG_PRINT(
	    "\n%s\n================================================================"
	    "================\n",
	    house_keeper.get_name())
	m_parameters->print(prefix);
}

template <typename Derived>
bool CSGObjectBase<Derived>::save_serializable(
    CSerializableFile* file, const char* prefix)
{
	SG_DEBUG("START SAVING CSGObject '%s'\n", house_keeper.get_name())
	try
	{
		save_serializable_pre();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING(
		    "%s%s::save_serializable_pre(): ShogunException: "
		    "%s\n",
		    prefix, house_keeper.get_name(), e.what());
		return false;
	}

	if (!m_save_pre_called)
	{
		SG_SWARNING(
		    "%s%s::save_serializable_pre(): Implementation "
		    "error: BASE_CLASS::SAVE_SERIALIZABLE_PRE() not "
		    "called!\n",
		    prefix, house_keeper.get_name());
		return false;
	}

	if (!m_parameters->save(file, prefix))
		return false;

	try
	{
		save_serializable_post();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING(
		    "%s%s::save_serializable_post(): ShogunException: "
		    "%s\n",
		    prefix, house_keeper.get_name(), e.what());
		return false;
	}

	if (!m_save_post_called)
	{
		SG_SWARNING(
		    "%s%s::save_serializable_post(): Implementation "
		    "error: BASE_CLASS::SAVE_SERIALIZABLE_POST() not "
		    "called!\n",
		    prefix, house_keeper.get_name());
		return false;
	}

	if (prefix == NULL || *prefix == '\0')
		file->close();

	SG_DEBUG("DONE SAVING CSGObject '%s' (%p)\n", house_keeper.get_name(), this)

	return true;
}

template <typename Derived>
bool CSGObjectBase<Derived>::load_serializable(
    CSerializableFile* file, const char* prefix)
{
	REQUIRE(file != NULL, "Serializable file object should be != NULL\n");

	SG_DEBUG("START LOADING CSGObject '%s'\n", house_keeper.get_name())
	try
	{
		load_serializable_pre();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING(
		    "%s%s::load_serializable_pre(): ShogunException: "
		    "%s\n",
		    prefix, house_keeper.get_name(), e.what());
		return false;
	}
	if (!m_load_pre_called)
	{
		SG_SWARNING(
		    "%s%s::load_serializable_pre(): Implementation "
		    "error: BASE_CLASS::LOAD_SERIALIZABLE_PRE() not "
		    "called!\n",
		    prefix, house_keeper.get_name());
		return false;
	}

	if (!m_parameters->load(file, prefix))
		return false;

	try
	{
		load_serializable_post();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING(
		    "%s%s::load_serializable_post(): ShogunException: "
		    "%s\n",
		    prefix, house_keeper.get_name(), e.what());
		return false;
	}

	if (!m_load_post_called)
	{
		SG_SWARNING(
		    "%s%s::load_serializable_post(): Implementation "
		    "error: BASE_CLASS::LOAD_SERIALIZABLE_POST() not "
		    "called!\n",
		    prefix, house_keeper.get_name());
		return false;
	}
	SG_DEBUG(
	    "DONE LOADING CSGObject '%s' (%p)\n", house_keeper.get_name(), this)

	return true;
}

template <typename Derived>
void CSGObjectBase<Derived>::load_serializable_pre() throw(ShogunException)
{
	m_load_pre_called = true;
}

template <typename Derived>
void CSGObjectBase<Derived>::load_serializable_post() throw(ShogunException)
{
	m_load_post_called = true;
}

template <typename Derived>
void CSGObjectBase<Derived>::save_serializable_pre() throw(ShogunException)
{
	m_save_pre_called = true;
}

template <typename Derived>
void CSGObjectBase<Derived>::save_serializable_post() throw(ShogunException)
{
	m_save_post_called = true;
}

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;
#endif

template <typename Derived>
void CSGObjectBase<Derived>::init()
{
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
	{
		int32_t idx = sg_mallocs->index_of(this);
		if (idx > -1)
		{
			MemoryBlock* b = sg_mallocs->get_element_ptr(idx);
			b->set_sgobject();
		}
	}
#endif

	m_parameters = new Parameter();
	m_model_selection_parameters = new Parameter();
	m_gradient_parameters = new Parameter();
	m_load_pre_called = false;
	m_load_post_called = false;
	m_save_pre_called = false;
	m_save_post_called = false;
	m_hash = 0;
}

template <typename Derived>
void CSGObjectBase<Derived>::print_modsel_params()
{
	SG_PRINT(
	    "parameters available for model selection for %s:\n",
	    house_keeper.get_name())

	index_t num_param = m_model_selection_parameters->get_num_parameters();

	if (!num_param)
		SG_PRINT("\tnone\n")

	for (index_t i = 0; i < num_param; i++)
	{
		TParameter* current = m_model_selection_parameters->get_parameter(i);
		index_t l = 200;
		char* type = SG_MALLOC(char, l);
		if (type)
		{
			current->m_datatype.to_string(type, l);
			SG_PRINT(
			    "\t%s (%s): %s\n", current->m_name, current->m_description,
			    type);
			SG_FREE(type);
		}
	}
}

template <typename Derived>
SGStringList<char> CSGObjectBase<Derived>::get_modelsel_names()
{
	index_t num_param = m_model_selection_parameters->get_num_parameters();

	SGStringList<char> result(num_param, -1);

	index_t max_string_length = -1;

	for (index_t i = 0; i < num_param; i++)
	{
		char* name = m_model_selection_parameters->get_parameter(i)->m_name;
		index_t len = strlen(name);
		// +1 to have a zero terminated string
		result.strings[i] = SGString<char>(name, len + 1);

		if (len > max_string_length)
			max_string_length = len;
	}

	result.max_string_length = max_string_length;

	return result;
}

template <typename Derived>
char* CSGObjectBase<Derived>::get_modsel_param_descr(const char* param_name)
{
	index_t index = get_modsel_param_index(param_name);

	if (index < 0)
	{
		SG_ERROR(
		    "There is no model selection parameter called \"%s\" for %s",
		    param_name, house_keeper.get_name());
	}

	return m_model_selection_parameters->get_parameter(index)->m_description;
}

template <typename Derived>
index_t CSGObjectBase<Derived>::get_modsel_param_index(const char* param_name)
{
	/* use fact that names extracted from below method are in same order than
	 * in m_model_selection_parameters variable */
	SGStringList<char> names = get_modelsel_names();

	/* search for parameter with provided name */
	index_t index = -1;
	for (index_t i = 0; i < names.num_strings; i++)
	{
		TParameter* current = m_model_selection_parameters->get_parameter(i);
		if (!strcmp(param_name, current->m_name))
		{
			index = i;
			break;
		}
	}

	return index;
}

template <typename Derived>
void CSGObjectBase<Derived>::get_parameter_incremental_hash(
    uint32_t& hash, uint32_t& carry, uint32_t& total_length)
{
	for (index_t i = 0; i < m_parameters->get_num_parameters(); i++)
	{
		TParameter* p = m_parameters->get_parameter(i);

		SG_DEBUG(
		    "Updating hash for parameter %s.%s\n", house_keeper.get_name(),
		    p->m_name);

		if (p->m_datatype.m_ptype == PT_SGOBJECT)
		{
			if (p->m_datatype.m_ctype == CT_SCALAR)
			{
				Derived* child = *((Derived**)(p->m_parameter));

				if (child)
				{
					child->get_parameter_incremental_hash(
					    hash, carry, total_length);
				}
			}
			else if (
			    p->m_datatype.m_ctype == CT_VECTOR ||
			    p->m_datatype.m_ctype == CT_SGVECTOR)
			{
				Derived** child = (*(Derived***)(p->m_parameter));

				for (index_t j = 0; j < *(p->m_datatype.m_length_y); j++)
				{
					if (child[j])
					{
						child[j]->get_parameter_incremental_hash(
						    hash, carry, total_length);
					}
				}
			}
		}
		else
			p->get_incremental_hash(hash, carry, total_length);
	}
}

template <typename Derived>
void CSGObjectBase<Derived>::build_gradient_parameter_dictionary(
    CMap<TParameter*, Derived*>* dict)
{
	for (index_t i = 0; i < m_gradient_parameters->get_num_parameters(); i++)
	{
		TParameter* p = m_gradient_parameters->get_parameter(i);
		dict->add(p, (Derived*)this);
	}

	for (index_t i = 0; i < m_model_selection_parameters->get_num_parameters();
	     i++)
	{
		TParameter* p = m_model_selection_parameters->get_parameter(i);
		Derived* child = *(Derived**)(p->m_parameter);

		if ((p->m_datatype.m_ptype == PT_SGOBJECT) &&
		    (p->m_datatype.m_ctype == CT_SCALAR) && child)
		{
			child->build_gradient_parameter_dictionary(dict);
		}
	}
}

namespace shogun
{
	template class CSGObjectBase<CSGObject>;
	template class CSGObjectBase<ObservedValue>;
} // namespace shogun