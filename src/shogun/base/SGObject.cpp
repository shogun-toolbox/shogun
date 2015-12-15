/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Written (W) 2011-2013 Heiko Strathmann
 * Written (W) 2013-2014 Thoralf Klein
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/RefCount.h>

#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Version.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/DynArray.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/io/SerializableFile.h>

#include <shogun/base/class_list.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace shogun
{
	class Parallel;

	extern Parallel* sg_parallel;
	extern SGIO* sg_io;
	extern Version* sg_version;

	template<> void CSGObject::set_generic<bool>()
	{
		m_generic = PT_BOOL;
	}

	template<> void CSGObject::set_generic<char>()
	{
		m_generic = PT_CHAR;
	}

	template<> void CSGObject::set_generic<int8_t>()
	{
		m_generic = PT_INT8;
	}

	template<> void CSGObject::set_generic<uint8_t>()
	{
		m_generic = PT_UINT8;
	}

	template<> void CSGObject::set_generic<int16_t>()
	{
		m_generic = PT_INT16;
	}

	template<> void CSGObject::set_generic<uint16_t>()
	{
		m_generic = PT_UINT16;
	}

	template<> void CSGObject::set_generic<int32_t>()
	{
		m_generic = PT_INT32;
	}

	template<> void CSGObject::set_generic<uint32_t>()
	{
		m_generic = PT_UINT32;
	}

	template<> void CSGObject::set_generic<int64_t>()
	{
		m_generic = PT_INT64;
	}

	template<> void CSGObject::set_generic<uint64_t>()
	{
		m_generic = PT_UINT64;
	}

	template<> void CSGObject::set_generic<float32_t>()
	{
		m_generic = PT_FLOAT32;
	}

	template<> void CSGObject::set_generic<float64_t>()
	{
		m_generic = PT_FLOAT64;
	}

	template<> void CSGObject::set_generic<floatmax_t>()
	{
		m_generic = PT_FLOATMAX;
	}

	template<> void CSGObject::set_generic<CSGObject*>()
	{
		m_generic = PT_SGOBJECT;
	}

	template<> void CSGObject::set_generic<complex128_t>()
	{
		m_generic = PT_COMPLEX128;
	}

} /* namespace shogun  */

using namespace shogun;

CSGObject::CSGObject()
{
	init();
	set_global_objects();
	m_refcount = new RefCount(0);

	SG_SGCDEBUG("SGObject created (%p)\n", this)
}

CSGObject::CSGObject(const CSGObject& orig)
:io(orig.io), parallel(orig.parallel), version(orig.version)
{
	init();
	set_global_objects();
	m_refcount = new RefCount(0);

	SG_SGCDEBUG("SGObject copied (%p)\n", this)
}

CSGObject::~CSGObject()
{
	SG_SGCDEBUG("SGObject destroyed (%p)\n", this)

	unset_global_objects();
	delete m_parameters;
	delete m_model_selection_parameters;
	delete m_gradient_parameters;
	delete m_refcount;
}

#ifdef USE_REFERENCE_COUNTING
int32_t CSGObject::ref()
{
	int32_t count = m_refcount->ref();
	SG_SGCDEBUG("ref() refcount %ld obj %s (%p) increased\n", count, this->get_name(), this)
	return m_refcount->ref_count();
}

int32_t CSGObject::ref_count()
{
	int32_t count = m_refcount->ref_count();
	SG_SGCDEBUG("ref_count(): refcount %d, obj %s (%p)\n", count, this->get_name(), this)
	return m_refcount->ref_count();
}

int32_t CSGObject::unref()
{
	int32_t count = m_refcount->unref();
	if (count<=0)
	{
		SG_SGCDEBUG("unref() refcount %ld, obj %s (%p) destroying\n", count, this->get_name(), this)
		delete this;
		return 0;
	}
	else
	{
		SG_SGCDEBUG("unref() refcount %ld obj %s (%p) decreased\n", count, this->get_name(), this)
		return m_refcount->ref_count();
	}
}
#endif //USE_REFERENCE_COUNTING

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;

void CSGObject::list_memory_allocs()
{
	shogun::list_memory_allocs();
}
#endif

CSGObject * CSGObject::shallow_copy() const
{
	SG_NOTIMPLEMENTED
	return NULL;
}

CSGObject * CSGObject::deep_copy() const
{
	SG_NOTIMPLEMENTED
	return NULL;
}

void CSGObject::set_global_objects()
{
	if (!sg_io || !sg_parallel || !sg_version)
	{
		fprintf(stderr, "call init_shogun() before using the library, dying.\n");
		exit(1);
	}

	SG_REF(sg_io);
	SG_REF(sg_parallel);
	SG_REF(sg_version);

	io=sg_io;
	parallel=sg_parallel;
	version=sg_version;
}

void CSGObject::unset_global_objects()
{
	SG_UNREF(version);
	SG_UNREF(parallel);
	SG_UNREF(io);
}

void CSGObject::set_global_io(SGIO* new_io)
{
	SG_REF(new_io);
	SG_UNREF(sg_io);
	sg_io=new_io;
}

SGIO* CSGObject::get_global_io()
{
	SG_REF(sg_io);
	return sg_io;
}

void CSGObject::set_global_parallel(Parallel* new_parallel)
{
	SG_REF(new_parallel);
	SG_UNREF(sg_parallel);
	sg_parallel=new_parallel;
}

void CSGObject::update_parameter_hash()
{
	SG_DEBUG("entering\n")

	uint32_t carry=0;
	uint32_t length=0;

	m_hash=0;
	get_parameter_incremental_hash(m_hash, carry, length);
	m_hash=CHash::FinalizeIncrementalMurmurHash3(m_hash, carry, length);

	SG_DEBUG("leaving\n")
}

bool CSGObject::parameter_hash_changed()
{
	SG_DEBUG("entering\n")

	uint32_t hash=0;
	uint32_t carry=0;
	uint32_t length=0;

	get_parameter_incremental_hash(hash, carry, length);
	hash=CHash::FinalizeIncrementalMurmurHash3(hash, carry, length);

	SG_DEBUG("leaving\n")
	return (m_hash!=hash);
}

Parallel* CSGObject::get_global_parallel()
{
	SG_REF(sg_parallel);
	return sg_parallel;
}

void CSGObject::set_global_version(Version* new_version)
{
	SG_REF(new_version);
	SG_UNREF(sg_version);
	sg_version=new_version;
}

Version* CSGObject::get_global_version()
{
	SG_REF(sg_version);
	return sg_version;
}

bool CSGObject::is_generic(EPrimitiveType* generic) const
{
	*generic = m_generic;

	return m_generic != PT_NOT_GENERIC;
}

void CSGObject::unset_generic()
{
	m_generic = PT_NOT_GENERIC;
}

void CSGObject::print_serializable(const char* prefix)
{
	SG_PRINT("\n%s\n================================================================================\n", get_name())
	m_parameters->print(prefix);
}

bool CSGObject::save_serializable(CSerializableFile* file,
		const char* prefix)
{
	SG_DEBUG("START SAVING CSGObject '%s'\n", get_name())
	try
	{
		save_serializable_pre();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::save_serializable_pre(): ShogunException: "
				   "%s\n", prefix, get_name(),
				   e.get_exception_string());
		return false;
	}

	if (!m_save_pre_called)
	{
		SG_SWARNING("%s%s::save_serializable_pre(): Implementation "
				   "error: BASE_CLASS::SAVE_SERIALIZABLE_PRE() not "
				   "called!\n", prefix, get_name());
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
		SG_SWARNING("%s%s::save_serializable_post(): ShogunException: "
				   "%s\n", prefix, get_name(),
				   e.get_exception_string());
		return false;
	}

	if (!m_save_post_called)
	{
		SG_SWARNING("%s%s::save_serializable_post(): Implementation "
				   "error: BASE_CLASS::SAVE_SERIALIZABLE_POST() not "
				   "called!\n", prefix, get_name());
		return false;
	}

	if (prefix == NULL || *prefix == '\0')
		file->close();

	SG_DEBUG("DONE SAVING CSGObject '%s' (%p)\n", get_name(), this)

	return true;
}

bool CSGObject::load_serializable(CSerializableFile* file,
		const char* prefix)
{
	REQUIRE(file != NULL, "Serializable file object should be != NULL\n");

	SG_DEBUG("START LOADING CSGObject '%s'\n", get_name())
	try
	{
		load_serializable_pre();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::load_serializable_pre(): ShogunException: "
				   "%s\n", prefix, get_name(),
				   e.get_exception_string());
		return false;
	}
	if (!m_load_pre_called)
	{
		SG_SWARNING("%s%s::load_serializable_pre(): Implementation "
				   "error: BASE_CLASS::LOAD_SERIALIZABLE_PRE() not "
				   "called!\n", prefix, get_name());
		return false;
	}

	try
	{
		load_serializable_post();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::load_serializable_post(): ShogunException: "
		            "%s\n", prefix, get_name(),
		            e.get_exception_string());
		return false;
	}

	if (!m_load_post_called)
	{
		SG_SWARNING("%s%s::load_serializable_post(): Implementation "
		            "error: BASE_CLASS::LOAD_SERIALIZABLE_POST() not "
		            "called!\n", prefix, get_name());
		return false;
	}
	SG_DEBUG("DONE LOADING CSGObject '%s' (%p)\n", get_name(), this)

	return true;
}

void CSGObject::load_serializable_pre() throw (ShogunException)
{
	m_load_pre_called = true;
}

void CSGObject::load_serializable_post() throw (ShogunException)
{
	m_load_post_called = true;
}

void CSGObject::save_serializable_pre() throw (ShogunException)
{
	m_save_pre_called = true;
}

void CSGObject::save_serializable_post() throw (ShogunException)
{
	m_save_post_called = true;
}

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;
#endif

void CSGObject::init()
{
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
	{
		int32_t idx=sg_mallocs->index_of(this);
		if (idx>-1)
		{
			MemoryBlock* b=sg_mallocs->get_element_ptr(idx);
			b->set_sgobject();
		}
	}
#endif

	io = NULL;
	parallel = NULL;
	version = NULL;
	m_parameters = new Parameter();
	m_model_selection_parameters = new Parameter();
	m_gradient_parameters=new Parameter();
	m_generic = PT_NOT_GENERIC;
	m_load_pre_called = false;
	m_load_post_called = false;
	m_save_pre_called = false;
	m_save_post_called = false;
	m_hash = 0;
}

void CSGObject::print_modsel_params()
{
	SG_PRINT("parameters available for model selection for %s:\n", get_name())

	index_t num_param=m_model_selection_parameters->get_num_parameters();

	if (!num_param)
		SG_PRINT("\tnone\n")

	for (index_t i=0; i<num_param; i++)
	{
		TParameter* current=m_model_selection_parameters->get_parameter(i);
		index_t  l=200;
		char* type=SG_MALLOC(char, l);
		if (type)
		{
			current->m_datatype.to_string(type, l);
			SG_PRINT("\t%s (%s): %s\n", current->m_name, current->m_description,
					type);
			SG_FREE(type);
		}
	}
}

SGStringList<char> CSGObject::get_modelsel_names()
{
    index_t num_param=m_model_selection_parameters->get_num_parameters();

    SGStringList<char> result(num_param, -1);

	index_t max_string_length=-1;

    for (index_t i=0; i<num_param; i++)
    {
        char* name=m_model_selection_parameters->get_parameter(i)->m_name;
        index_t len=strlen(name);
		// +1 to have a zero terminated string
        result.strings[i]=SGString<char>(name, len+1);

        if (len>max_string_length)
            max_string_length=len;
    }

	result.max_string_length=max_string_length;

    return result;
}

char* CSGObject::get_modsel_param_descr(const char* param_name)
{
	index_t index=get_modsel_param_index(param_name);

	if (index<0)
	{
		SG_ERROR("There is no model selection parameter called \"%s\" for %s",
				param_name, get_name());
	}

	return m_model_selection_parameters->get_parameter(index)->m_description;
}

index_t CSGObject::get_modsel_param_index(const char* param_name)
{
	/* use fact that names extracted from below method are in same order than
	 * in m_model_selection_parameters variable */
	SGStringList<char> names=get_modelsel_names();

	/* search for parameter with provided name */
	index_t index=-1;
	for (index_t i=0; i<names.num_strings; i++)
	{
		TParameter* current=m_model_selection_parameters->get_parameter(i);
		if (!strcmp(param_name, current->m_name))
		{
			index=i;
			break;
		}
	}

	return index;
}

void CSGObject::get_parameter_incremental_hash(uint32_t& hash, uint32_t& carry,
		uint32_t& total_length)
{
	for (index_t i=0; i<m_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_parameters->get_parameter(i);

		SG_DEBUG("Updating hash for parameter %s.%s\n", get_name(), p->m_name);

		if (p->m_datatype.m_ptype == PT_SGOBJECT)
		{
			if (p->m_datatype.m_ctype == CT_SCALAR)
			{
				CSGObject* child = *((CSGObject**)(p->m_parameter));

				if (child)
				{
					child->get_parameter_incremental_hash(hash, carry,
							total_length);
				}
			}
			else if (p->m_datatype.m_ctype==CT_VECTOR ||
					p->m_datatype.m_ctype==CT_SGVECTOR)
			{
				CSGObject** child=(*(CSGObject***)(p->m_parameter));

				for (index_t j=0; j<*(p->m_datatype.m_length_y); j++)
				{
					if (child[j])
					{
						child[j]->get_parameter_incremental_hash(hash, carry,
								total_length);
					}
				}
			}
		}
		else
			p->get_incremental_hash(hash, carry, total_length);
	}
}

void CSGObject::build_gradient_parameter_dictionary(CMap<TParameter*, CSGObject*>* dict)
{
	for (index_t i=0; i<m_gradient_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_gradient_parameters->get_parameter(i);
		dict->add(p, this);
	}

	for (index_t i=0; i<m_model_selection_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_model_selection_parameters->get_parameter(i);
		CSGObject* child=*(CSGObject**)(p->m_parameter);

		if ((p->m_datatype.m_ptype == PT_SGOBJECT) &&
				(p->m_datatype.m_ctype == CT_SCALAR) &&	child)
		{
			child->build_gradient_parameter_dictionary(dict);
		}
	}
}

bool CSGObject::equals(CSGObject* other, float64_t accuracy, bool tolerant)
{
	SG_DEBUG("entering %s::equals()\n", get_name());

	if (other==this)
	{
		SG_DEBUG("leaving %s::equals(): other object is me\n", get_name());
		return true;
	}

	if (!other)
	{
		SG_DEBUG("leaving %s::equals(): other object is NULL\n", get_name());
		return false;
	}

	SG_DEBUG("comparing \"%s\" to \"%s\"\n", get_name(), other->get_name());

	/* a crude type check based on the get_name */
	if (strcmp(other->get_name(), get_name()))
	{
		SG_INFO("leaving %s::equals(): name of other object differs\n", get_name());
		return false;
	}

	/* should not be necessary but just ot be sure that type has not changed.
	 * Will assume that parameters are in same order with same name from here */
	if (m_parameters->get_num_parameters()!=other->m_parameters->get_num_parameters())
	{
		SG_INFO("leaving %s::equals(): number of parameters of other object "
				"differs\n", get_name());
		return false;
	}

	for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
	{
		SG_DEBUG("comparing parameter %d\n", i);

		TParameter* this_param=m_parameters->get_parameter(i);
		TParameter* other_param=other->m_parameters->get_parameter(i);

		/* some checks to make sure parameters have same order and names and
		 * are not NULL. Should never be the case but check anyway. */
		if (!this_param && !other_param)
			continue;

		if (!this_param && other_param)
		{
			SG_DEBUG("leaving %s::equals(): parameter %d is NULL where other's "
					"parameter \"%s\" is not\n", get_name(), other_param->m_name);
			return false;
		}

		if (this_param && !other_param)
		{
			SG_DEBUG("leaving %s::equals(): parameter %d is \"%s\" where other's "
						"parameter is NULL\n", get_name(), this_param->m_name);
			return false;
		}

		SG_DEBUG("comparing parameter \"%s\" to other's \"%s\"\n",
				this_param->m_name, other_param->m_name);

		/* hard-wired exception for DynamicObjectArray parameter num_elements */
		if (!strcmp("DynamicObjectArray", get_name()) &&
				!strcmp(this_param->m_name, "num_elements") &&
				!strcmp(other_param->m_name, "num_elements"))
		{
			SG_DEBUG("Ignoring DynamicObjectArray::num_elements field\n");
			continue;
		}

		/* hard-wired exception for DynamicArray parameter num_elements */
		if (!strcmp("DynamicArray", get_name()) &&
				!strcmp(this_param->m_name, "num_elements") &&
				!strcmp(other_param->m_name, "num_elements"))
		{
			SG_DEBUG("Ignoring DynamicArray::num_elements field\n");
			continue;
		}

		/* use equals method of TParameter from here */
		if (!this_param->equals(other_param, accuracy, tolerant))
		{
			SG_INFO("leaving %s::equals(): parameters at position %d with name"
					" \"%s\" differs from other object parameter with name "
					"\"%s\"\n",
					get_name(), i, this_param->m_name, other_param->m_name);
			return false;
		}
	}

	SG_DEBUG("leaving %s::equals(): object are equal\n", get_name());
	return true;
}

CSGObject* CSGObject::clone()
{
	SG_DEBUG("entering %s::clone()\n", get_name());

	SG_DEBUG("constructing an empty instance of %s\n", get_name());
	CSGObject* copy=new_sgserializable(get_name(), this->m_generic);

	SG_REF(copy);

	REQUIRE(copy, "Could not create empty instance of \"%s\". The reason for "
			"this usually is that get_name() of the class returns something "
			"wrong, or that a class has a wrongly set generic type.\n",
			get_name());

	for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
	{
		SG_DEBUG("cloning parameter \"%s\" at index %d\n",
				m_parameters->get_parameter(i)->m_name, i);

		if (!m_parameters->get_parameter(i)->copy(copy->m_parameters->get_parameter(i)))
		{
			SG_DEBUG("leaving %s::clone(): Clone failed. Returning NULL\n",
					get_name());
			return NULL;
		}
	}

	SG_DEBUG("leaving %s::clone(): Clone successful\n", get_name());
	return copy;
}
