/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Written (W) 2011-2013 Heiko Strathmann
 * Written (W) 2013 Thoralf Klein
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Version.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/ParameterMap.h>
#include <shogun/base/DynArray.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/io/SerializableFile.h>

#include "class_list.h"

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
: SGRefObject()
{
	init();
	set_global_objects();
}

CSGObject::CSGObject(const CSGObject& orig)
:SGRefObject(orig), io(orig.io), parallel(orig.parallel), version(orig.version)
{
	init();
	set_global_objects();
}

CSGObject::~CSGObject()
{
	unset_global_objects();
	delete m_parameters;
	delete m_model_selection_parameters;
	delete m_gradient_parameters;
	delete m_parameter_map;
}

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
		const char* prefix, int32_t param_version)
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

	/* save parameter version */
	if (!save_parameter_version(file, prefix, param_version))
		return false;

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
		const char* prefix, int32_t param_version)
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

	/* try to load version of parameters */
	int32_t file_version=load_parameter_version(file, prefix);
	SG_DEBUG("file_version=%d, current_version=%d\n", file_version, param_version)

	if (file_version<0)
	{
		SG_WARNING("%s%s::load_serializable(): File contains no parameter "
			"version. Seems like your file is from the days before this "
			"was introduced. Ignore warning or serialize with this version "
			"of shogun to get rid of above and this warnings.\n",
			prefix, get_name());
	}

	if (file_version>param_version)
	{
		if (param_version==Version::get_version_parameter())
		{
			SG_WARNING("%s%s::load_serializable(): parameter version of file "
					"larger than the one of shogun. Try with a more recent"
					"version of shogun.\n", prefix, get_name());
		}
		else
		{
			SG_WARNING("%s%s::load_serializable(): parameter version of file "
					"larger than the current. This is probably an implementation"
					" error.\n", prefix, get_name());
		}
		return false;
	}

	if (file_version==param_version)
	{
		/* load normally if file has current version */
		SG_DEBUG("loading normally\n")

		/* load all parameters, except new ones */
		for (int32_t i=0; i<m_parameters->get_num_parameters(); i++)
		{
			TParameter* current=m_parameters->get_parameter(i);

			/* skip new parameters */
			if (is_param_new(SGParamInfo(current, param_version)))
				continue;

			if (!current->load(file, prefix))
				return false;
		}
	}
	else
	{
		/* load all parameters from file, mappings to current version */
		DynArray<TParameter*>* param_base=load_all_file_parameters(file_version,
				param_version, file, prefix);

		/* create an array of param infos from current parameters */
		DynArray<const SGParamInfo*>* param_infos=
				new DynArray<const SGParamInfo*>();
		for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
		{
			TParameter* current=m_parameters->get_parameter(i);

			/* skip new parameters */
			if (is_param_new(SGParamInfo(current, param_version)))
				continue;

			param_infos->append_element(
					new SGParamInfo(current, param_version));
		}

		/* map all parameters, result may be empty if input is */
		map_parameters(param_base, file_version, param_infos);
		SG_DEBUG("mapping is done!\n")

		/* this is assumed now, mapping worked or no parameters in base */
		ASSERT(file_version==param_version || !param_base->get_num_elements())

		/* delete above created param infos */
		for (index_t i=0; i<param_infos->get_num_elements(); ++i)
			delete param_infos->get_element(i);

		delete param_infos;

		/* replace parameters by loaded and mapped */
		SG_DEBUG("replacing parameter data by loaded/mapped values\n")
		for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
		{
			TParameter* current=m_parameters->get_parameter(i);
			char* s=SG_MALLOC(char, 200);
			current->m_datatype.to_string(s, 200);
			SG_DEBUG("processing \"%s\": %s\n", current->m_name, s)
			SG_FREE(s);

			/* skip new parameters */
			if (is_param_new(SGParamInfo(current, param_version)))
			{
				SG_DEBUG("%s is new, skipping\n", current->m_name)
				continue;
			}

			/* search for current parameter in mapped ones */
			index_t index=CMath::binary_search(param_base->get_array(),
					param_base->get_num_elements(), current);

			TParameter* migrated=param_base->get_element(index);

			/* now copy data from migrated TParameter instance
			 * (this automatically deletes the old data allocations) */
			SG_DEBUG("copying migrated data into parameter\n")
			current->copy_data(migrated);
		}

		/* delete the migrated parameter data base */
		SG_DEBUG("deleting old parameter base\n")
		for (index_t i=0; i<param_base->get_num_elements(); ++i)
		{
			TParameter* current=param_base->get_element(i);
			SG_DEBUG("deleting old \"%s\"\n", current->m_name)
			delete current;
		}
		delete param_base;
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

DynArray<TParameter*>* CSGObject::load_file_parameters(
		const SGParamInfo* param_info, int32_t file_version,
		CSerializableFile* file, const char* prefix)
{
	/* ensure that recursion works */
	SG_SDEBUG("entering %s::load_file_parameters\n", get_name())
	if (file_version>param_info->m_param_version)
	{
		SG_SERROR("parameter version of \"%s\" in file (%d) is more recent than"
				" provided %d!\n", param_info->m_name, file_version,
				param_info->m_param_version);
	}

	DynArray<TParameter*>* result_array=new DynArray<TParameter*>();

	/* do mapping */
	char* s=param_info->to_string();
	SG_SDEBUG("try to get mapping for: %s\n", s)
	SG_FREE(s);

	/* mapping has only be deleted if was created here (no mapping was found) */
	bool free_mapped=false;
	DynArray<const SGParamInfo*>* mapped=m_parameter_map->get(param_info);
	if (!mapped)
	{
		/* since a new mapped array will be created, set deletion flag */
		free_mapped=true;
		mapped=new DynArray<const SGParamInfo*>();

		/* if no mapping was found, nothing has changed. Simply create new param
		 * info with decreased version */
		SG_SDEBUG("no mapping found\n")
		if (file_version<param_info->m_param_version)
		{
			/* create new array and put param info with decreased version in */
			mapped->append_element(new SGParamInfo(param_info->m_name,
					param_info->m_ctype, param_info->m_stype,
					param_info->m_ptype, param_info->m_param_version-1));

			SG_SDEBUG("using:\n")
			for (index_t i=0; i<mapped->get_num_elements(); ++i)
			{
				s=mapped->get_element(i)->to_string();
				SG_SDEBUG("\t%s\n", s)
				SG_FREE(s);
			}
		}
		else
		{
			/* create new array and put original param info in */
			SG_SDEBUG("reached file version\n")
			mapped->append_element(param_info->duplicate());
		}
	}
	else
	{
		SG_SDEBUG("found:\n")
		for (index_t i=0; i<mapped->get_num_elements(); ++i)
		{
			s=mapped->get_element(i)->to_string();
			SG_SDEBUG("\t%s\n", s)
			SG_FREE(s);
		}
	}


	/* case file version same as provided version.
	 * means that parameters have to be loaded from file, recursion stops */
	if (file_version==param_info->m_param_version)
	{
		SG_SDEBUG("recursion stop, loading from file\n")
		/* load all parameters in mapping from file */
		for (index_t i=0; i<mapped->get_num_elements(); ++i)
		{
			const SGParamInfo* current=mapped->get_element(i);
			s=current->to_string();
			SG_SDEBUG("loading %s\n", s)
			SG_FREE(s);

			TParameter* loaded;
			/* allocate memory for length and matrix/vector
			 * This has to be done because this stuff normally is in the class
			 * variables which do not exist in this case. Deletion is handled
			 * via the allocated_from_scratch flag of TParameter */

			/* create type and copy lengths, empty data for now */
			TSGDataType type(current->m_ctype, current->m_stype,
					current->m_ptype);
			loaded=new TParameter(&type, NULL, current->m_name, "");

			/* allocate data/length variables for the TParameter, lengths are not
			 * important now, so set to one */
			SGVector<index_t> dims(2);
			dims[0]=1;
			dims[1]=1;
			loaded->allocate_data_from_scratch(dims);

			/* tell instance to load data from file */
			if (!loaded->load(file, prefix))
			{
				s=param_info->to_string();
				SG_ERROR("Could not load %s. The reason for this might be wrong "
						"parameter mappings\n", s);
				SG_FREE(s);
			}

			SG_DEBUG("loaded lengths: y=%d, x=%d\n",
					loaded->m_datatype.m_length_y ? *loaded->m_datatype.m_length_y : -1,
					loaded->m_datatype.m_length_x ? *loaded->m_datatype.m_length_x : -1);

			/* append new TParameter to result array */
			result_array->append_element(loaded);
		}
		SG_SDEBUG("done loading from file\n")
	}
	/* recursion with mapped type, a mapping exists in this case (ensured by
	 * above assert) */
	else
	{
		/* for all elements in mapping, do recursion */
		for (index_t i=0; i<mapped->get_num_elements(); ++i)
		{
			const SGParamInfo* current=mapped->get_element(i);
			s=current->to_string();
			SG_SDEBUG("starting recursion over %s\n", s)

			/* recursively get all file parameters for this parameter */
			DynArray<TParameter*>* recursion_array=
					load_file_parameters(current, file_version, file, prefix);

			SG_SDEBUG("recursion over %s done\n", s)
			SG_FREE(s);

			/* append all recursion data to current array */
			SG_SDEBUG("appending all results to current result\n")
			for (index_t j=0; j<recursion_array->get_num_elements(); ++j)
				result_array->append_element(recursion_array->get_element(j));

			/* clean up */
			delete recursion_array;
		}
	}

	SG_SDEBUG("cleaning up old mapping \n")


	/* clean up mapping */
	if (free_mapped)
	{
		for (index_t i=0; i<mapped->get_num_elements(); ++i)
			delete mapped->get_element(i);

		delete mapped;
	}

	SG_SDEBUG("leaving %s::load_file_parameters\n", get_name())
	return result_array;
}

DynArray<TParameter*>* CSGObject::load_all_file_parameters(int32_t file_version,
		int32_t current_version, CSerializableFile* file, const char* prefix)
{
	DynArray<TParameter*>* result=new DynArray<TParameter*>();

	for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
	{
		TParameter* current=m_parameters->get_parameter(i);

		/* extract current parameter info */
		const SGParamInfo* info=new SGParamInfo(current, current_version);

		/* skip new parameters */
		if (is_param_new(*info))
		{
			delete info;
			continue;
		}

		/* in the other case, load parameters data from file */
		DynArray<TParameter*>* temp=load_file_parameters(info, file_version,
				file, prefix);

		/* and append them all to array */
		for (index_t j=0; j<temp->get_num_elements(); ++j)
			result->append_element(temp->get_element(j));

		/* clean up */
		delete temp;
		delete info;
	}

	/* sort array before returning */
	CMath::qsort(result->get_array(), result->get_num_elements());

	return result;
}

void CSGObject::map_parameters(DynArray<TParameter*>* param_base,
		int32_t& base_version, DynArray<const SGParamInfo*>* target_param_infos)
{
	SG_DEBUG("entering %s::map_parameters\n", get_name())
	/* NOTE: currently the migration is done step by step over every version */

	if (!target_param_infos->get_num_elements())
	{
		SG_DEBUG("no target parameter infos\n")
		SG_DEBUG("leaving %s::map_parameters\n", get_name())
		return;
	}

	/* map all target parameter infos once */
	DynArray<const SGParamInfo*>* mapped_infos=
			new DynArray<const SGParamInfo*>();
	DynArray<SGParamInfo*>* to_delete=new DynArray<SGParamInfo*>();
	for (index_t i=0; i<target_param_infos->get_num_elements(); ++i)
	{
		const SGParamInfo* current=target_param_infos->get_element(i);

		char* s=current->to_string();
		SG_DEBUG("trying to get parameter mapping for %s\n", s)
		SG_FREE(s);

		DynArray<const SGParamInfo*>* mapped=m_parameter_map->get(current);

		if (mapped)
		{
			mapped_infos->append_element(mapped->get_element(0));
			for (index_t j=0; j<mapped->get_num_elements(); ++j)
			{
				s=mapped->get_element(j)->to_string();
				SG_DEBUG("found mapping: %s\n", s)
				SG_FREE(s);
			}
		}
		else
		{
			/* these have to be deleted above */
			SGParamInfo* no_change=new SGParamInfo(*current);
			no_change->m_param_version--;
			s=no_change->to_string();
			SG_DEBUG("no mapping found, using %s\n", s)
			SG_FREE(s);
			mapped_infos->append_element(no_change);
			to_delete->append_element(no_change);
		}
	}

	/* assert that at least one mapping exists */
	ASSERT(mapped_infos->get_num_elements())
	int32_t mapped_version=mapped_infos->get_element(0)->m_param_version;

	/* assert that all param versions are equal for now (if not empty param) */
	for (index_t i=1; i<mapped_infos->get_num_elements(); ++i)
	{
		ASSERT(mapped_infos->get_element(i)->m_param_version==mapped_version ||
				*mapped_infos->get_element(i)==SGParamInfo());
	}

	/* recursion, after this call, base is at version of mapped infos */
	if (mapped_version>base_version)
		map_parameters(param_base, base_version, mapped_infos);

	/* delete mapped parameter infos array */
	delete mapped_infos;

	/* delete newly created parameter infos which have to name or type change */
	for (index_t i=0; i<to_delete->get_num_elements(); ++i)
		delete to_delete->get_element(i);

	delete to_delete;

	ASSERT(base_version==mapped_version)

	/* do migration of one version step, create new base */
	DynArray<TParameter*>* new_base=new DynArray<TParameter*>();
	for (index_t i=0; i<target_param_infos->get_num_elements(); ++i)
	{
		char* s=target_param_infos->get_element(i)->to_string();
		SG_DEBUG("migrating one step to target: %s\n", s)
		SG_FREE(s);
		TParameter* p=migrate(param_base, target_param_infos->get_element(i));
		new_base->append_element(p);
	}

	/* replace base by new base, delete old base, if it was created in migrate */
	SG_DEBUG("deleting parameters base version %d\n", base_version)
	for (index_t i=0; i<param_base->get_num_elements(); ++i)
		delete param_base->get_element(i);

	SG_DEBUG("replacing old parameter base\n")
	*param_base=*new_base;
	base_version=mapped_version+1;

	SG_DEBUG("new parameter base of size %d:\n", param_base->get_num_elements())
	for (index_t i=0; i<param_base->get_num_elements(); ++i)
	{
		TParameter* current=param_base->get_element(i);
		TSGDataType type=current->m_datatype;
		if (type.m_ptype==PT_SGOBJECT)
		{
			if (type.m_ctype==CT_SCALAR)
			{
				CSGObject* object=*(CSGObject**)current->m_parameter;
				SG_DEBUG("(%d:) \"%s\": sgobject \"%s\" at %p\n", i,
						current->m_name, object ? object->get_name() : "",
								object);
			}
			else
			{
				index_t len=1;
				len*=type.m_length_x ? *type.m_length_x : 1;
				len*=type.m_length_y ? *type.m_length_y : 1;
				CSGObject** array=*(CSGObject***)current->m_parameter;
				for (index_t j=0; j<len; ++j)
				{
					CSGObject* object=array[j];
					SG_DEBUG("(%d:) \"%s\": sgobject \"%s\" at %p\n", i,
							current->m_name, object ? object->get_name() : "",
									object);
				}
			}
		}
		else
		{
			char* s=SG_MALLOC(char, 200);
			current->m_datatype.to_string(s, 200);
			SG_DEBUG("(%d:) \"%s\": type: %s at %p\n", i, current->m_name, s,
					current->m_parameter);
			SG_FREE(s);
		}
	}

	/* because content was copied, new base may be deleted */
	delete new_base;

	/* sort the just created new base */
	SG_DEBUG("sorting base\n")
	CMath::qsort(param_base->get_array(), param_base->get_num_elements());

	/* at this point the param_base is at the same version as the version of
	 * the provided parameter infos */
	SG_DEBUG("leaving %s::map_parameters\n", get_name())
}

void CSGObject::one_to_one_migration_prepare(DynArray<TParameter*>* param_base,
		const SGParamInfo* target, TParameter*& replacement,
		TParameter*& to_migrate, char* old_name)
{
	SG_DEBUG("CSGObject::entering CSGObject::one_to_one_migration_prepare() for "
			"\"%s\"\n", target->m_name);

	/* generate type of target structure */
	TSGDataType type(target->m_ctype, target->m_stype, target->m_ptype);

	/* first find index of needed data.
	 * in this case, element in base with same name or old name */
	char* name=target->m_name;
	if (old_name)
		name=old_name;

	/* dummy for searching, search and save result in to_migrate parameter */
	TParameter* t=new TParameter(&type, NULL, name, "");
	index_t i=CMath::binary_search(param_base->get_array(),
			param_base->get_num_elements(), t);
	delete t;

	/* assert that something is found */
	ASSERT(i>=0)
	to_migrate=param_base->get_element(i);

	/* result structure, data NULL for now */
	replacement=new TParameter(&type, NULL, target->m_name,
			to_migrate->m_description);

	SGVector<index_t> dims(2);
	dims[0]=1;
	dims[1]=1;
	/* allocate content to write into, lengths are needed for this */
	if (to_migrate->m_datatype.m_length_x)
		dims[0]=*to_migrate->m_datatype.m_length_x;

	if (to_migrate->m_datatype.m_length_y)
		dims[1]=*to_migrate->m_datatype.m_length_y;

	replacement->allocate_data_from_scratch(dims);

	/* in case of sgobject, copy pointer data and SG_REF */
	if (to_migrate->m_datatype.m_ptype==PT_SGOBJECT)
	{
		/* note that the memory is already allocated before the migrate call */
		CSGObject* object=*((CSGObject**)to_migrate->m_parameter);
		*((CSGObject**)replacement->m_parameter)=object;
		SG_REF(object);
		SG_DEBUG("copied and SG_REF sgobject pointer for \"%s\" at %p\n",
				object->get_name(), object);
	}

	/* tell the old TParameter to delete its data on deletion */
	to_migrate->m_delete_data=true;

	SG_DEBUG("CSGObject::leaving CSGObject::one_to_one_migration_prepare() for "
			"\"%s\"\n", target->m_name);
}

TParameter* CSGObject::migrate(DynArray<TParameter*>* param_base,
		const SGParamInfo* target)
{
	SG_DEBUG("entering %s::migrate\n", get_name())
	/* this is only executed, iff there was no migration method which handled
	 * migration to the provided target. In this case, it is assumed that the
	 * parameter simply has not changed. Verify this here and return copy of
	 * data in case its true.
	 * If not, throw an exception -- parameter migration HAS to be implemented
	 * by hand everytime, a parameter changes type or name. */

	TParameter* result=NULL;

	/* first find index of needed data.
	 * in this case, element in base with same name */
	/* type is also needed */
	TSGDataType type(target->m_ctype, target->m_stype,
			target->m_ptype);

	/* dummy for searching, search and save result */
	TParameter* t=new TParameter(&type, NULL, target->m_name, "");
	index_t i=CMath::binary_search(param_base->get_array(),
			param_base->get_num_elements(), t);
	delete t;

	/* check if name change occurred while no migration method was specified */
	if (i<0)
	{
		SG_ERROR("Name change for parameter that has to be mapped to \"%s\","
				" and to no migration method available\n", target->m_name);
	}

	TParameter* to_migrate=param_base->get_element(i);

	/* check if element in base is equal to target one */
	if (*target==SGParamInfo(to_migrate, target->m_param_version))
	{
		char* s=SG_MALLOC(char, 200);
		to_migrate->m_datatype.to_string(s, 200);
		SG_DEBUG("nothing changed, using old data: %s\n", s)
		SG_FREE(s);
		result=new TParameter(&to_migrate->m_datatype, NULL, to_migrate->m_name,
				to_migrate->m_description);

		SGVector<index_t> dims(2);
		dims[0]=1;
		dims[1]=1;
		if (to_migrate->m_datatype.m_length_x)
			dims[0]=*to_migrate->m_datatype.m_length_x;

		if (to_migrate->m_datatype.m_length_y)
			dims[1]=*to_migrate->m_datatype.m_length_y;

		/* allocate lengths and evtl scalar data but not non-scalar data (no
		 * new_cont call */
		result->allocate_data_from_scratch(dims, false);

		/* now use old data */
		if (to_migrate->m_datatype.m_ctype==CT_SCALAR &&
				to_migrate->m_datatype.m_ptype!=PT_SGOBJECT)
		{
			/* copy data */
			SG_DEBUG("copying scalar data\n")
			memcpy(result->m_parameter,to_migrate->m_parameter,
					to_migrate->m_datatype.get_size());
		}
		else
		{
			/* copy content of pointer */
			SG_DEBUG("copying content of poitner for non-scalar data\n")
			*(void**)result->m_parameter=*(void**)(to_migrate->m_parameter);
		}
	}
	else
	{
		char* s=target->to_string();
		SG_ERROR("No migration method available for %s!\n", s)
		SG_FREE(s);
	}

	SG_DEBUG("leaving %s::migrate\n", get_name())

	return result;
}

bool CSGObject::save_parameter_version(CSerializableFile* file,
		const char* prefix, int32_t param_version)
{
	TSGDataType t(CT_SCALAR, ST_NONE, PT_INT32);
	TParameter p(&t, &param_version, "version_parameter",
			"Version of parameters of this object");
	return p.save(file, prefix);
}

int32_t CSGObject::load_parameter_version(CSerializableFile* file,
		const char* prefix)
{
	REQUIRE(file != NULL, "Serializable file object should be != NULL");

	TSGDataType t(CT_SCALAR, ST_NONE, PT_INT32);
	int32_t v;
	TParameter tp(&t, &v, "version_parameter", "");
	if (tp.load(file, prefix))
		return v;
	else
		return -1;
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
	m_parameter_map=new ParameterMap();
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

bool CSGObject::is_param_new(const SGParamInfo param_info) const
{
	/* check if parameter is new in this version (has empty mapping) */
	DynArray<const SGParamInfo*>* value=m_parameter_map->get(&param_info);
	bool result=value && *value->get_element(0) == SGParamInfo();

	return result;
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
