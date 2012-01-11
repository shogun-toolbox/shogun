/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/init.h>
#include <shogun/base/Version.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/ParameterMap.h>
#include <shogun/base/DynArray.h>

#include <stdlib.h>
#include <stdio.h>

#include <shogun/features/SimpleFeatures.h>
namespace shogun
{
	class CMath;
	class Parallel;
	class IO;
	class Version;

	extern CMath* sg_math;
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

} /* namespace shogun  */

using namespace shogun;

CSGObject::CSGObject()
{
	init();
	set_global_objects();

	SG_GCDEBUG("SGObject created (%p)\n", this);
}

CSGObject::CSGObject(const CSGObject& orig)
:io(orig.io), parallel(orig.parallel), version(orig.version)
{
	init();
	set_global_objects();
}

CSGObject::~CSGObject()
{
	SG_GCDEBUG("SGObject destroyed (%p)\n", this);

#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_DESTROY(&m_ref_lock);
#endif
	unset_global_objects();
	delete m_parameters;
	delete m_model_selection_parameters;
	delete m_parameter_map;
}

#ifdef USE_REFERENCE_COUNTING

int32_t CSGObject::ref()
{
#ifdef HAVE_PTHREAD
		PTHREAD_LOCK(&m_ref_lock);
#endif //HAVE_PTHREAD
		++m_refcount;
		int32_t count=m_refcount;
#ifdef HAVE_PTHREAD
		PTHREAD_UNLOCK(&m_ref_lock);
#endif //HAVE_PTHREAD
		SG_GCDEBUG("ref() refcount %ld obj %s (%p) increased\n", count, this->get_name(), this);
		return m_refcount;
}

int32_t CSGObject::ref_count()
{
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK(&m_ref_lock);
#endif //HAVE_PTHREAD
	int32_t count=m_refcount;
#ifdef HAVE_PTHREAD
	PTHREAD_UNLOCK(&m_ref_lock);
#endif //HAVE_PTHREAD
	SG_GCDEBUG("ref_count(): refcount %d, obj %s (%p)\n", count, this->get_name(), this);
	return count;
}

int32_t CSGObject::unref()
{
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK(&m_ref_lock);
#endif //HAVE_PTHREAD
	if (m_refcount==0 || --m_refcount==0)
	{
		SG_GCDEBUG("unref() refcount %ld, obj %s (%p) destroying\n", m_refcount, this->get_name(), this);
#ifdef HAVE_PTHREAD
		PTHREAD_UNLOCK(&m_ref_lock);
#endif //HAVE_PTHREAD
		delete this;
		return 0;
	}
	else
	{
		SG_GCDEBUG("unref() refcount %ld obj %s (%p) decreased\n", m_refcount, this->get_name(), this);
#ifdef HAVE_PTHREAD
		PTHREAD_UNLOCK(&m_ref_lock);
#endif //HAVE_PTHREAD
		return m_refcount;
	}
}
#endif //USE_REFERENCE_COUNTING


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
	SG_UNREF(sg_io);
	sg_io=new_io;
	SG_REF(sg_io);
}

SGIO* CSGObject::get_global_io()
{
	SG_REF(sg_io);
	return sg_io;
}

void CSGObject::set_global_parallel(Parallel* new_parallel)
{
	SG_UNREF(sg_parallel);
	sg_parallel=new_parallel;
	SG_REF(sg_parallel);
}

Parallel* CSGObject::get_global_parallel()
{
	SG_REF(sg_parallel);
	return sg_parallel;
}

void CSGObject::set_global_version(Version* new_version)
{
	SG_UNREF(sg_version);
	sg_version=new_version;
	SG_REF(sg_version);
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
	SG_PRINT("\n%s\n================================================================================\n", get_name());
	m_parameters->print(prefix);
}

bool CSGObject::save_serializable(CSerializableFile* file,
		const char* prefix, int32_t param_version)
{
	SG_DEBUG("START SAVING CSGObject '%s'\n", get_name());
	try
	{
		save_serializable_pre();
	}
	catch (ShogunException e)
	{
		SG_SWARNING("%s%s::save_serializable_pre(): ShogunException: "
				   "%s\n", prefix, get_name(),
				   e.get_exception_string());
		return false;
	}
	if (!m_save_pre_called)
	{
		SG_SWARNING("%s%s::save_serializable_pre(): Implementation "
				   "error: BASE_CLASS::LOAD_SERIALIZABLE_PRE() not "
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
	catch (ShogunException e)
	{
		SG_SWARNING("%s%s::save_serializable_post(): ShogunException: "
				   "%s\n", prefix, get_name(),
				   e.get_exception_string());
		return false;
	}

	if (!m_save_post_called)
	{
		SG_SWARNING("%s%s::save_serializable_post(): Implementation "
				   "error: BASE_CLASS::LOAD_SERIALIZABLE_POST() not "
				   "called!\n", prefix, get_name());
		return false;
	}

	if (prefix == NULL || *prefix == '\0')
		file->close();

	SG_DEBUG("DONE SAVING CSGObject '%s' (%p)\n", get_name(), this);

	return true;;
}

bool CSGObject::load_serializable(CSerializableFile* file,
		const char* prefix, int32_t param_version)
{
	SG_DEBUG("START LOADING CSGObject '%s'\n", get_name());
	try
	{
		load_serializable_pre();
	}
	catch (ShogunException e)
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
//	SG_PRINT("file_version=%d, current_version=%d\n", file_version, param_version);

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
		if (param_version==VERSION_PARAMETER)
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
//		SG_PRINT("loading normally\n");

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

//		if (!m_parameters->load(file, prefix))
//			return false;
	}
	else
	{
		/* load all parameters from file, mappings to current version */
		DynArray<TParameter*>* param_base=load_file_parameters(file_version,
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

		/* map all parameters */
		map_parameters(param_base, file_version, param_infos);
//		SG_PRINT("mapping is done!\n");

		/* delete above created param infos */
		for (index_t i=0; i<param_infos->get_num_elements(); ++i)
			delete param_infos->get_element(i);

		delete param_infos;

		/* this is assumed now */
		ASSERT(file_version==param_version);

		/* replace parameters by loaded and mapped */
//		SG_PRINT("replacing parameter data by new values\n");
		for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
		{
			TParameter* current=m_parameters->get_parameter(i);

			/* skip new parameters */
			if (is_param_new(SGParamInfo(current, param_version)))
				continue;

			/* search for current parameter in mapped ones */
			index_t index=CMath::binary_search(param_base->get_array(),
					param_base->get_num_elements(), current);

			TParameter* migrated=param_base->get_element(index);

			/* now copy data from migrated TParameter instance
			 * (this automatically deletes the old data allocations) */
			current->copy_data(migrated);
		}

		/* delete the migrated parameter data base, data was copied so delete all
		 * data */
		for (index_t i=0; i<param_base->get_num_elements(); ++i)
		{
			TParameter* current=param_base->get_element(i);
//			SG_PRINT("deleting old \"%s\"\n", current->m_name);

			/* get rid of TParameter instance without deleting data, but lengths
			 * data pointer */
			current->delete_all_but_data();

			delete current;
		}
		delete param_base;
	}
//	SG_PRINT("loading is done\n");

	try
	{
		load_serializable_post();
	}
	catch (ShogunException e)
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
	SG_DEBUG("DONE LOADING CSGObject '%s' (%p)\n", get_name(), this);

	return true;
}

TParameter* CSGObject::load_file_parameter(const SGParamInfo* param_info,
		int32_t file_version, CSerializableFile* file, const char* prefix)
{
	/* ensure that recursion works */
	if (file_version>param_info->m_param_version)
	{
		SG_SERROR("parameter version of \"%s\" in file (%d) is more recent than"
				" provided %d!\n", param_info->m_name, file_version,
				param_info->m_param_version);
	}

	TParameter* result;

	/* do mapping */
//	char* s=param_info->to_string();
//	SG_SPRINT("try to get mapping for: %s\n", s);
//	SG_FREE(s);
	const SGParamInfo* old=m_parameter_map->get(param_info);
	bool free_old=false;
	if (!old)
	{
		/* if no mapping was found, nothing has changed. Simply create new param
		 * info with decreased version */
//		SG_SPRINT("no mapping found, ");
		if (file_version<param_info->m_param_version)
		{
			old=new SGParamInfo(param_info->m_name, param_info->m_ctype,
					param_info->m_stype, param_info->m_ptype,
					param_info->m_param_version-1);
			free_old=true;
//			s=old->to_string();
//			SG_SPRINT("using %s\n", s);
//			SG_FREE(s);
		}
//		else
//		{
//			SG_SPRINT("reached file version\n");
//		}
	}
//	else
//	{
//			s=old->to_string();
//			SG_SPRINT("found: %s\n", s);
//			SG_FREE(s);
//	}


	/* case file version same as provided version.
	 * means that parameter has to be loaded from file, recursion stops here */
	if (file_version==param_info->m_param_version)
	{
		/* allocate memory for length and matrix/vector
		 * This has to be done because this stuff normally is in the class
		 * variables which do not exist in this case. Deletion is handled
		 * via the m_delete_data flag of TParameter */

		/* create type and copy lengths, empty data for now
		 * dummy data will be created below. Note that the delete_data flag is
		 * set here which will handle the deletion of the data pointer, data,
		 * and possible length variables */
		TSGDataType type(param_info->m_ctype, param_info->m_stype,
				param_info->m_ptype);
		result=new TParameter(&type, NULL, param_info->m_name, "");
		result->m_delete_data=true;

		/* allocate data/length variables for the TParameter, lengths are not
		 * important now, so set to one */
		result->allocate_data_from_scratch(1, 1);

		/* tell instance to load data from file */
		if (!result->load(file, prefix))
		{
			char* s=param_info->to_string();
			SG_ERROR("Could not load %s. The reason for this might be wrong "
					"parameter mappings\n", s);
			SG_FREE(s);
		}
//		SG_SPRINT("done\n");
	}
	/* recursion with mapped type, a mapping exists in this case (ensured by
	 * above assert) */
	else
		result=load_file_parameter(old, file_version, file, prefix);

	if (free_old)
		delete old;

	return result;
}

DynArray<TParameter*>* CSGObject::load_file_parameters(int32_t file_version,
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

		/* in the other case, load parameter data from file */
		result->append_element(load_file_parameter(info, file_version, file,
				prefix));

		/* clean up */
		delete info;
	}

	/* sort array before returning */
	CMath::qsort(result->get_array(), result->get_num_elements());

	return result;
}

void CSGObject::map_parameters(DynArray<TParameter*>* param_base,
		int32_t& base_version, DynArray<const SGParamInfo*>* target_param_infos)
{
//	SG_PRINT("entering map_parameters\n");
	/* NOTE: currently the migration is done step by step over every version */

	/* map all target parameter infos once */
	DynArray<const SGParamInfo*>* mapped_infos=
			new DynArray<const SGParamInfo*>();
	DynArray<SGParamInfo*>* to_delete=new DynArray<SGParamInfo*>();
	for (index_t i=0; i<target_param_infos->get_num_elements(); ++i)
	{
		const SGParamInfo* current=target_param_infos->get_element(i);

//		char* s=current->to_string();
//		SG_PRINT("trying to get mapping for %s\n", s);
//		SG_FREE(s);

		const SGParamInfo* mapped=m_parameter_map->get(current);

		if (mapped)
		{
			mapped_infos->append_element(mapped);
//			s=mapped->to_string();
//			SG_PRINT("found: %s\n", s);
//			SG_FREE(s);
		}
		else
		{
			/* these have to be deleted above */
			SGParamInfo* no_change=new SGParamInfo(*current);
			no_change->m_param_version--;
//			s=no_change->to_string();
//			SG_PRINT("no mapping found, using %s\n", s);
//			SG_FREE(s);
			mapped_infos->append_element(no_change);
			to_delete->append_element(no_change);
		}
	}

	/* assert that at least one mapping exists */
	ASSERT(mapped_infos->get_num_elements());
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

	ASSERT(base_version==mapped_version);

	/* do migration of one version step, create new base */
	DynArray<TParameter*>* new_base=new DynArray<TParameter*>();
	for (index_t i=0; i<target_param_infos->get_num_elements(); ++i)
	{
//		char* s=target_param_infos->get_element(i)->to_string();
//		SG_PRINT("migrating one step to target: %s\n", s);
//		SG_FREE(s);
		TParameter* p=migrate(param_base, target_param_infos->get_element(i));
		new_base->append_element(p);
	}

	/* replace base by new base, delete old base, if it was created in migrate */
//	SG_PRINT("deleting parameters base version %d\n", base_version);
	for (index_t i=0; i<param_base->get_num_elements(); ++i)
		delete param_base->get_element(i);

//	SG_PRINT("replacing base\n");
	*param_base=*new_base;
	base_version=mapped_version+1;

//	SG_PRINT("new param base:\n");
//	for (index_t i=0; i<param_base->get_num_elements(); ++i)
//	{
//		SG_PRINT("%s, ", param_base->get_element(i)->m_name);
//		param_base->get_element(i)->print("");
//	}
//	SG_PRINT("\n");

	/* because content was copied, new base may be deleted */
	delete new_base;

	/* sort the just created new base */
//	SG_PRINT("sorting base\n");
	CMath::qsort(param_base->get_array(), param_base->get_num_elements());

	/* at this point the param_base is at the same version as the version of
	 * the provided parameter infos */
//	SG_PRINT("leaving map_parameters\n");
}

void CSGObject::one_to_one_migration_prepare(DynArray<TParameter*>* param_base,
		const SGParamInfo* target, TParameter*& replacement,
		TParameter*& to_migrate, char* old_name)
{
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
	ASSERT(i>=0);
	to_migrate=param_base->get_element(i);

	/* result structure, data NULL for now */
	replacement=new TParameter(&type, NULL, target->m_name,
			to_migrate->m_description);

	/* allocate content to write into, lengths are needed for this */
	index_t len_x=1;
	if (to_migrate->m_datatype.m_length_x!=NULL)
		len_x=*to_migrate->m_datatype.m_length_x;

	index_t len_y=1;
	if (to_migrate->m_datatype.m_length_y!=NULL)
		len_y=*to_migrate->m_datatype.m_length_y;

//	SG_SPRINT("allocate_data_from_scratch call with len_y=%d, len_x=%d\n", len_y, len_x);
	replacement->allocate_data_from_scratch(len_y, len_x);

	/* in case of sgobject, copy pointer data and SG_REF */
	if (to_migrate->m_datatype.m_ptype==PT_SGOBJECT)
	{
		/* note that the memory is already allocated before the migrate call */
		CSGObject* object=*((CSGObject**)to_migrate->m_parameter);
		*((CSGObject**)replacement->m_parameter)=object;
		SG_REF(object);
	}

	/* tell the old TParameter to delete its data on deletion */
	to_migrate->m_delete_data=true;
}

TParameter* CSGObject::migrate(DynArray<TParameter*>* param_base,
		const SGParamInfo* target)
{
//	SG_PRINT("entering CSGObject::migrate by %s\n", get_name());
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
		SG_ERROR("Name change for parameter that has to be mapped to \"%s\","
				" and to no migration method available\n", target->m_name);

	TParameter* to_migrate=param_base->get_element(i);

	/* check if element in base is equal to target one */
	if (*target==SGParamInfo(to_migrate, target->m_param_version))
	{
//		SG_PRINT("nothing changed, using old data\n");
		result=new TParameter(&to_migrate->m_datatype, to_migrate->m_parameter,
				to_migrate->m_name, to_migrate->m_description);

		/* in case of sgobject, allocate data for pointer (not done yet)
		 * and copy the value of the pointer to not loose the sgobject */
		if (to_migrate->m_datatype.m_ptype==PT_SGOBJECT)
		{
			result->m_parameter=SG_MALLOC(CSGObject*, 1);
			CSGObject* object=*((CSGObject**)to_migrate->m_parameter);
			*((CSGObject**)result->m_parameter)=object;
			SG_REF(object);
		}
	}
	else
	{
		char* s=target->to_string();
		SG_ERROR("No migration method available for %s!\n", s);
		SG_FREE(s);
	}

	SG_PRINT("leaving CSGObject::migrate\n");

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
#include <shogun/lib/Set.h>
extern CSet<shogun::MemoryBlock>* sg_mallocs;
#endif

void CSGObject::init()
{
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_INIT(&m_ref_lock);
#endif

#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
	{
		int32_t idx=sg_mallocs->index_of(MemoryBlock(this));
		if (idx>-1)
		{
			MemoryBlock* b=sg_mallocs->get_element_ptr(idx);
			b->set_sgobject();
		}
	}
#endif

	m_refcount = 0;
	io = NULL;
	parallel = NULL;
	version = NULL;
	m_parameters = new Parameter();
	m_model_selection_parameters = new Parameter();
	m_parameter_map=new ParameterMap();
	m_generic = PT_NOT_GENERIC;
	m_load_pre_called = false;
	m_load_post_called = false;
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

	/* clean up */
	names.destroy_list();

	return index;
}

bool CSGObject::is_param_new(const SGParamInfo param_info) const
{
	/* check if parameter is new in this version (has empty mapping) */
	const SGParamInfo* value=m_parameter_map->get(&param_info);
	bool result=value && *value==SGParamInfo();

//	if (result)
//		SG_PRINT("\"%s\" is new\n", param_info.m_name);

	return result;
}
