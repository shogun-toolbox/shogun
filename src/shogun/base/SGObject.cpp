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

void CSGObject::set_global_version(Version* mapped_version)
{
	SG_UNREF(sg_version);
	sg_version=mapped_version;
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
								   const char* prefix)
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
	if (!save_parameter_version(file, prefix))
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
								   const char* prefix)
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

	if (file_version<0)
	{
		SG_WARNING("%s%s::load_serializable(): File contains no parameter "
				"version. Seems like your file is from the days before this "
				"was introduced. Ignore warning or serialize with this version "
				"of shogun to get rid of above and this warnings.\n",
				prefix, get_name());
	}

	if (file_version>version->get_version_parameter())
	{
		SG_WARNING("%s%s::load_serializable(): parameter version of file "
				"larger than the one of shogun. Try with a more recent version "
				"of shogun.\n", prefix, get_name());
		return false;
	}

	/* load all parameters from file */
	DynArray<TParameter*>* file_parameters=load_file_parameters(file_version,
			file, prefix);

	/* create an array of param infos from current parameters */
	DynArray<SGParamInfo*>* param_infos=new DynArray<SGParamInfo*>();
	for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
	{
		param_infos->append_element(new SGParamInfo(m_parameters->get_parameter(i),
				VERSION_PARAMETER));
	}

	/* map all parameters */
	SG_PRINT("all parameters:\n");
	m_parameter_map->print_map();
	SG_PRINT("version: %d\n", param_infos->get_element(0)->m_param_version);
	map_parameters(file_parameters, file_version, param_infos);
	SG_PRINT("mapping is done!\n");

	/* delete above created param infos */
	for (index_t i=0; i<param_infos->get_num_elements(); ++i)
		delete param_infos->get_element(i);

	delete param_infos;

	ASSERT(file_version==VERSION_PARAMETER);

	/* extract array for searching in next step */
	SGVector<TParameter*> v=SGVector<TParameter*> (file_parameters->get_array(),
			file_parameters->get_num_elements());

	/* replace parameters by loaded and mapped */
	SG_PRINT("replacing parameter data by new values\n");
	for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
	{
		TParameter* current=m_parameters->get_parameter(i);

		/* search for current parameter in mapped ones */
		index_t index=CMath::binary_search(v, current);
		TParameter* migrated=file_parameters->get_element(index);

		/* no mapped parameter was found, just load */
		if (index<0)
		{
			SG_NOTIMPLEMENTED;
			current->load(file, prefix);
		}
		/* mapped parameter was found, use it */
		else
		{
			SG_PRINT("replacing \"%s\"\n", current->m_name);
			memcpy(current->m_parameter,
					migrated->m_parameter,
					current->m_datatype.get_size());
		}
	}

	/* delete the migrated parameter data base */
	for (index_t i=0; i<file_parameters->get_num_elements(); ++i)
	{
		TParameter* current=file_parameters->get_element(i);
		SG_PRINT("deleting old %s\n", current->m_name);

		/* in case of scalar types, memory has to be deleted by hand
		 * (this is not done in delete_cont because scalar variables are
		 * normally on stack memory */
		current->delete_cont();
		if (current->m_datatype.m_ctype==CT_SCALAR)
			SG_FREE(current->m_parameter);

		delete current;
	}
	delete file_parameters;

//	if (!m_parameters->load(file, prefix))
//		return false;

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

TParameter* CSGObject::load_file_parameter(SGParamInfo* param_info,
		int32_t file_version, CSerializableFile* file, const char* prefix)
{
	/* ensure that recursion works */
	if (file_version>param_info->m_param_version)
		SG_ERROR("parameter version in file is more recent than provided!\n");

	TParameter* result;

	/* do mapping */
	SGParamInfo* old=m_parameter_map->get(param_info);

	/* if no mapping is found, versions have to be the same,
	 * or something went wrong */
	ASSERT(old || file_version==param_info->m_param_version);

	/* case file version same as provided version.
	 * means that parameter has to be loaded from file, recursion stops here */
	if (file_version==param_info->m_param_version)
	{
		/* allocate memory for length in case of matrix and vector */
		index_t* len_x=NULL;
		index_t* len_y=NULL;
		switch (param_info->m_ctype)
		{
		case CT_VECTOR: case CT_SGVECTOR:
			len_y=SG_MALLOC(index_t, 1);
			break;
		case CT_SGMATRIX: case CT_MATRIX:
			len_x=SG_MALLOC(index_t, 1);
			len_y=SG_MALLOC(index_t, 1);
			break;
		case CT_SCALAR:
			break;
		case CT_NDARRAY:
			SG_NOTIMPLEMENTED;
		default:
			break;
		}
		TSGDataType type(param_info->m_ctype, param_info->m_stype,
				param_info->m_ptype, len_y, len_x);

		/* allocate space for data, size depends on type */
		void* data=SG_MALLOC(char, type.get_size());

		/* create TParameter instance */
		result=new TParameter(&type, data, param_info->m_name, "");

		/* tell instance to load data from file */
		result->load(file, prefix);

		SG_SPRINT("loaded %s with value %d\n", result->m_name,
				*((int32_t*)result->m_parameter));
	}
	/* recursion with mapped type, a mapping exists in this case (ensured by
	 * above assert) */
	else
		result=load_file_parameter(old, file_version, file, prefix);

	return result;
}

DynArray<TParameter*>* CSGObject::load_file_parameters(int32_t file_version,
		CSerializableFile* file, const char* prefix)
{
	DynArray<TParameter*>* result=new DynArray<TParameter*>();

	for (index_t i=0; i<m_parameters->get_num_parameters(); ++i)
	{
		/* extract current parameter info */
		SGParamInfo* info=new SGParamInfo(m_parameters->get_parameter(i),
				VERSION_PARAMETER);

		/* load parameter data from file */
		result->append_element(load_file_parameter(info, file_version, file,
				prefix));

		/* clean up */
		delete info;
	}

	/* sort array before returning */
	SGVector<TParameter*> to_sort(result->get_array(),
			result->get_num_elements());
	CMath::qsort(to_sort);

	return result;
}

void CSGObject::map_parameters(DynArray<TParameter*>* param_base,
		int32_t& base_version, DynArray<SGParamInfo*>* target_param_infos)
{
	SG_PRINT("entering map_parameters\n");
	/* NOTE: currently the migration is done step by step over every version */

	/* map all target parameter infos once */
	DynArray<SGParamInfo*>* mapped_infos=new DynArray<SGParamInfo*>();
	for (index_t i=0; i<target_param_infos->get_num_elements(); ++i)
	{
		char* s=target_param_infos->get_element(i)->to_string();
		SG_PRINT("trying to get %s\n", s);
		SG_FREE(s);

		SGParamInfo* mapped=m_parameter_map->get(
				target_param_infos->get_element(i));

		if (mapped)
		{
			mapped_infos->append_element(mapped);
			s=mapped->to_string();
			SG_PRINT("found: %s\n", s);
			SG_FREE(s);
		}
	}

	/* assert that at least one mapping exists for now TODO*/
	ASSERT(mapped_infos->get_num_elements());
	int32_t mapped_version=mapped_infos->get_element(0)->m_param_version;

	/* assert that all param versions are equal for now TODO*/
	for (index_t i=1; i<mapped_infos->get_num_elements(); ++i)
		ASSERT(mapped_infos->get_element(i)->m_param_version==mapped_version);

	/* recursion, after this call, base is at version of mapped infos */
	if (mapped_version>base_version)
		map_parameters(param_base, base_version, mapped_infos);

	/* delete mapped parameter infos array */
	delete mapped_infos;

	ASSERT(base_version==mapped_version);

	/* do migration of one version step, create new base */
	DynArray<TParameter*>* new_base=new DynArray<TParameter*>();
	for (index_t i=0; i<target_param_infos->get_num_elements(); ++i)
	{
		char* s=target_param_infos->get_element(i)->to_string();
		SG_PRINT("migrating one step to target: %s\n", s);
		SG_FREE(s);
		TParameter* p=migrate(param_base, target_param_infos->get_element(i));
		new_base->append_element(p);
	}

	/* replace base by new base, delete old base */
	SG_PRINT("deleting parameters base version %d\n", base_version);
	for (index_t i=0; i<param_base->get_num_elements(); ++i)
	{
		TParameter* p=param_base->get_element(i);
		p->delete_cont();
		if (p->m_datatype.m_ctype==CT_SCALAR)
			SG_FREE(p->m_parameter);

		delete p;
	}

	SG_PRINT("replacing base\n");
	*param_base=*new_base;
	base_version=mapped_version+1;

	/* because content was copied, new base may be deleted */
	delete new_base;

	/* sort the just created new base */
	SG_PRINT("sorting base\n");
	SGVector<TParameter*> to_sort(param_base->get_array(),
			param_base->get_num_elements());
	CMath::qsort(to_sort);

	/* at this point the param_base is at the same version as the version of
	 * the provided parameter infos */
}

TParameter* CSGObject::migrate(DynArray<TParameter*>* param_base,
		SGParamInfo* target)
{
		char* s=target->to_string();
		SG_SERROR("No migration method available for %s!\n", s);
		SG_FREE(s);
		return NULL;
	}

bool CSGObject::save_parameter_version(CSerializableFile* file,
		const char* prefix)
{
	TSGDataType t(CT_SCALAR, ST_NONE, PT_INT32);
	int32_t v=version->get_version_parameter();
	TParameter p(&t, &v, "version_parameter",
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

SGVector<char*> CSGObject::get_modelsel_names()
{
	SGVector<char*> result=SGVector<char*>(
			m_model_selection_parameters->get_num_parameters());

	for (index_t i=0; i<result.vlen; ++i)
		result.vector[i]=m_model_selection_parameters->get_parameter(i)->m_name;

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
	SGVector<char*> names=get_modelsel_names();

	/* search for parameter with provided name */
	index_t index=-1;
	for (index_t i=0; i<names.vlen; ++i)
	{
		TParameter* current=m_model_selection_parameters->get_parameter(i);
		if (!strcmp(param_name, current->m_name))
		{
			index=i;
			break;
		}
	}

	/* clean up */
	names.destroy_vector();

	return index;
}
