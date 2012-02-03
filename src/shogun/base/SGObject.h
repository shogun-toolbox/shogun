/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2010 Soeren Sonnenburg
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2008-2010 Fraunhofer Institute FIRST and Max Planck Society
 */

#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/lib/memory.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/Version.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif //HAVE_PTHREAD

/** \namespace shogun
 * @brief all of classes and functions are contained in the shogun namespace
 */
namespace shogun
{
class IO;
class Parallel;
class Version;
class Parameter;
class ParameterMap;
class SGParamInfo;
class CSerializableFile;
struct TParameter;
template <class T> class DynArray;

// define reference counter macros
//
#ifdef USE_REFERENCE_COUNTING
#define SG_REF(x) { if (x) (x)->ref(); }
#define SG_UNREF(x) { if (x) { if ((x)->unref()==0) (x)=NULL; } }
#define SG_UNREF_NO_NULL(x) { if (x) { (x)->unref(); } }
#else
#define SG_REF(x)
#define SG_UNREF(x)
#define SG_UNREF_NO_NULL(x)
#endif

/*******************************************************************************
 * Macros for registering parameters/model selection parameters
 ******************************************************************************/
#define SG_ADD(param, name, description, ms_available) {\
		m_parameters->add(param, name, description);\
		if (ms_available)\
			m_model_selection_parameters->add(param, name, description);\
}
/*******************************************************************************
 * End of macros for registering parameters/model selection parameters
 ******************************************************************************/

/** model selection availability */
enum EModelSelectionAvailability {
	MS_NOT_AVAILABLE=0, MS_AVAILABLE
};

/** @brief Class SGObject is the base class of all shogun objects.
 *
 * Apart from dealing with reference counting that is used to manage shogung
 * objects in memory (erase unused object, avoid cleaning objects when they are
 * still in use), it provides interfaces for:
 *
 * -# parallel - to determine the number of used CPUs for a method (cf. Parallel)
 * -# io - to output messages and general i/o (cf. IO)
 * -# version - to provide version information of the shogun version used (cf. Version)
 */
class CSGObject
{
public:
	/** default constructor */
	CSGObject();

	/** copy constructor */
	CSGObject(const CSGObject& orig);

	/** destructor */
	virtual ~CSGObject();

#ifdef USE_REFERENCE_COUNTING
	/** increase reference counter
	 *
	 * @return reference count
	 */
	int32_t ref();

	/** display reference counter
	 *
	 * @return reference count
	 */
	int32_t ref_count();

	/** decrement reference counter and deallocate object if refcount is zero
	 * before or after decrementing it
	 *
	 * @return reference count
	 */
	int32_t unref();
#endif //USE_REFERENCE_COUNTING

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name() const = 0;

	/** If the SGSerializable is a class template then TRUE will be
	 *  returned and GENERIC is set to the type of the generic.
	 *
	 *  @param generic set to the type of the generic if returning
	 *                 TRUE
	 *
	 *  @return TRUE if a class template.
	 */
	virtual bool is_generic(EPrimitiveType* generic) const;

	/** set generic type to T
	 */
	template<class T> void set_generic();

	/** unset generic type
	 *
	 * this has to be called in classes specializing a template class
	 */
	void unset_generic();

	/** prints registered parameters out
	 *
	 * 	@param prefix prefix for members
	 */
	virtual void print_serializable(const char* prefix="");

	/** Save this object to file.
	 *
	 * @param file where to save the object; will be closed during
	 * returning if PREFIX is an empty string.
	 * @param prefix prefix for members
	 * @param param_version (optional) a parameter version different to (this
	 * is mainly for testing, better do not use)
	 * @return TRUE if done, otherwise FALSE
	 */
	virtual bool save_serializable(CSerializableFile* file,
			const char* prefix="", int32_t param_version=VERSION_PARAMETER);

	/** Load this object from file.  If it will fail (returning FALSE)
	 *  then this object will contain inconsistent data and should not
	 *  be used!
	 *
	 *  @param file where to load from
	 *  @param prefix prefix for members
	 *  @param param_version (optional) a parameter version different to (this
	 * is mainly for testing, better do not use)
	 *
	 *  @return TRUE if done, otherwise FALSE
	 */
	virtual bool load_serializable(CSerializableFile* file,
			const char* prefix="", int32_t param_version=VERSION_PARAMETER);

	/** loads some specified parameters from a file with a specified version
	 * The provided parameter info has a version which is recursively mapped
	 * until the file parameter version is reached.
	 * Note that there may be possibly multiple parameters in the mapping,
	 * therefore, a set of TParameter instances is returned
	 *
	 * @param param_info information of parameter
	 * @param file_version parameter version of the file, must be <= provided
	 * parameter version
	 * @param file file to load from
	 * @param prefix prefix for members
	 * @return new array with TParameter instances with the attached data
	 */
	DynArray<TParameter*>* load_file_parameters(const SGParamInfo* param_info,
			int32_t file_version, CSerializableFile* file,
			const char* prefix="");

	/** maps all parameters of this instance to the provided file version and
	 * loads all parameter data from the file into an array, which is sorted
	 * (basically calls load_file_parameter(...) for all parameters and puts all
	 * results into a sorted array)
	 *
	 * @param file_version parameter version of the file
	 * @param current_version version from which mapping begins (you want to use
	 * VERSION_PARAMETER for this in most cases)
	 * @param file file to load from
	 * @param prefix prefix for members
	 * @return (sorted) array of created TParameter instances with file data
	 */
	DynArray<TParameter*>* load_all_file_parameters(int32_t file_version,
			int32_t current_version,
			CSerializableFile* file, const char* prefix="");

	/** Takes a set of TParameter instances (base) with a certain version and a
	 * set of target parameter infos and recursively maps the base level wise
	 * to the current version using CSGObject::migrate(...).
	 * The base is replaced. After this call, the base version containing
	 * parameters should be of same version/type as the initial target parameter
	 * infos.
	 * Note for this to work, the migrate methods and all the internal parameter
	 * mappings have to match
	 *
	 * @param param_base set of TParameter instances that are mapped to the
	 * provided target parameter infos
	 * @param base_version version of the parameter base
	 * @param target_param_infos set of SGParamInfo instances that specify the
	 * target parameter base */
	void map_parameters(DynArray<TParameter*>* param_base,
			int32_t& base_version,
			DynArray<const SGParamInfo*>* target_param_infos);

	/** set the io object
	 *
	 * @param io io object to use
	 */
	void set_global_io(SGIO* io);

	/** get the io object
	 *
	 * @return io object
	 */
	SGIO* get_global_io();

	/** set the parallel object
	 *
	 * @param parallel parallel object to use
	 */
	void set_global_parallel(Parallel* parallel);

	/** get the parallel object
	 *
	 * @return parallel object
	 */
	Parallel* get_global_parallel();

	/** set the version object
	 *
	 * @param version version object to use
	 */
	void set_global_version(Version* version);

	/** get the version object
	 *
	 * @return version object
	 */
	Version* get_global_version();

	/** @return vector of names of all parameters which are registered for model
	 * selection */
	SGStringList<char> get_modelsel_names();

	/** prints all parameter registered for model selection and their type */
	void print_modsel_params();

	/** Returns description of a given parameter string, if it exists. SG_ERROR
	 * otherwise
	 *
	 * @param param_name name of the parameter
	 * @return description of the parameter
	 */
	char* get_modsel_param_descr(const char* param_name);

	/** Returns index of model selection parameter with provided index
	 *
	 * @param param_name name of model selection parameter
	 * @return index of model selection parameter with provided name,
	 * -1 if there is no such
	 */
	index_t get_modsel_param_index(const char* param_name);

#ifdef TRACE_MEMORY_ALLOCS
	static void list_memory_allocs()
	{
        	::list_memory_allocs();
	}
#endif

protected:
	/** creates a new TParameter instance, which contains migrated data from
	 * the version that is provided. The provided parameter data base is used
	 * for migration, this base is a collection of all parameter data of the
	 * previous version.
	 * Migration is done FROM the data in param_base TO the provided param info
	 * Migration is always one version step.
	 * Method has to be implemented in subclasses, if no match is found, base
	 * method has to be called.
	 *
	 * If there is an element in the param_base which equals the target,
	 * a copy of the element is returned. This represents the case when nothing
	 * has changed and therefore, the migrate method is not overloaded in a
	 * subclass
	 *
	 * @param param_base set of TParameter instances to use for migration
	 * @param target parameter info for the resulting TParameter
	 * @return a new TParameter instance with migrated data from the base of the
	 * type which is specified by the target parameter
	 */
	virtual TParameter* migrate(DynArray<TParameter*>* param_base,
			const SGParamInfo* target);

	/** This method prepares everything for a one-to-one parameter migration.
	 * One to one here means that only ONE element of the parameter base is
	 * needed for the migration (the one with the same name as the target).
	 * Data is allocated for the target (in the type as provided in the target
	 * SGParamInfo), and a corresponding new TParameter instance is written to
	 * replacement. The to_migrate pointer points to the single needed
	 * TParameter instance needed for migration.
	 * If a name change happened, the old name may be specified by old_name.
	 * In addition, the m_delete_data flag of to_migrate is set to true.
	 * So if you want to migrate data, the only thing to do after this call is
	 * converting the data in the m_parameter fields.
	 * If unsure how to use - have a look into an example for this.
	 * (base_migration_type_conversion.cpp for example)
	 *
	 * @param param_base set of TParameter instances to use for migration
	 * @param target parameter info for the resulting TParameter
	 * @param replacement (used as output) here the TParameter instance which is
	 * returned by migration is created into
	 * @param to_migrate the only source that is used for migration
	 * @param old_name with this parameter, a name change may be specified
	 *
	 */
	virtual void one_to_one_migration_prepare(DynArray<TParameter*>* param_base,
			const SGParamInfo* target, TParameter*& replacement,
			TParameter*& to_migrate, char* old_name=NULL);

	/** Can (optionally) be overridden to pre-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
	 *  is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	virtual void load_serializable_pre() throw (ShogunException);

	/** Can (optionally) be overridden to post-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST
	 *  is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	virtual void load_serializable_post() throw (ShogunException);

	/** Can (optionally) be overridden to pre-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
	 *  is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	virtual void save_serializable_pre() throw (ShogunException);

	/** Can (optionally) be overridden to post-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_POST
	 *  is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	virtual void save_serializable_post() throw (ShogunException);

private:
	void set_global_objects();
	void unset_global_objects();
	void init();

	/** Checks in the underlying parameter mapping if this parameter leads to
	 * the empty parameter which means that it was newly added in this version
	 *
	 * @param param_info parameter information of that parameter
	 */
	bool is_param_new(const SGParamInfo param_info) const;

	/** stores the current parameter version in the provided file
	 * @param file file to stort parameter in
	 * @param prefix prefix for the save
	 * @param param_version (optionally) a parameter version different to (this
	 * is mainly for testing, better do not use)
	 * current one may be specified
	 * @return true iff successful
	 */
	bool save_parameter_version(CSerializableFile* file, const char* prefix="",
			int32_t param_version=VERSION_PARAMETER);

	/** loads the parameter version of the provided file.
	 * @return parameter version of file, -1 if there is no such
	 */
	int32_t load_parameter_version(CSerializableFile* file,
			const char* prefix="");

public:
	/** io */
	SGIO* io;

	/** parallel */
	Parallel* parallel;

	/** version */
	Version* version;

	/** parameters */
	Parameter* m_parameters;

	/** model selection parameters */
	Parameter* m_model_selection_parameters;

	/** map for different parameter versions */
	ParameterMap* m_parameter_map;

private:

	EPrimitiveType m_generic;
	bool m_load_pre_called;
	bool m_load_post_called;
	bool m_save_pre_called;
	bool m_save_post_called;

	int32_t m_refcount;

#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T m_ref_lock;
#endif //HAVE_PTHREAD
};
}
#endif // __SGOBJECT_H__
