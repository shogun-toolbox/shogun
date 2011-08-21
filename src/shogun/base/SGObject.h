/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2010 Soeren Sonnenburg
 * Copyright (C) 2008-2010 Fraunhofer Institute FIRST and Max Planck Society
 */

#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/lib/memory.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/Version.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

/** \namespace shogun
 * @brief all of classes and functions are contained in the shogun namespace
 */
namespace shogun
{
class IO;
class Parallel;
class Version;
class Parameter;
class CSerializableFile;

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
	inline int32_t ref()
	{
#ifdef HAVE_PTHREAD
		PTHREAD_LOCK(m_ref_lock);
#endif
		++m_refcount;
		SG_GCDEBUG("ref() refcount %ld obj %s (%p) increased\n", m_refcount, this->get_name(), this);
#ifdef HAVE_PTHREAD
		PTHREAD_UNLOCK(m_ref_lock);
#endif
		return m_refcount;
	}

	/** display reference counter
	 *
	 * @return reference count
	 */
	inline int32_t ref_count()
	{
#ifdef HAVE_PTHREAD
		PTHREAD_LOCK(m_ref_lock);
#endif
		int32_t count=m_refcount;
#ifdef HAVE_PTHREAD
		PTHREAD_UNLOCK(m_ref_lock);
#endif
		SG_GCDEBUG("ref_count(): refcount %d, obj %s (%p)\n", count, this->get_name(), this);
		return count;
	}

	/** decrement reference counter and deallocate object if refcount is zero
	 * before or after decrementing it
	 *
	 * @return reference count
	 */
	inline int32_t unref()
	{
#ifdef HAVE_PTHREAD
		PTHREAD_LOCK(m_ref_lock);
#endif
		if (m_refcount==0 || --m_refcount==0)
		{
			SG_GCDEBUG("unref() refcount %ld, obj %s (%p) destroying\n", m_refcount, this->get_name(), this);
#ifdef HAVE_PTHREAD
			PTHREAD_UNLOCK(m_ref_lock);
#endif
			delete this;
			return 0;
		}
		else
		{
			SG_GCDEBUG("unref() refcount %ld obj %s (%p) decreased\n", m_refcount, this->get_name(), this);
#ifdef HAVE_PTHREAD
			PTHREAD_UNLOCK(m_ref_lock);
#endif
			return m_refcount;
		}
	}
#endif

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
	 *  @param file where to save the object; will be closed during
	 *              returning if PREFIX is an empty string.
	 *  @param prefix prefix for members
	 *
	 *  @return TRUE if done, otherwise FALSE
	 */
	virtual bool save_serializable(CSerializableFile* file,
								   const char* prefix="");

	/** Load this object from file.  If it will fail (returning FALSE)
	 *  then this object will contain inconsistent data and should not
	 *  be used!
	 *
	 *  @param file where to load from
	 *  @param prefix prefix for members
	 *
	 *  @return TRUE if done, otherwise FALSE
	 */
	virtual bool load_serializable(CSerializableFile* file,
								   const char* prefix="");

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
	SGVector<char*> get_modelsel_names();

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

	/** stores the current parameter version in the provided file
	 *  @return true iff successful
	 */
	bool save_parameter_version(CSerializableFile* file, const char* prefix="");

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

private:
	EPrimitiveType m_generic;
	bool m_load_pre_called;
	bool m_load_post_called;
	bool m_save_pre_called;
	bool m_save_post_called;

	int32_t m_refcount;

#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T m_ref_lock;
#endif
};
}
#endif // __SGOBJECT_H__
