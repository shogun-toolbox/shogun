/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __SGSERIALIZABLE_H__
#define __SGSERIALIZABLE_H__

#include "lib/DataType.h"
#include "lib/ShogunException.h"

namespace shogun
{
class Parameter;
class CSerializableFile;
class IO;

/* define reference counter macros
 */
#ifdef USE_REFERENCE_COUNTING
#  define SG_REF(x) { if (x) (x)->ref(); }
#  define SG_UNREF(x) { if (x) { if ((x)->unref()==0) (x)=0; } }
#else /* USE_REFERENCE_COUNTING  */
#  define SG_REF(x)
#  define SG_UNREF(x)
#endif /* USE_REFERENCE_COUNTING  */

class CSGSerializable
{
	EPrimitveType m_generic;
	bool load_pre_called, load_post_called;

#ifdef USE_REFERENCE_COUNTING
	int32_t m_refcount;
#endif /* USE_REFERENCE_COUNTING  */

protected:
	Parameter* m_parameters;

	/** Can (optionally) be overridden to pre-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
	 *  is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	virtual void load_serializable_pre(void) throw (ShogunException);

	/** Can (optionally) be overridden to post-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST
	 *  is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	virtual void load_serializable_post(void) throw (ShogunException);

public:
	/** default constructor  */
	explicit CSGSerializable(void);

	/** default destructor  */
	virtual ~CSGSerializable(void);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name(void) const = 0;

	/** If the SGSerializable is a class template then TRUE will be
	 *  returned and GENERIC is set to the type of the generic.
	 *
	 *  @param generic set to the type of the generic if returning
	 *                 TRUE
	 *
	 *  @return TRUE if a class template.
	 */
	virtual bool is_generic(EPrimitveType* generic) const;

	/** set generic type to T
	 */
	template<class T> void set_generic(void);

	/** unset generic type
	 */
	void unset_generic(void);

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

/* **************************************************************** */
#ifdef USE_REFERENCE_COUNTING

	/** increase reference counter
	 *
	 * @return reference count
	 */
	int32_t ref(void);

	/** display reference counter
	 *
	 *  @return reference count
	 */
	int32_t ref_count(void) const;

	/** decrement reference counter and deallocate object if refcount
	 *  is zero before or after decrementing it
	 *
	 *  @return reference count
	 */
	int32_t unref(void);

#endif /* USE_REFERENCE_COUNTING  */
/* **************************************************************** */

};
}

#endif /* __SGSERIALIZABLE_H__  */
