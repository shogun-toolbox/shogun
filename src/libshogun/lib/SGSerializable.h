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

namespace shogun
{
class Parameter;
class CSerializableFile;
class IO;

class CSGSerializable
{
	EPrimitveType m_generic;

protected:
	Parameter* m_parameters;

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
	 *  @param file where to save the object; will be closed during
	 *              returning if PREFIX is an empty string.
	 *  @param prefix prefix for members
	 *
	 *  @return TRUE if done, otherwise FALSE
	 */
	virtual bool load_serializable(CSerializableFile* file,
								   const char* prefix="");
};
}

#endif /* __SGSERIALIZABLE_H__  */
