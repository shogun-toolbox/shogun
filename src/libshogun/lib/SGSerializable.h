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

namespace shogun
{
class CParameter;
class CSerializableFile;
class CIO;

class CSGSerializable
{
protected:
	CParameter* m_parameters;

public:
	/** default constructor  */
	explicit CSGSerializable(void);

	/** default destructor  */
	virtual ~CSGSerializable(void);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 * the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	virtual const char* get_name(void) const = 0;

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
