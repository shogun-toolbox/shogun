/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Jacob Walker, Thoralf Klein, 
 *          Bjoern Esser
 */
#ifndef __SGSTRING_H__
#define __SGSTRING_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>

namespace shogun
{

template<class T> class SGVector;
class File;

/** @brief shogun string */
template<class T> class SGString
{
public:
	/** default constructor */
	SGString();

	/** constructor for setting params */
	SGString(T* s, index_t l, bool free_s=false);

	/** constructor for setting params from a SGVector*/
	SGString(SGVector<T> v);

	/** constructor to create new string in memory */
	SGString(index_t len, bool free_s=false);

	/** copy constructor */
	SGString(const SGString &orig);

	/** @return true iff pointer and size are equal */
	bool operator==(const SGString & other) const;

	/** @return true iff content is equal */
	bool equals(const SGString& other) const;

	/** Clone string */
	SGString<T> clone() const;

	/** free string */
	void free_string();

	/** destroy string */
	void destroy_string();

	/**
	 * get the string (no copying is done here)
	 *
	 * @return the refcount increased string
	 */
	inline SGString<T> get()
	{
		return *this;
	}

	/** load string from file
	 *
	 * @param loader File object via which to load data
	 */
	void load(File* loader);

	/** save string to file
	 *
	 * @param saver File object via which to save data
	 */
	void save(File* saver);

public:
	/** string  */
	T* string;
	/** length of string  */
	index_t slen;
	/** whether string needs to be freed */
	bool do_free;
};
}
#endif // __SGSTRING_H__
