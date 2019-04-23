/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein, Yuyu Zhang, 
 *          Bjoern Esser
 */
#ifndef __SGSTRINGLIST_H__
#define __SGSTRINGLIST_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/SGString.h>

namespace shogun
{
class File;
template <class T> class SGString;

/** @brief template class SGStringList */
template <class T> class SGStringList : public SGReferencedData
{
public:
	/** default constructor */
	SGStringList();

	/** constructor for setting params */
	SGStringList(SGString<T>* s, index_t num_s, index_t max_length,
			bool ref_counting=true);

	/** constructor to create new string list in memory */
	SGStringList(index_t num_s, index_t max_length, bool ref_counting=true);

	/** copy constructor */
	SGStringList(const SGStringList &orig);

	/** destructor */
	virtual ~SGStringList();

	/**
	 * get the string list (no copying is done here)
	 *
	 * @return the refcount increased string list
	 */
	inline SGStringList<T> get()
	{
		return *this;
	}

	/** load strings from file
	 *
	 * @param loader File object via which to load data
	 */
	void load(File* loader);

	/** save strings to file
	 *
	 * @param saver File object via which to save data
	 */
	void save(File* saver);

	/** clone string list
	 *
	 * @return a deep copy of current string list
	 */
	SGStringList<T> clone() const;


	/** Equals method
	 * @param other SGStringList to compare with
	 * @return false iff the number of strings, the maximum string length or
	 * any of the string items are different, true otherwise
	 */
	bool equals(const SGStringList<T>& other) const;

protected:

	/** copy data */
	virtual void copy_data(const SGReferencedData &orig);

	/** init data */
	virtual void init_data();

	/** free data */
	void free_data();

public:
	/** number of strings */
	index_t num_strings;

	/** length of longest string */
	index_t max_string_length;

	/** this contains the array of features */
	SGString<T>* strings;
};
}
#endif // __SGSTRINGLIST_H__
