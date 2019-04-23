/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Saurabh Goyal
 */

#ifndef _CSORTULONGSTRING__H__
#define _CSORTULONGSTRING__H__

#include <shogun/lib/config.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/preprocessor/StringPreprocessor.h>
#include <shogun/lib/common.h>

namespace shogun
{
/** @brief Preprocessor SortUlongString, sorts the indivual strings in ascending order.
 *
 * This is useful in conjunction with the CCommUlongStringKernel and will result
 * in the spectrum kernel. For this to work the strings have to be mapped into
 * a binary higher order representation first (cf. obtain_from_*() functions in
 * CStringFeatures)
 */
class SortUlongString : public StringPreprocessor<uint64_t>
{
public:
	/** default constructor */
	SortUlongString();

	/** destructor */
	virtual ~SortUlongString();

	/// cleanup
	virtual void cleanup();
	/// initialize preprocessor from file
	virtual bool load(FILE* f);
	/// save preprocessor init-data to file
	virtual bool save(FILE* f);

	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual uint64_t* apply_to_string(uint64_t* f, int32_t &len);

	/** @return object name */
	virtual const char* get_name() const { return "SortUlongString"; }

	/// return a type of preprocessor
	virtual EPreprocessorType get_type() const { return P_SORTULONGSTRING; }

protected:
	virtual void apply_to_string_list(std::vector<SGVector<uint64_t>>& string_list);
};
}
#endif
