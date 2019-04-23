/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Saurabh Goyal
 */

#ifndef _CSORTWORDSTRING__H__
#define _CSORTWORDSTRING__H__

#include <shogun/lib/config.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/preprocessor/StringPreprocessor.h>
#include <shogun/lib/common.h>


namespace shogun
{
/** @brief Preprocessor SortWordString, sorts the indivual strings in ascending order.
 *
 * This is useful in conjunction with the CCommWordStringKernel and will result
 * in the spectrum kernel. For this to work the strings have to be mapped into
 * a binary higher order representation first (cf. obtain_from_*() functions in
 * CStringFeatures)
 */
class SortWordString : public StringPreprocessor<uint16_t>
{
public:
	/** default constructor */
	SortWordString();

	/** destructor */
	virtual ~SortWordString();

	/// cleanup
	virtual void cleanup();
	/// initialize preprocessor from file
	virtual bool load(FILE* f);
	/// save preprocessor init-data to file
	virtual bool save(FILE* f);

	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual uint16_t* apply_to_string(uint16_t* f, int32_t &len);

	/** @return object name */
	virtual const char* get_name() const { return "SortWordString"; }

	/// return a type of preprocessor
	virtual EPreprocessorType get_type() const { return P_SORTWORDSTRING; }

protected:
	virtual void apply_to_string_list(SGStringList<uint16_t> string_list);
};
}
#endif
