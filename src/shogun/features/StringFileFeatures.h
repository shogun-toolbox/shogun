/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang, Viktor Gal,
 *          Evan Shelhamer, Bjoern Esser
 */

#ifndef _CSTRINGFILEFEATURES__H__
#define _CSTRINGFILEFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/features/Alphabet.h>
#include <shogun/io/MemoryMappedFile.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
class Alphabet;
template <class T> class MemoryMappedFile;

/** @brief File based string features.
 *
 * StringFeatures that are file based. Underneath memory mapped files are used.
 * Derived from CStringFeatures thus transparently enabling all of the
 * StringFeature functionality.
 *
 * Supported file format contains one string per line, lines of variable length
 * are supported and must be separated by '\n'.
 */
template <class ST> class StringFileFeatures : public StringFeatures<ST>
{
	public:

	/** default constructor
	 *
	 */
	StringFileFeatures();

	/** constructor
	 *
	 * @param fname filename of the file containing line based features
	 * @param alpha alphabet (type) to use for string features
	 */
	StringFileFeatures(const char* fname, EAlphabet alpha);

	/** default destructor
	 *
	 */
	virtual ~StringFileFeatures();

	/** Returns the name of the SGSerializable instance.
	 *
	 * @return name of the SGSerializable
	 */
	virtual const char* get_name() const { return "StringFileFeatures"; }

	protected:
	/** get next line from file
	 *
	 * The returned line may be modfied in case the file was opened
	 * read/write. It is otherwise read-only.
	 *
	 * @param len length of line (returned via reference)
	 * @param offs offset to be passed for reading next line, should be 0
	 *			initially (returned via reference)
	 * @param line_nr used to indicate errors (returned as reference should be 0
	 *			initially)
	 * @param file_length total length of the file (for error checking)
	 *
	 * @return line (NOT ZERO TERMINATED)
	 */
	ST* get_line(uint64_t& len, uint64_t& offs, int32_t& line_nr, uint64_t file_length);

	/** cleanup string features */
	virtual void cleanup();

    /** cleanup a single feature vector */
    virtual void cleanup_feature_vector(int32_t num);

	/** obtain meta information from file
	 *
	 * i.e., determine number of strings and their lengths
	 */
	void fetch_meta_info_from_file(int32_t granularity=1048576);

	protected:
	/** memory mapped file*/
	std::shared_ptr<MemoryMappedFile<ST>> file;
};
}
#endif // _CSTRINGFILEFEATURES__H__
