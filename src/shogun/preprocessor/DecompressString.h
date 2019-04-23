/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _CDECOMPRESS_STRING__H__
#define _CDECOMPRESS_STRING__H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Compressor.h>
#include <shogun/preprocessor/StringPreprocessor.h>

namespace shogun
{
template <class ST> class CStringFeatures;
class Compressor;

/** @brief Preprocessor that decompresses compressed strings.
 *
 * Each string in CStringFeatures might be stored compressed in memory.
 * This preprocessor decompresses these strings on the fly. This may be
 * especially usefull for long strings and when datasets become too large
 * to fit in memoryin uncompressed form but still when they are compressed.
 *
 * Then avoiding expensive disk i/o strings are on-the-fly decompressed.
 *
 */
template <class ST> class DecompressString : public StringPreprocessor<ST>
{
	public:
		/** default constructor  */
		DecompressString();

		/** constructor
		 */
		DecompressString(E_COMPRESSION_TYPE ct);

		/** destructor */
		virtual ~DecompressString();

		/// cleanup
		virtual void cleanup();

		/// initialize preprocessor from file
		bool load(FILE* f);

		/// save preprocessor init-data to file
		bool save(FILE* f);

		/// apply preproc on single feature vector
		virtual ST* apply_to_string(ST* f, int32_t &len);

		/** @return object name */
		virtual const char* get_name() const { return "DecompressString"; }

		/// return a type of preprocessor TODO: template specification of get_type
		virtual EPreprocessorType get_type() const;

	protected:
		virtual void apply_to_string_list(SGStringList<ST> string_list);

		/** compressor used to decompress strings */
		std::shared_ptr<Compressor> compressor;
};
}
#endif
