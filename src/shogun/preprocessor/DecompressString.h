/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Berlin Institute of Technology
 */

#ifndef _CDECOMPRESS_STRING__H__
#define _CDECOMPRESS_STRING__H__

#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Compressor.h>
#include <shogun/preprocessor/StringPreprocessor.h>

namespace shogun
{
template <class ST> class CStringFeatures;
class CCompressor;
enum E_COMPRESSION_TYPE;

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
template <class ST> class CDecompressString : public CStringPreprocessor<ST>
{
	public:
		/** default constructor  */
		CDecompressString()
			: CStringPreprocessor<ST>()
		{
			compressor=NULL;
		}

		/** constructor
		 */
		CDecompressString(E_COMPRESSION_TYPE ct)
			: CStringPreprocessor<ST>()
		{
			compressor=new CCompressor(ct);
		}

		/** destructor */
		virtual ~CDecompressString()
		{
			delete compressor;
		}

		/// initialize preprocessor from features
		virtual bool init(CFeatures* f)
		{
			ASSERT(f->get_feature_class()==C_STRING);
			return true;
		}

		/// cleanup
		virtual void cleanup()
		{
		}

		/// initialize preprocessor from file
		bool load(FILE* f)
		{
			SG_SET_LOCALE_C;
			SG_RESET_LOCALE;
			return false;
		}

		/// save preprocessor init-data to file
		bool save(FILE* f)
		{
			SG_SET_LOCALE_C;
			SG_RESET_LOCALE;
			return false;
		}

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual bool apply_to_string_features(CFeatures* f)
		{
			int32_t i;
			int32_t num_vec=((CStringFeatures<ST>*)f)->get_num_vectors();

			for (i=0; i<num_vec; i++)
			{
				int32_t len=0;
				bool free_vec;
				ST* vec=((CStringFeatures<ST>*)f)->
					get_feature_vector(i, len, free_vec);

				ST* decompressed=apply_to_string(vec, len);
				((CStringFeatures<ST>*)f)->
					free_feature_vector(vec, i, free_vec);
				((CStringFeatures<ST>*)f)->
					cleanup_feature_vector(i);
				((CStringFeatures<ST>*)f)->
					set_feature_vector(i, decompressed, len);
			}
			return true;
		}

		/// apply preproc on single feature vector
		virtual ST* apply_to_string(ST* f, int32_t &len)
		{
			uint64_t compressed_size=((int32_t*) f)[0];
			uint64_t uncompressed_size=((int32_t*) f)[1];

			int32_t offs=CMath::ceil(2.0*sizeof(int32_t)/sizeof(ST));
			ASSERT(uint64_t(len)==uint64_t(offs)+compressed_size);

			len=uncompressed_size;
			uncompressed_size*=sizeof(ST);
			ST* vec=SG_MALLOC(ST, len);
			compressor->decompress((uint8_t*) (&f[offs]), compressed_size,
					(uint8_t*) vec, uncompressed_size);

			ASSERT(uncompressed_size==((uint64_t) len)*sizeof(ST));
			return vec;
		}

		/** @return object name */
		virtual inline const char* get_name() const { return "DecompressString"; }

		/// return a type of preprocessor TODO: template specification of get_type
		virtual inline EPreprocessorType get_type() const { return P_DECOMPRESSSTRING; }

	protected:
		/** compressor used to decompress strings */
		CCompressor* compressor;
};
}
#endif
