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

#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/common.h"
#include "lib/Compressor.h"
#include "preproc/StringPreProc.h"

namespace shogun
{
template <class ST> class CStringFeatures;
class CCompressor;
enum E_COMPRESSION_TYPE;

template <class ST> class CDecompressString : public CStringPreProc<ST>
{
	public:
		/** constructor
		 */
		CDecompressString(E_COMPRESSION_TYPE ct)
			: CStringPreProc<ST>("DecompressString", "DECS")
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
			return false;
		}

		/// save preprocessor init-data to file
		bool save(FILE* f)
		{
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
			ST* vec=new ST[len];
			compressor->decompress((uint8_t*) (&f[offs]), compressed_size,
					(uint8_t*) vec, uncompressed_size);

			ASSERT(uncompressed_size==((uint64_t) len)*sizeof(ST));
			return vec;
		}

	protected:
		CCompressor* compressor;
};
}
#endif
