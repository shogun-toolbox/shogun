/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Berlin Institute of Technology
 */
#ifndef __COMPRESSOR_H__
#define __COMPRESSOR_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/io.h>

#ifdef USE_LZO
#include "lzo/lzoconf.h"
#include <lzo1x.h>
#endif

#ifdef USE_GZIP
#include <zlib.h>
#endif

#ifdef USE_BZIP2
#include <bzlib.h>
#endif

#ifdef USE_LZMA
#include <lzma.h>
#endif

namespace shogun
{
	enum E_COMPRESSION_TYPE
	{
		NONE,
		LZO,
		GZIP,
		BZIP2,
		LZMA
	}

	/** Compression library for compressing and decompressing buffers using 
	 * one of the standard compression algorithms, LZO, GZIP, BZIP2 or LZMA.
	 *
	 * The general recommendation is to use LZO whenever lightweight compression
	 * is sufficient but high i/o throughputs are needed (at 1/2 the speed of memcpy).
	 *
	 * If size is all that matters use LZMA (which especially when compressing
	 * can be very slow though).
	 *
	 * Note that besides lzo compression, this library is thread safe.
	 *
	 */
	class CCompressor : public CSGObject
	{
	public:
		CCompressor(E_COMPRESSION_TYPE ct) : CSGObject(), compression_type(ct)
		{
			init();
		}

		virtual ~CCompressor()
		{
			cleanup();
		}

		void compress(uint8_t* uncompressed, uint64_t uncompressed,
				uint8_t* &compressed, uint64_t &compressed_size, int32_t level=1)
		{
			switch (compression_type)
			{
				uint64_t initial_buffer_size=0;
#ifdef USE_LZO
			case LZO:
				ASSERT(level==1);
				initial_buffer_size=uncompressed_size + uncompressed_size / 16 + 64 + 3;
				compressed_size=initial_buffer_size;
				compressed=new uint8_t[initial_buffer_size];

				if (lzo1x_1_compress(data, data_size,
							compressed_data, compressed_data_size, lzo_wrkmem) != LZO_E_OK)
				{
					SG_ERROR("Error lzo-compressing data\n");
				}
				break;
#endif
#ifdef USE_GZIP
			case GZIP:
				initial_buffer_size=1.001*uncompressed_size + 12;
				compressed_size=initial_buffer_size;
				compressed=new uint8_t[initial_buffer_size];
				if (compress2(compressed, compressed_size, uncompressed, uncompressed_size, level) != Z_OK)
				{
					SG_ERROR("Error gzip-compressing data\n");
				}
				break;
#endif
#ifdef USE_BZIP2
			case BZIP2:
				bz_stream strm;
				initial_buffer_size=1.001*uncompressed_size + 12;
				compressed_size=initial_buffer_size;
				compressed=new uint8_t[initial_buffer_size];
				if (BZ2_bzCompressInit(&strm, level, 0, 0)!=BZ_OK)
					SG_ERROR("Error initializing bzip2 compressor\n");
				strm.next_in=uncompressed;
				strm.avail_in=(unsigned int) uncompressed_size;
				strm.next_out=compressed;
				strm.avail_out=(unsigned int) compressed_size;
				if (BZ2_bzCompress(&strm, BZ_RUN) != BZ_RUN_OK)
					SG_ERROR("Error bzip2-compressing data\n");
				if (BZ2_bzCompress(&strm, BZ_FINISH) != BZ_FINISH_OK)
					SG_ERROR("Error bzip2-compressing data\n");
				BZ2_bzCompressEnd(&strm);

				break
#endif
#ifdef USE_LZMA
			case LZMA:
					break;
#endif
				if (compressed_data)
				{
					CMath::resize(compressed_data,
							initial_buffer_size, compressed_data_size);
				}
			default:
				break;
			}

		}

		void decompress(uint8_t* compressed, uint64_t compressed_size,
				uint8_t* decompressed, uint64_t& decompressed_size)
		{
			switch (compression_type)
			{
#ifdef USE_LZO
			case LZO:
				if (lzo1x_decompress(compressed, compressed_size, decompressed,
							decompressed_size, NULL) != LZO_E_OK)
				{
					SG_ERROR("Error uncompressing lzo-data\n");
				}
				break;
#endif
#ifdef USE_GZIP
			case GZIP:
				if (uncompress(uncompressed, uncompressed_size, compressed, compressed_size) != Z_OK)
				{
					SG_ERROR("Error uncompressing gzip-data\n");
				}
				break;
#endif
#ifdef USE_BZIP2
			case BZIP2:
				bz_stream strm;
				if (BZ2_bzDeCompressInit(&strm, 0, 0)!=BZ_OK)
					SG_ERROR("Error initializing bzip2 decompressor\n");
				strm.next_in=compressed;
				strm.avail_in=(unsigned int) compressed_size;
				strm.next_out=uncompressed;
				strm.avail_out=(unsigned int) uncompressed_size;
				if (BZ2_bzDeCompress(&strm) != BZ_STREAM_END)
					SG_ERROR("Error uncompressing bzip2-data\n");
				BZ2_bzDeCompressEnd(&strm);
				break;
#endif
#ifdef USE_LZMA
			case LZMA:
					break;
#endif
			default:
				break;
			}

		}

	protected:
		void init()
		{
			switch (compression_type)
			{
#ifdef USE_LZO
			case LZO:
				if (lzo_init() != LZO_E_OK)
					SG_ERROR("Error initializing LZO Compression\n");
				lzo_wrkmem = (lzo_bytep) lzo_malloc(LZO1X_1_MEM_COMPRESS);
				if (!lzo_wrkmem)
					SG_ERROR("Error allocating LZO workmem\n");
				
				break;
#endif
			default:
				break;
			}
		}

		void cleanup()
		{
			switch (compression_type)
			{
#ifdef USE_LZO
			case LZO:
				lzo_free(wrkmem);
				break;
#endif
			default:
				break;
			}
		}

	protected:
			E_COMPRESSION_TYPE compression_type;
#ifdef USE_LZO
			lzo_bytep lzo_wrkmem;
#endif

	};
}
#endif //__COMPRESSOR_H__
