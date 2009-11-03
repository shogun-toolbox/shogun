/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Berlin Institute of Technology
 */
#include "lib/Compressor.h"
#include "lib/Mathematics.h"

#ifdef USE_LZO
#include <lzo/lzoconf.h>
#include <lzo/lzoutil.h>
#include <lzo/lzo1x.h>
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

using namespace shogun;

void CCompressor::compress(uint8_t* uncompressed, uint64_t uncompressed_size,
		uint8_t* &compressed, uint64_t &compressed_size, int32_t level)
{
	uint64_t initial_buffer_size=0;

	switch (compression_type)
	{
		case NONE:
			{
				compressed_size=uncompressed_size;
				compressed=new uint8_t[compressed_size];
				memcpy(compressed, uncompressed, uncompressed_size);
				break;
			}
#ifdef USE_LZO
		case LZO:
			{
				ASSERT(level==1);
				initial_buffer_size=uncompressed_size +
					uncompressed_size / 16+ 64 + 3;

				compressed_size=initial_buffer_size;
				compressed=new uint8_t[initial_buffer_size];

				lzo_uint lzo_size=compressed_size;

				if (lzo1x_1_compress(uncompressed, uncompressed_size,
							compressed, &lzo_size, lzo_wrkmem) != LZO_E_OK)
				{
					SG_ERROR("Error lzo-compressing data\n");
				}
				compressed_size=lzo_size;
				break;
			}
#endif
#ifdef USE_GZIP
		case GZIP:
			{
				initial_buffer_size=1.001*uncompressed_size + 12;
				compressed_size=initial_buffer_size;
				compressed=new uint8_t[initial_buffer_size];
				uLongf gz_size=compressed_size;

				if (compress2(compressed, &gz_size, uncompressed,
							uncompressed_size, level) != Z_OK)
				{
					SG_ERROR("Error gzip-compressing data\n");
				}
				compressed_size=gz_size;
				break;
			}
#endif
#ifdef USE_BZIP2
		case BZIP2:
			{
				bz_stream strm;
				initial_buffer_size=1.001*uncompressed_size + 12;
				compressed_size=initial_buffer_size;
				compressed=new uint8_t[initial_buffer_size];
				if (BZ2_bzCompressInit(&strm, level, 0, 0)!=BZ_OK)
					SG_ERROR("Error initializing bzip2 compressor\n");

				strm.next_in=(char*) uncompressed;
				strm.avail_in=(unsigned int) uncompressed_size;
				strm.next_out=(char*) compressed;
				strm.avail_out=(unsigned int) compressed_size;
				if (BZ2_bzCompress(&strm, BZ_RUN) != BZ_RUN_OK)
					SG_ERROR("Error bzip2-compressing data\n");
				if (BZ2_bzCompress(&strm, BZ_FINISH) != BZ_FINISH_OK)
					SG_ERROR("Error bzip2-compressing data\n");
				BZ2_bzCompressEnd(&strm);

				break;
			}
#endif
#ifdef USE_LZMA
		case LZMA:
			{
				lzma_stream strm = LZMA_STREAM_INIT;
				initial_buffer_size = lzma_stream_buffer_bound(uncompressed_size);
				compressed_size=initial_buffer_size;
				compressed=new uint8_t[initial_buffer_size];
				strm.next_in=uncompressed;
				strm.avail_in=(size_t) uncompressed_size;
				strm.next_out=compressed;
				strm.avail_out=(size_t) compressed_size;

				if (lzma_easy_encoder(&strm, level, LZMA_CHECK_CRC32) != LZMA_OK)
					SG_ERROR("Error initializing lzma compressor\n");
				if (lzma_code(&strm, LZMA_RUN) != LZMA_OK)
					SG_ERROR("Error lzma-compressing data\n");
				if (lzma_code(&strm, LZMA_FINISH) != LZMA_OK)
					SG_ERROR("Error lzma-compressing data\n");
				lzma_end(&strm);
				break;
			}
#endif
			if (compressed)
				CMath::resize(compressed, initial_buffer_size, compressed_size);
		default:
			break;
	}
}

void CCompressor::decompress(uint8_t* compressed, uint64_t compressed_size,
		uint8_t* uncompressed, uint64_t& uncompressed_size)
{
	switch (compression_type)
	{
		case NONE:
			{
				ASSERT(uncompressed_size>compressed_size);
				uncompressed_size=compressed_size;
				memcpy(uncompressed, compressed, uncompressed_size);
				break;
			}
#ifdef USE_LZO
		case LZO:
			{
				lzo_uint lzo_size=uncompressed_size;
				if (lzo1x_decompress(compressed, compressed_size, uncompressed,
							&lzo_size, NULL) != LZO_E_OK)
				{
					SG_ERROR("Error uncompressing lzo-data\n");
				}
				uncompressed_size=lzo_size;
				break;
			}
#endif
#ifdef USE_GZIP
		case GZIP:
			{
				uLongf gz_size=uncompressed_size;
				if (uncompress(uncompressed, &gz_size, compressed,
							compressed_size) != Z_OK)
				{
					SG_ERROR("Error uncompressing gzip-data\n");
				}
				uncompressed_size=gz_size;
				break;
			}
#endif
#ifdef USE_BZIP2
		case BZIP2:
			{
				bz_stream strm;
				if (BZ2_bzDecompressInit(&strm, 0, 0)!=BZ_OK)
					SG_ERROR("Error initializing bzip2 decompressor\n");
				strm.next_in=(char*) compressed;
				strm.avail_in=(unsigned int) compressed_size;
				strm.next_out=(char*) uncompressed;
				strm.avail_out=(unsigned int) uncompressed_size;
				if (BZ2_bzDecompress(&strm) != BZ_STREAM_END)
					SG_ERROR("Error uncompressing bzip2-data\n");
				BZ2_bzDecompressEnd(&strm);
				break;
			}
#endif
#ifdef USE_LZMA
		case LZMA:
			{
				lzma_stream strm = LZMA_STREAM_INIT;
				strm.next_in=compressed;
				strm.avail_in=(size_t) compressed_size;
				strm.next_out=uncompressed;
				strm.avail_out=(size_t) uncompressed_size;

				if (lzma_stream_decoder(&strm, uncompressed_size, 0)!= LZMA_OK)
					SG_ERROR("Error initializing lzma decompressor\n");
				if (lzma_code(&strm, LZMA_RUN) != LZMA_OK)
					SG_ERROR("Error decompressing lzma data\n");
				if (lzma_code(&strm, LZMA_FINISH) != LZMA_OK)
					SG_ERROR("Error decompressing lzma data\n");
				lzma_end(&strm);
				break;
			}
#endif
		default:
			break;
	}
}

void CCompressor::init()
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

void CCompressor::cleanup()
{
	switch (compression_type)
	{
#ifdef USE_LZO
	case LZO:
		lzo_free(lzo_wrkmem);
		break;
#endif
	default:
		break;
	}
}
