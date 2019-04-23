/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Weijie Lin, Thoralf Klein, Fernando Iglesias
 */
#include <shogun/lib/Compressor.h>
#include <shogun/io/SGIO.h>
#include <string.h>

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

#ifdef USE_SNAPPY
#include <snappy.h>
#endif

using namespace shogun;

Compressor::Compressor()
	:SGObject(), compression_type(UNCOMPRESSED)
{
	unstable(SOURCE_LOCATION);
}

void Compressor::compress(uint8_t* uncompressed, uint64_t uncompressed_size,
		uint8_t* &compressed, uint64_t &compressed_size, int32_t level)
{
	uint64_t initial_buffer_size=0;

	if (uncompressed_size==0)
	{
		compressed=NULL;
		compressed_size=0;
		return;
	}

	switch (compression_type)
	{
		case UNCOMPRESSED:
			{
				initial_buffer_size=uncompressed_size;
				compressed_size=uncompressed_size;
				compressed=SG_MALLOC(uint8_t, compressed_size);
				sg_memcpy(compressed, uncompressed, uncompressed_size);
				break;
			}
#ifdef USE_LZO
		case LZO:
			{
				if (lzo_init() != LZO_E_OK)
					error("Error initializing LZO Compression");

				lzo_bytep lzo_wrkmem = (lzo_bytep) lzo_malloc(LZO1X_999_MEM_COMPRESS);
				if (!lzo_wrkmem)
					error("Error allocating LZO workmem");

				initial_buffer_size=uncompressed_size +
					uncompressed_size / 16+ 64 + 3;

				compressed_size=initial_buffer_size;
				compressed=SG_MALLOC(uint8_t, initial_buffer_size);

				lzo_uint lzo_size=compressed_size;

				int ret;
				if (level<9)
				{
					ret=lzo1x_1_15_compress(uncompressed, uncompressed_size,
								compressed, &lzo_size, lzo_wrkmem);
				}
				else
				{
					ret=lzo1x_999_compress(uncompressed, uncompressed_size,
								compressed, &lzo_size, lzo_wrkmem);
				}

				compressed_size=lzo_size;
				lzo_free(lzo_wrkmem);

				if (ret!= LZO_E_OK)
					error("Error lzo-compressing data");

				break;
			}
#endif
#ifdef USE_GZIP
		case GZIP:
			{
				initial_buffer_size=1.001*uncompressed_size + 12;
				compressed_size=initial_buffer_size;
				compressed=SG_MALLOC(uint8_t, initial_buffer_size);
				uLongf gz_size=compressed_size;

				if (compress2(compressed, &gz_size, uncompressed,
							uncompressed_size, level) != Z_OK)
				{
					error("Error gzip-compressing data");
				}
				compressed_size=gz_size;
				break;
			}
#endif
#ifdef USE_BZIP2
		case BZIP2:
			{
				bz_stream strm;
				strm.bzalloc=NULL;
				strm.bzfree=NULL;
				strm.opaque=NULL;
				initial_buffer_size=1.01*uncompressed_size + 600;
				compressed_size=initial_buffer_size;
				compressed=SG_MALLOC(uint8_t, initial_buffer_size);
				if (BZ2_bzCompressInit(&strm, level, 0, 0)!=BZ_OK)
					error("Error initializing bzip2 compressor");

				strm.next_in=(char*) uncompressed;
				strm.avail_in=(unsigned int) uncompressed_size;
				strm.next_out=(char*) compressed;
				strm.avail_out=(unsigned int) compressed_size;
				if (BZ2_bzCompress(&strm, BZ_RUN) != BZ_RUN_OK)
					error("Error bzip2-compressing data (BZ_RUN)");

				int ret=0;
				while (true)
				{
					ret=BZ2_bzCompress(&strm, BZ_FINISH);
					if (ret==BZ_FINISH_OK)
						continue;
					if (ret==BZ_STREAM_END)
						break;
					else
						error("Error bzip2-compressing data (BZ_FINISH)");
				}
				BZ2_bzCompressEnd(&strm);
				compressed_size=(((uint64_t) strm.total_out_hi32) << 32) + strm.total_out_lo32;
				break;
			}
#endif
#ifdef USE_LZMA
		case LZMA:
			{
				lzma_stream strm = LZMA_STREAM_INIT;
				initial_buffer_size = lzma_stream_buffer_bound(uncompressed_size);
				compressed_size=initial_buffer_size;
				compressed=SG_MALLOC(uint8_t, initial_buffer_size);
				strm.next_in=uncompressed;
				strm.avail_in=(size_t) uncompressed_size;
				strm.next_out=compressed;
				strm.avail_out=(size_t) compressed_size;

				if (lzma_easy_encoder(&strm, level, LZMA_CHECK_CRC32) != LZMA_OK)
					error("Error initializing lzma compressor");
				if (lzma_code(&strm, LZMA_RUN) != LZMA_OK)
					error("Error lzma-compressing data (LZMA_RUN)");

				lzma_ret ret;
				while (true)
				{
					ret=lzma_code(&strm, LZMA_FINISH);
					if (ret==LZMA_OK)
						continue;
					if (ret==LZMA_STREAM_END)
						break;
					else
						error("Error lzma-compressing data (LZMA_FINISH)");
				}
				lzma_end(&strm);
				compressed_size=strm.total_out;
				break;
			}
#endif
#ifdef USE_SNAPPY
		case SNAPPY:
			{
				compressed=SG_MALLOC(uint8_t, snappy::MaxCompressedLength((size_t) uncompressed_size));
				size_t output_length;
				snappy::RawCompress((char*) uncompressed, size_t(uncompressed_size), (char*) compressed, &output_length);
				compressed_size=(uint64_t) output_length;
				break;
			}
#endif
		default:
			error("Unknown compression type");
	}

	if (compressed)
		compressed = SG_REALLOC(uint8_t, compressed, initial_buffer_size, compressed_size);
}

void Compressor::decompress(uint8_t* compressed, uint64_t compressed_size,
		uint8_t* uncompressed, uint64_t& uncompressed_size)
{
	if (compressed_size==0)
	{
		uncompressed_size=0;
		return;
	}

	switch (compression_type)
	{
		case UNCOMPRESSED:
			{
				ASSERT(uncompressed_size>=compressed_size)
				uncompressed_size=compressed_size;
				sg_memcpy(uncompressed, compressed, uncompressed_size);
				break;
			}
#ifdef USE_LZO
		case LZO:
			{
				if (lzo_init() != LZO_E_OK)
					error("Error initializing LZO Compression");

				lzo_bytep lzo_wrkmem = (lzo_bytep) lzo_malloc(LZO1X_999_MEM_COMPRESS);
				if (!lzo_wrkmem)
					error("Error allocating LZO workmem");

				lzo_uint lzo_size=uncompressed_size;
				if (lzo1x_decompress(compressed, compressed_size, uncompressed,
							&lzo_size, NULL) != LZO_E_OK)
				{
					error("Error uncompressing lzo-data");
				}
				uncompressed_size=lzo_size;

				lzo_free(lzo_wrkmem);
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
					error("Error uncompressing gzip-data");
				}
				uncompressed_size=gz_size;
				break;
			}
#endif
#ifdef USE_BZIP2
		case BZIP2:
			{
				bz_stream strm;
				strm.bzalloc=NULL;
				strm.bzfree=NULL;
				strm.opaque=NULL;
				if (BZ2_bzDecompressInit(&strm, 0, 0)!=BZ_OK)
					error("Error initializing bzip2 decompressor");
				strm.next_in=(char*) compressed;
				strm.avail_in=(unsigned int) compressed_size;
				strm.next_out=(char*) uncompressed;
				strm.avail_out=(unsigned int) uncompressed_size;
				if (BZ2_bzDecompress(&strm) != BZ_STREAM_END || strm.avail_in!=0)
					error("Error uncompressing bzip2-data");
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

				uint64_t memory_limit=lzma_easy_decoder_memusage(9);

				if (lzma_stream_decoder(&strm, memory_limit, 0)!= LZMA_OK)
					error("Error initializing lzma decompressor");
				if (lzma_code(&strm, LZMA_RUN) != LZMA_STREAM_END)
					error("Error decompressing lzma data");
				lzma_end(&strm);
				break;
			}
#endif
#ifdef USE_SNAPPY
		case SNAPPY:
			{
				size_t uncompressed_length;
				if (!snappy::GetUncompressedLength( (char*) compressed,
						(size_t) compressed_size, &uncompressed_length))
					error("Error obtaining uncompressed length");

				ASSERT(uncompressed_length<=uncompressed_size)
				uncompressed_size=uncompressed_length;
				if (!snappy::RawUncompress((char*) compressed,
							(size_t) compressed_size,
							(char*) uncompressed))
					error("Error uncompressing snappy data");

				break;
			}
#endif
		default:
			error("Unknown compression type");
	}
}
