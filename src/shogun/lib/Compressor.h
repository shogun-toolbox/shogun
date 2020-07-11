/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein, Evan Shelhamer, 
 *          Yuyu Zhang
 */
#ifndef __COMPRESSOR_H__
#define __COMPRESSOR_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
	/** compression type */
	enum E_COMPRESSION_TYPE
	{
		UNCOMPRESSED,
		LZO,
		GZIP,
		BZIP2,
		LZMA,
		SNAPPY
	};


	/** @brief Compression library for compressing and decompressing buffers using
	 * one of the standard compression algorithms:
	 *
	 *   LZO, GZIP, BZIP2 or LZMA, SNAPPY.
	 *
	 * The general recommendation is to use SNAPPY or LZO whenever lightweight compression
	 * is sufficient but high i/o throughputs are needed (at 1/2 the speed of memcpy).
	 *
	 * If size is all that matters use LZMA (which especially when compressing
	 * can be very slow though).
	 *
	 * Note that besides lzo compression, this library is thread safe.
	 *
	 */
	class Compressor : public SGObject
	{
	public:
		/** default constructor  */
		Compressor();

		/** default constructor
		 *
		 * @param ct compression to use: one of UNCOMPRESSED, LZO, GZIP, BZIP2 or LZMA
		 */
		Compressor(E_COMPRESSION_TYPE ct) : SGObject(), compression_type(ct)
		{
		}

		/** default destructor */
		~Compressor() override
		{
		}

		/** compress data
		 *
		 * compresses the buffer uncompressed using the selected compression
		 * algorithm and returns compressed data and its size
		 *
		 * @param uncompressed - uncompressed data to be compressed
		 * @param uncompressed_size - size of the uncompressed data
		 * @param compressed - pointer to hold compressed data (returned)
		 * @param compressed_size - size of compressed data (returned)
		 * @param level - compression level between 1 and 9
		 */
		void compress(uint8_t* uncompressed, uint64_t uncompressed_size,
				uint8_t* &compressed, uint64_t &compressed_size, int32_t level=1);

		/** decompress data
		 *
		 * Decompresses the buffer using the selected compression
		 * algorithm to the memory block specified in uncompressed.
		 * Note: Compressed and uncompressed size must be known prior to
		 * calling this function.
		 *
		 * @param compressed - pointer to compressed data
		 * @param compressed_size - size of compressed data
		 * @param uncompressed - pointer to buffer to hold uncompressed data
		 * @param uncompressed_size - size of the uncompressed data
		 */
		void decompress(uint8_t* compressed, uint64_t compressed_size,
				uint8_t* uncompressed, uint64_t& uncompressed_size);

		/** @return object name */
		const char* get_name() const override { return "Compressor"; }

	protected:
		/** compressor type */
		E_COMPRESSION_TYPE compression_type;
	};
}
#endif //__COMPRESSOR_H__
