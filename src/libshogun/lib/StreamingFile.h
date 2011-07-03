/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_FILE_H__
#define __STREAMING_FILE_H__

#include "lib/config.h"
#include "base/DynArray.h"
#include "lib/common.h"
#include "lib/File.h"
#include "lib/io.h"
#include "lib/DataType.h"
#include <ctype.h>

namespace shogun
{
/** @brief A Streaming File access class.
 *
 * - Vectors are read as one vector per line
 * - NOT YET IMPLEMENTED:
 * - Matrices are written out as one column per line
 * - Sparse Matrices are written out as one column per line (libsvm/svmlight
 *	 style format)
 * - Strings are written out as one string per line
 *
 */
	class CStreamingFile: public CSGObject
	{
	public:
		/** default constructor	 */
		CStreamingFile();

		/** constructor
		 *
		 * @param f already opened file
		 * @param name variable name (e.g. "x" or "/path/to/x")
		 */
		CStreamingFile(FILE* f, const char* name=NULL);

		/** constructor
		 *
		 * @param fname filename to open
		 * @param rw mode, 'r' or 'w'
		 */
		CStreamingFile(char* fname, char rw='r');

		/** default destructor */
		virtual ~CStreamingFile();

		/** 
		 * Closes the file
		 */
		void close()
		{
			SG_FREE(filename);
			if (file)
				fclose(file);
			filename=NULL;
			file=NULL;
		}


#define GET_VECTOR_DECL(sg_type)					\
		virtual void get_vector					\
			(sg_type*& vector, int32_t& len);		\
									\
		virtual void get_vector_and_label			\
			(sg_type*& vector, int32_t& len, float64_t& label); \
									\
		virtual void get_string					\
			(sg_type*& vector, int32_t& len);		\
									\
		virtual void get_string_and_label			\
			(sg_type*& vector, int32_t& len, float64_t& label); \
									\
		virtual void get_sparse_vector				\
			(SGSparseVectorEntry<sg_type>*& vector, int32_t& len); \
									\
		virtual void get_sparse_vector_and_label		\
			(SGSparseVectorEntry<sg_type>*& vector, int32_t& len, float64_t& label);

		GET_VECTOR_DECL(bool)
		GET_VECTOR_DECL(uint8_t)
		GET_VECTOR_DECL(char)
		GET_VECTOR_DECL(int32_t)
		GET_VECTOR_DECL(float32_t)
		GET_VECTOR_DECL(float64_t)
		GET_VECTOR_DECL(int16_t)
		GET_VECTOR_DECL(uint16_t)
		GET_VECTOR_DECL(int8_t)
		GET_VECTOR_DECL(uint32_t)
		GET_VECTOR_DECL(int64_t)
		GET_VECTOR_DECL(uint64_t)
		GET_VECTOR_DECL(floatmax_t)
#undef GET_VECTOR_DECL

		/** @return object name */
		inline virtual const char* get_name() const { return "StreamingFile"; }

	protected:
		/// File object
		FILE* file;
		/// Task
		char task;
		/// Name of the handled file
		char* filename;
		
	};
}
#endif //__STREAMING_FILE_H__
