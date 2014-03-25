/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SIMPLEFILE_H__
#define __SIMPLEFILE_H__

#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/SGObject.h>

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

namespace shogun
{
/** @brief Template class SimpleFile to read and write from files.
 *
 * Currently only simple reading and writing of blocks is supported.
 */
template <class T> class CSimpleFile : public CSGObject
{
	public:
		/** default constructor  */
		CSimpleFile() :CSGObject(), line_buffer_size(1024*1024), line_buffer(NULL)
		{
			SG_UNSTABLE("CSimpleFile::CSimpleFile()", "\n")

			file=NULL;
			filename=get_strdup("");
			status = false;

			set_generic<T>();
		}

		/** constructor
		 * rw is either r for read and w for write
		 *
		 * @param fname filename
		 * @param f file descriptor
		 */
		CSimpleFile(char* fname, FILE* f)
		: CSGObject(), line_buffer_size(1024*1024), line_buffer(NULL)
		{
			file=f;
			filename=get_strdup(fname);
			status = (file!=NULL && filename!=NULL);
		}

		virtual ~CSimpleFile()
		{
			SG_FREE(filename);
			free_line_buffer();
		}

		/** load
		 *
		 * @param target load target
		 * @param num number of read elements
		 * @return loaded target or NULL if unsuccessful
		 */
		T* load(T* target, int64_t& num)
		{
			if (status)
			{
				status=false;

				if (num==0)
				{
					bool seek_status=true;
					int64_t cur_pos=ftell(file);

					if (cur_pos!=-1)
					{
						if (!fseek(file, 0, SEEK_END))
						{
							if ((num=(int64_t) ftell(file)) != -1)
							{
								SG_INFO("file of size %ld bytes == %ld entries detected\n", num,num/sizeof(T))
								num/=sizeof(T);
							}
							else
								seek_status=false;
						}
						else
							seek_status=false;
					}

					if ((fseek(file,cur_pos, SEEK_SET)) == -1)
						seek_status=false;

					if (!seek_status)
					{
						SG_ERROR("filesize autodetection failed\n")
						num=0;
						return NULL;
					}
				}

				if (num>0)
				{
					if (!target)
						target=SG_MALLOC(T, num);

					if (target)
					{
						size_t num_read=fread((void*) target, sizeof(T), num, file);
						status=((int64_t) num_read == num);

						if (!status)
							SG_ERROR("only %ld of %ld entries read. io error\n", (int64_t) num_read, num)
					}
					else
						SG_ERROR("failed to allocate memory while trying to read %ld entries from file \"s\"\n", (int64_t) num, filename)
				}
				return target;
			}
			else
			{
				num=-1;
				return NULL;
			}
		}

		/** save
		 *
		 * @param target target to save to
		 * @param num number of elements to write
		 * @return if saving was successful
		 */
		bool save(T* target, int64_t num)
		{
			if (status)
			{
				status=false;
				if (num>0)
				{
					if (!target)
						target=SG_MALLOC(T, num);

					if (target)
					{
						status=(fwrite((void*) target, sizeof(T), num, file)==
							(size_t) num);
					}
				}
			}
			return status;
		}

		/** read a line (buffered; to be implemented)
		 *
		 * @param line linebuffer to write to
		 * @param len maximum length
		 */
		void get_buffered_line(char* line, uint64_t len)
		{

			/*
			if (!line_buffer)
			{
				line_buffer=SG_MALLOC(char, line_buffer_size);
				size_t num_read=fread((void*) target, sizeof(T), num, file);

					if (target)
					{
						size_t num_read=fread((void*) target, sizeof(T), num, file);
						status=((int64_t) num_read == num);

						if (!status)
							SG_ERROR("only %ld of %ld entries read. io error\n", (int64_t) num_read, num)
					}
					else
						SG_ERROR("failed to allocate memory while trying to read %ld entries from file \"s\"\n", (int64_t) num, filename)

						*/
		}

		/** free the line buffer */
		void free_line_buffer()
		{
			SG_FREE(line_buffer);
			line_buffer=NULL;
		}

		/** set the size of the line buffer
		 *
		 * @param bufsize size of the line buffer
		 */
		inline void set_line_buffer_size(int32_t bufsize)
		{
			if (bufsize<=0)
				bufsize=1024*1024;

			free_line_buffer();
			line_buffer_size=bufsize;
		}

		/** check if status is ok
		 *
		 * @return if status is ok
		 */
		inline bool is_ok() { return status; }

		/** @return object name */
		virtual const char* get_name() const { return "SimpleFile"; }

	protected:
		/** file descriptor */
		FILE* file;
		/** status of file operations */
		bool status;
		/** task */
		char task;
		/** filename */
		char* filename;

		/** size of line buffer */
		int32_t line_buffer_size;
		/** line buffer */
		char* line_buffer;
};
}
#endif
