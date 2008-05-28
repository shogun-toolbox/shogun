/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SIMPLEFILE_H__
#define __SIMPLEFILE_H__

#include "lib/io.h"
#include "base/SGObject.h"

#include <stdio.h>
#include <string.h>

/** template class SimpleFile */
template <class T> class CSimpleFile : public CSGObject
{
	public:
		/** constructor
		 * rw is either r for read and w for write
		 *
		 * @param fname filename
		 * @param f file descriptor
		 */
		CSimpleFile(CHAR* fname, FILE* f)
		: CSGObject()
		{
			file=f;
			filename=strdup(fname);
			status = (file!=NULL && filename!=NULL);
		}

		~CSimpleFile() { free(filename); }

		/** load
		 *
		 * @param target load target
		 * @param num number of read elements
		 * @return loaded target or NULL if unsuccessful
		 */
		T* load(T* target, LONG& num=0)
		{
			if (status)
			{
				status=false;

				if (num==0)
				{
					bool seek_status=true;
					LONG cur_pos=ftell(file);

					if (cur_pos!=-1)
					{
						if (!fseek(file, 0, SEEK_END))
						{
							if ((num=(int)ftell(file)) != -1)
							{
								SG_INFO( "file of size %ld bytes == %ld entries detected\n", num,num/sizeof(T));
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
						SG_ERROR( "filesize autodetection failed\n");
						num=0;
						return NULL;
					}
				}

				if (num>0)
				{
					if (!target)
						target=new T[num];

					if (target)
					{
						size_t num_read=fread((void*) target, sizeof(T), num, file);
						status=((LONG) num_read == num);

						if (!status)
							SG_ERROR( "only %ld of %ld entries read. io error\n", (LONG) num_read, num);
					}
					else
						SG_ERROR( "failed to allocate memory while trying to read %ld entries from file \"s\"\n", (LONG) num, filename);
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
		bool save(T* target, LONG num)
		{
			if (status)
			{
				status=false;
				if (num>0)
				{
					if (!target)
						target=new T[num];

					if (target)
					{
						status=(fwrite((void*) target, sizeof(T), num, file) == (unsigned long) num);
					}
				}
			}
			return status;
		}

		/** check if status is ok
		 *
		 * @return if status is ok
		 */
		inline bool is_ok() { return status; }

	protected:
		/** file descriptor */
		FILE* file;
		/** status of file operations */
		bool status;
		/** task */
		CHAR task;
		/** filename */
		CHAR* filename;
};
#endif
