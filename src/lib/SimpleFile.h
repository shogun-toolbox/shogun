/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SIMPLEFILE_H__
#define __SIMPLEFILE_H__

#include "lib/io.h"

#include <stdio.h>
#include <string.h>

template <class T> class CSimpleFile
{
public:
	/// rw is either r for read and w for write
	CSimpleFile(CHAR* fname, FILE* f)
	{
		file=f;
		this->fname=strdup(fname);
		status = (file!=NULL && fname!=NULL);
	}

	~CSimpleFile()
	{
		free(fname);
	}

	//num is the number of read elements
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
							CIO::message(M_INFO, "file of size %ld bytes == %ld entries detected\n", num,num/sizeof(T));
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
					CIO::message(M_ERROR, "filesize autodetection failed\n");
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
						CIO::message(M_ERROR, "only %ld of %ld entries read. io error\n", (LONG) num_read, num);
				}
				else
					CIO::message(M_ERROR, "failed to allocate memory while trying to read %ld entries from file \"s\"\n", (LONG) num, fname);
			}
			return target;
		}
		else 
		{
			num=-1;
			return NULL;
		}
	}

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

	inline bool is_ok()
	{
		return status;
	}

protected:
	FILE* file;
	bool status;
	CHAR task;
	CHAR* fname;
};
#endif
