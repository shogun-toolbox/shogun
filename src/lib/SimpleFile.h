#ifndef __SIMPLEFILE_H__
#define __SIMPLEFILE_H__

#include "lib/io.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

template <class T> class CSimpleFile
{
public:
	/// rw is either r for read and w for write
	CSimpleFile(char* fname, FILE* f)
	{
		file=f;
		this->fname=strdup(fname);
		status = (file!=NULL && fname!=NULL);
	}

	~CSimpleFile()
	{
	}

	//num is the number of read elements
	T* load(T* target, long& num=0)
	{
		if (status)
		{
			status=false;

			if (num==0)
			{
				bool seek_status=true;
				long cur_pos=ftell(file);

				if (cur_pos!=-1)
				{
					if (!fseek(file, 0, SEEK_END))
					{
						if ((num=(int)ftell(file)) != -1)
						{
							CIO::message("file of size %ld bytes == %ld entries detected\n", num,num/sizeof(T));
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
					CIO::message("filesize autodetection failed\n");
					num=0;
					return NULL;
				}
			}

			if (num>0)
			{
				if (!target)
					target=new T[num];

				if (target)
					status=(fread((void*) target, sizeof(T), num, file) == (unsigned int) num);
			}
			return target;
		}
		else 
		{
			num=-1;
			return NULL;
		}
	}

	bool save(T* target, long num)
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
	char task;
	char* fname;
};
#endif
