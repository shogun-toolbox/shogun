#ifndef __SIMPLEFILE_H__
#define __SIMPLEFILE_H__

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
	T* load(T* target, int& num=0)
	{
		if (status)
		{
			status=false;

			if (num==0)
			{
				if (!fseek(file, 0, SEEK_END))
				{
					if ((num=(int)ftell(file)) != -1)
						num/=sizeof(T);
				}
			}

			if (num>0)
			{
				if (!target)
					target=new T[num];

				if (target)
				{
					status=(fread((void*) target, sizeof(T), num, file) == (unsigned int) num);
				}
			}
			return target;
		}
		else 
		{
			num=-1;
			return NULL;
		}
	}

	bool save(T* target, int num)
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
					status=(fwrite((void*) target, sizeof(T), num, file) == num);
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
