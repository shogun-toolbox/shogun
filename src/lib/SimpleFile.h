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

	T* load(int num, T* target)
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
					status=(fread((void*) target, sizeof(T), num, file) == (unsigned int) num);
				}
			}
		}
		return target;
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
