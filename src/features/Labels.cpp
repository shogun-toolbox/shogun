#include "features/Labels.h"
#include "lib/common.h"
#include "lib/File.h"
#include "lib/io.h"

CLabels::CLabels(long num_lab) : num_labels(num_lab), labels(NULL)
{
}

CLabels::CLabels(char* fname)
{
	num_labels=0;
	labels=NULL;

	load(fname);
}

INT* CLabels::get_labels(long &len)
{

	len=num_labels;

	if (num_labels>0)
	{
		INT* labels=new INT[num_labels] ;
		for (long i=0; i<len; i++)
			labels[i]=get_label(i) ;
		return labels ;
	}
	else 
		return NULL;
}

bool CLabels::load(char* fname)
{
	bool status=false;

	delete[] labels;
	num_labels=0;

	CFile f(fname, 'r', F_INT);
	labels=f.load_int_data(NULL, num_labels);

    if (!f.is_ok())
		CIO::message("loading file \"%s\" failed", fname);
	else
	{
		CIO::message("%ld labels successfully read\n", num_labels);
		status=true;
	}

	return status;
}

bool CLabels::save(char* fname)
{
	return false;
}
