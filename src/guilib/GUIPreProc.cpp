#include "guilib/GUIPreProc.h"
#include "preproc/LogPlusOne.h"
#include "preproc/NormOne.h"
#include "preproc/PruneVarSubMean.h"
#include "preproc/PCACut.h"
#include "preproc/SortWord.h"
#include "preproc/SortWordString.h"
#include "lib/io.h"
#include "lib/config.h"

#include <string.h>
#include <stdio.h>
#include <assert.h>

CGUIPreProc::CGUIPreProc(CGUI * gui_)
  : gui(gui_)
{
	preprocs=NULL;
	num_preprocs=0;
}

CGUIPreProc::~CGUIPreProc()
{
	for (INT i=0; i<num_preprocs; i++)
		delete preprocs[i];
	delete[] preprocs;
}


bool CGUIPreProc::add_preproc(CHAR* param)
{
	CPreProc* preproc=NULL;

	param=CIO::skip_spaces(param);
#ifdef HAVE_ATLAS
#ifdef HAVE_LAPACK
	if (strncmp(param,"PCACUT",6)==0)
	{
		INT do_whitening=0; 
		double thresh=1e-6 ;
		sscanf(param+6, "%i %le", &do_whitening, &thresh) ;
		CIO::message(M_INFO, "PCACUT parameters: do_whitening=%i thresh=%e", do_whitening, thresh) ;
		preproc=new CPCACut(do_whitening, thresh);
	}
	else 
#endif // LAPACK
#endif // ATLAS
	if (strncmp(param,"NORMONE",7)==0)
	{
		preproc=new CNormOne();
	}
	else if (strncmp(param,"LOGPLUSONE",10)==0)
	{
		preproc=new CLogPlusOne();
	}
	else if (strncmp(param,"SORTWORDSTRING",14)==0)
	{
		preproc=new CSortWordString();
	}
	else if (strncmp(param,"SORTWORD",8)==0)
	{
		preproc=new CSortWord();
	}
	else if (strncmp(param,"PRUNEVARSUBMEAN",15)==0)
	{
		INT divide_by_std=0; 
		sscanf(param+15, "%i", &divide_by_std);

		if (divide_by_std)
			CIO::message(M_INFO, "normalizing VARIANCE\n");
		else
			CIO::message(M_WARN, "NOT normalizing VARIANCE\n");

		preproc=new CPruneVarSubMean(divide_by_std==1);
	}
	else 
	{
		CIO::not_implemented();
		return false;
	}

	return add_preproc(preproc);
}

bool CGUIPreProc::clean_preproc(CHAR* param)
{
	while (num_preprocs>0) 
		del_preproc(param) ;
	
	return true ;
}

bool CGUIPreProc::del_preproc(CHAR* param)
{
	INT i,j;
	INT num=num_preprocs-1;

	CPreProc** pps=NULL; 
	param=CIO::skip_spaces(param);

	sscanf(param, "%i", &num);

	if (num<0)
		num=0;

	if (num>num_preprocs-1)
		num=num_preprocs-1;

	CIO::message(M_INFO, "deleting preproc %i/(%i)\n", num, num_preprocs);

	if (num_preprocs>0)
		delete preprocs[num];

	if (num_preprocs==1)
	{
		delete[]  preprocs;
		preprocs=NULL;
		num_preprocs=0;
	}
	else if (num_preprocs>1)
		pps= new CPreProc*[num_preprocs-1];

	if (pps)
	{
		j=0;
		for (i=0; i<num_preprocs; i++)
		{
			if (i!=num)
				pps[j++]=preprocs[i];
		}
		num_preprocs--;
		delete[] preprocs;
		preprocs=pps;
		return true;
	}
	else
		return false;
}

bool CGUIPreProc::load(CHAR* param)
{
	bool result=false;

	param=CIO::skip_spaces(param);

	CPreProc* preproc=NULL;

	FILE* file=fopen(param, "r");
	CHAR id[5]="UDEF";

	if (file)
	{
		assert(fread(id, sizeof(char), 4, file)==4);
	
#ifdef HAVE_ATLAS
#ifdef HAVE_LAPACK
		if (strncmp(id, "PCAC", 4)==0)
		{
			preproc=new CPCACut();
		}
		else 
#endif // LAPACK
#endif // ATLAS
		if (strncmp(id, "NRM1", 4)==0)
		{
			preproc=new CNormOne();
		}
		else if (strncmp(id, "PVSM", 4)==0)
		{
			preproc=new CPruneVarSubMean();
		}
		else
			CIO::message(M_ERROR, "unrecognized file\n");

		if (preproc && preproc->load_init_data(file))
		{
			printf("file successfully read\n");
			result=true;
		}

		fclose(file);
	}
	else
		CIO::message(M_ERROR, "opening file %s failed\n", param);

	if (result)
		return add_preproc(preproc);

	return result;
}

bool CGUIPreProc::save(CHAR* param)
{
	CHAR fname[1024];
	INT num=num_preprocs-1;
	bool result=false; param=CIO::skip_spaces(param);
	sscanf(param, "%s %i", fname, &num);

	if (num>=0 && num<num_preprocs && preprocs[num])
	{
		FILE* file=fopen(fname, "w");
		CPreProc* preproc=preprocs[num];
	
		fwrite(preproc->get_id(), sizeof(char), 4, file);
		if ((!file) ||	(!preproc->save_init_data(file)))
			printf("writing to file %s failed!\n", param);
		else
		{
			printf("successfully written preproc init data into \"%s\" !\n", param);
			result=true;
		}

		if (file)
			fclose(file);
	}
	else
		CIO::message(M_ERROR, "create preproc first\n");

	return result;
}

bool CGUIPreProc::add_preproc(CPreProc* preproc)
{
	INT i;
	CPreProc** pps=new CPreProc*[num_preprocs+1];

	for (i=0; i<num_preprocs; i++)
		pps[i]=preprocs[i];
	delete[] preprocs;

	preprocs=pps;
	preprocs[num_preprocs]=preproc;

	num_preprocs++;
		
	return true;
}
