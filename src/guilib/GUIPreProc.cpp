#include "guilib/GUIPreProc.h"
#include "preproc/NormOne.h"
#include "preproc/PruneVarSubMean.h"
#include "preproc/PCACut.h"
#include "lib/io.h"

#include <string.h>
#include <stdio.h>
#include <assert.h>

CGUIPreProc::CGUIPreProc(CGUI * gui_)
  : gui(gui_)
{
	preprocs=NULL;
	num_preprocs=NULL;
}

CGUIPreProc::~CGUIPreProc()
{
	for (int i=0; i<num_preprocs; i++)
		delete preprocs[i];
	delete[] preprocs;
}

bool CGUIPreProc::add_preproc(char* param)
{
	CPreProc* preproc=NULL;

	param=CIO::skip_spaces(param);
#ifndef NO_LAPACK
	if (strncmp(param,"PCACUT",6)==0)
	{
		int do_whitening=0; 
		double thresh=1e-6 ;
		sscanf(param+6, "%i %le", &do_whitening, &thresh) ;
		CIO::message("PCACUT parameters: do_whitening=%i thresh=%e", do_whitening, thresh) ;
		preproc=new CPCACut(do_whitening, thresh);
	}
	else 
#endif
	  if (strncmp(param,"NORMONE",7)==0)
	{
		preproc=new CNormOne();
	}
	else if (strncmp(param,"PRUNEVARSUBMEAN",15)==0)
	{
		int divide_by_std=0; 
		sscanf(param+15, "%i", &divide_by_std);

		if (divide_by_std)
			CIO::message("normalizing VARIANCE\n");
		else
			CIO::message("NOT normalizing VARIANCE\n");

		preproc=new CPruneVarSubMean(divide_by_std==1);
	}
//	else if (strncmp(param,"NONE",4)==0)
//	{
//		delete preproc;
//		preproc=NULL;
//		return true;
//	}
	else 
	{
		CIO::not_implemented();
		return false;
	}

	return add_preproc(preproc);
}

bool CGUIPreProc::del_preproc(char* param)
{
	int i,j,num=num_preprocs-1;
	CPreProc** pps=NULL; 
	CPreProc* removed_preproc=NULL;
	param=CIO::skip_spaces(param);

	sscanf(param, "%i", &num);

	if (num_preprocs>0)
		delete preprocs[num];

	if (num_preprocs>1)
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

bool CGUIPreProc::load(char* param)
{
	bool result=false;

	param=CIO::skip_spaces(param);

	CPreProc* preproc=NULL;

	FILE* file=fopen(param, "r");
	char id[5]="UDEF";

	if (file)
	{
		assert(fread(id, sizeof(char), 4, file)==4);
	
#ifndef NO_LAPACK
		if (strncmp(id, "PCAC", 4)==0)
		{
			preproc=new CPCACut();
		}
		else 
#endif
		if (strncmp(id, "NRM1", 4)==0)
		{
			preproc=new CNormOne();
		}
		else if (strncmp(id, "PVSM", 4)==0)
		{
			preproc=new CPruneVarSubMean();
		}
		else
			CIO::message("unrecognized file\n");

		if (preproc && preproc->load_init_data(file))
		{
			printf("file successfully read\n");
			result=true;
		}

		fclose(file);
	}
	else
		CIO::message("opening file %s failed\n", param);

	if (result)
		return add_preproc(preproc);

	return result;
}

bool CGUIPreProc::save(char* param)
{
	char fname[1024];
	int num=num_preprocs-1;
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
		CIO::message("create preproc first\n");

	return result;
}

bool CGUIPreProc::add_preproc(CPreProc* preproc)
{
	int i;
	CPreProc** pps=new CPreProc*[num_preprocs+1];

	for (i=0; i<num_preprocs; i++)
		pps[i]=preprocs[i];
	delete[] preprocs;

	preprocs=pps;
	preprocs[num_preprocs]=preproc;

	num_preprocs++;
		
	return true;
}
