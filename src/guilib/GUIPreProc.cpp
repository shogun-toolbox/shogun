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
	preproc=NULL;
}

CGUIPreProc::~CGUIPreProc()
{
	delete preproc;
}

bool CGUIPreProc::set_preproc(char* param)
{
	param=CIO::skip_spaces(param);
	if (strcmp(param,"PCACUT")==0)
	{
		delete preproc;
		preproc=new CPCACut();
		return true;
	}
	else if (strcmp(param,"NORMONE")==0)
	{
		delete preproc;
		preproc=new CNormOne();
		return true;
	}
	else if (strcmp(param,"PRUNEVARSUBMEAN")==0)
	{
		delete preproc;
		preproc=new CPruneVarSubMean();
		return true;
	}
	else if (strcmp(param,"NONE")==0)
	{
		delete preproc;
		preproc=NULL;
		return true;
	}
	else 
		CIO::not_implemented();

	return false;
}

bool CGUIPreProc::load(char* param)
{
	bool result=false;

	param=CIO::skip_spaces(param);

	delete preproc;
	preproc=NULL;

	FILE* file=fopen(param, "r");
	char id[5]="UDEF";

	if (file)
	{
		assert(fread(id, sizeof(char), 4, file)==4);
	
		if (strncmp(id, "PCAC", 4)==0)
		{
			preproc=new CPCACut();
		}
		else if (strncmp(id, "NRM1", 4)==0)
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

	return result;
}

bool CGUIPreProc::save(char* param)
{
	bool result=false; param=CIO::skip_spaces(param);

	if (preproc)
	{
		FILE* file=fopen(param, "w");
	
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
