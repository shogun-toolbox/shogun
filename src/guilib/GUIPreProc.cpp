#include "guilib/GUIPreProc.h"
#include "preproc/NormOne.h"
#include "preproc/PruneVarSubMean.h"
#include "preproc/PCACut.h"
#include "lib/io.h"
#include <string.h>

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
