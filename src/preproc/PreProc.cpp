#include "PreProc.h"
#include "lib/io.h"

CPreProc::CPreProc(const char* name, const char* id)
  : preproc_name(name), preproc_id(id)
{
	CIO::message("creating PreProc \"%s\"\n", preproc_name);
}

CPreProc::~CPreProc()
{
  CIO::message("deleting PreProc \"%s\"\n", preproc_name) ;
}
