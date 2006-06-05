#include "preproc/PreProc.h"
#include "lib/io.h"

CPreProc::CPreProc(const CHAR* name, const CHAR* id)
  : preproc_name(name), preproc_id(id)
{
	CIO::message(M_INFO, "creating PreProc \"%s\"\n", preproc_name);
}

CPreProc::~CPreProc()
{
  CIO::message(M_INFO, "deleting PreProc \"%s\"\n", preproc_name) ;
}
