#include "PreProc.h"
#include "lib/io.h"

CPreProc::CPreProc(const char * name)
  : preproc_name(name)
{
}

CPreProc::~CPreProc()
{
  CIO::message("PreProc object destroyed\n") ;
}
