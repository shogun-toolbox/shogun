#include "guilib/GUIPreProc.h"

CGUIPreProc::CGUIPreProc(CGUI * gui_)
  : gui(gui_), prunevarsubmean(), pcacut(), normone(), preproc(&normone)
{
}

CGUIPreProc::~CGUIPreProc()
{
}

bool CGUIPreProc::set_preproc(char* param)
{
	return false;
}
