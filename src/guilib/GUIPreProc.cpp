#include "guilib/GUIPreProc.h"

CGUIPreProc::CGUIPreProc(CGUI * gui_)
  : gui(gui_), prunevarsubmean(), pcacut(), preproc(&pcacut)
{
}

CGUIPreProc::~CGUIPreProc()
{
}
