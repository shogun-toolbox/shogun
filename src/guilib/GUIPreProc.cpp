#include "guilib/GUIPreProc.h"

CGUIPreProc::CGUIPreProc(CGUI * gui_)
  : gui(gui_), prunevarsubmean(), preproc(&prunevarsubmean)
{
}

CGUIPreProc::~CGUIPreProc()
{
}
