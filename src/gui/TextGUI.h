#ifndef __TEXT_GUI__H
#define __TEXT_GUI__H

#include <stdio.h>
#include "guilib/GUIHMM.h"
#include "guilib/GUISVM.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIObservation.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUIFeatures.h"
#include "gui/GUI.h"

class CTextGUI: public CGUI
{
public:
	CTextGUI(int argc, const char** argv);
	~CTextGUI();

	/// print the genefinder prompt
	void print_prompt();
	/// print the complete help
	void print_help();

	/// get line from user/stdin/file input
	/// @return true at EOF
	bool get_line(FILE* infile=stdin, bool show_prompt=true);

};
#endif
