#ifndef __TEXT_GUI__H
#define __TEXT_GUI__H

#include <stdio.h>
#include "guilib/GUIHMM.h"
#include "guilib/GUISVM.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIObservation.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUIFeatures.h"

class CTextGUI
{
public:
	CTextGUI();
	~CTextGUI();

	/// print the genefinder prompt
	void print_prompt();
	/// print the complete help
	void print_help();

	/// get line from user/stdin/file input
	/// @return true at EOF
	bool get_line(FILE* infile=stdin);

	CGUISVM guisvm;
	CGUIHMM guihmm;
	CGUIKernel guikernel;
	CGUIObservation guiobs;
	CGUIPreProc guipreproc;
	CGUIFeatures guifeatures;
};
#endif
