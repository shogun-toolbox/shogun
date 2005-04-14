#ifndef __TEXT_GUI__H
#define __TEXT_GUI__H

#include <stdio.h>
#include "guilib/GUIHMM.h"
#include "guilib/GUISVM.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUIFeatures.h"
#include "gui/GUI.h"
#include "lib/Signal.h"

class CTextGUI: public CGUI
{
public:
	CTextGUI(INT argc, char** argv);
	~CTextGUI();

	/// print the genefinder prompt
	void print_prompt();
	/// print the complete help
	void print_help();

	/// get line from user/stdin/file input
	/// @return true at EOF
	CHAR* get_line(FILE* infile=stdin, bool show_prompt=true);

	/// get line from port 6766
	/// line is in input
	bool get_line_from_tcp();

	/// get line from user/stdin/file input
	/// @return true at EOF
	bool parse_line(CHAR* input);

protected:
	FILE* out_file;
	CHAR input[10000];
};
#endif
