/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __TEXT_GUI__H
#define __TEXT_GUI__H

#include "lib/config.h"

#ifndef HAVE_SWIG
#include <stdio.h>
#include "guilib/GUIHMM.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUIPreProc.h"
#include "guilib/GUIFeatures.h"
#include "gui/GUI.h"
#include "lib/Signal.h"

#include "guilib/GUIDistance.h"

class CTextGUI: public CGUI
{
public:
	CTextGUI(INT argc, char** argv);
	~CTextGUI();

	/// print the shogun prompt
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
	bool echo;
};
#endif //HAVE_SWIG
#endif
