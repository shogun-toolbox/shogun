#ifndef __TEXT_GUI__H
#define __TEXT_GUI__H

#include <stdio.h>

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
};
#endif
