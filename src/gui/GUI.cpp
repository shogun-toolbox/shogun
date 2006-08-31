/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

//// main - the one and only ///
//
#include "lib/config.h"

#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

#include "gui/TextGUI.h"
#include "lib/Signal.h"

extern CTextGUI* gui;
const INT READLINE_BUFFER_SIZE = 10000 ;

int main(int argc, char* argv[])
{	
	gui=new CTextGUI(argc, argv) ;

	if (argc<=1)
	{
		while (gui->parse_line(gui->get_line()));
	}
	else
	{
		if (argc>=2)
		{
			if ( argc>2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "/?") || !strcmp(argv[1], "--help"))
			{
				CIO::message(M_ERROR, "usage: shogun [ <-h|--help|/?|-i|<script> ]\n\n");
				CIO::message(M_INFO, "if no options are given genfinder enters interactive mode\n");
				CIO::message(M_INFO, "if <script> is specified the commands will be executed");
				CIO::message(M_INFO, "if -i is specified shogun will listen on port 7367 (==hex(sg), *dangerous* as commands from any source are accepted");
				return 1;
			}
			else if ( argc>2 || !strcmp(argv[1], "-i") || !strcmp(argv[1], "/?") || !strcmp(argv[1], "--help"))
			{
				int s=socket(AF_INET, SOCK_STREAM, 0);
				struct sockaddr_in sa;
				sa.sin_family=AF_INET;
				sa.sin_port=htons(7367);
				sa.sin_addr.s_addr=INADDR_ANY;
				bzero(&(sa.sin_zero), 8);

				bind(s, (sockaddr*) (&sa), sizeof(sockaddr_in));
				listen(s, 1);
				int s2=accept(s, NULL, NULL);
				CIO::message(M_INFO, "accepting connection\n");

				CHAR input[READLINE_BUFFER_SIZE];
				do
				{
					bzero(input, sizeof(input));
					int length=read(s2, input, sizeof(input));
					if (length>0 && length<(int) sizeof(input))
						input[length]='\0';
					else
					{
						CIO::message(M_ERROR, "error reading cmdline\n");
						return 1;
					}
				}
				while(gui->parse_line(input));
				return 0;
			}
			else
			{
				FILE* file=fopen(argv[1], "r");

				if (!file)
				{
					CIO::message(M_ERROR, "error opening/reading file: \"%s\"",argv[1]);
					return 1;
				}
				else
				{
					while(!feof(file) && gui->parse_line(gui->get_line(file, false)));
					fclose(file);
				}
			}
		}
	}

	CIO::message(M_INFO, "quitting...\n");
	delete gui ;

	return 0;
}
