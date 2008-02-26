#include "lib/config.h"

#if !defined(HAVE_SWIG)

#include <string.h>
#include <stdlib.h>

#include "interface/SGInterface.h"
#include "lib/ShogunException.h"
#include "guilib/GUICommands.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"

CSGInterface* interface=NULL;
extern CTextGUI* gui;

CSGInterface::CSGInterface()
{
	arg_counter=0;
	m_nlhs=0;
	m_nrhs=0;
}

CSGInterface::~CSGInterface()
{
}

INT CSGInterface::get_int_from_string()
{
	INT len=0;
	CHAR* str=get_string(len);
	return strtol(str, NULL, 10);
}

DREAL CSGInterface::get_real_from_string()
{
	INT len=0;
	CHAR* str=get_string(len);
	return strtod(str, NULL);
}

bool CSGInterface::get_bool_from_string()
{
	INT len=0;
	CHAR* str=get_string(len);
	return strtol(str, NULL, 10)!=0;
}

bool CSGInterface::handle()
{
	INT len=0;
	bool success=true;

	if (!gui)
		gui=new CTextGUI(0, NULL);

	if (!gui)
		SG_SERROR("GUI could not be initialized.\n");

#ifndef WIN32
    CSignal::set_handler();
#endif
	CHAR* action=NULL;
	try
	{
		action=interface->get_action(len);
	}
	catch (ShogunException e)
	{
		SG_SERROR("%s: %s", "string expected as first argument", e.get_exception_string());
	}

	SG_PRINT("action: %s, nlhs %d, nrhs %d\n", action, m_nlhs, m_nrhs);
	if (strmatch(action, len, "test"))
	{
		if (!test())
			SG_SERROR("action test failed!");
	}
	else if(strmatch(action, len, N_SEND_COMMAND))
	{
		parse_args(2, 0);
		CHAR* cmd=interface->get_string(len);
		SG_PRINT("cmd:%s\n", cmd);
		gui->parse_line(cmd);
		delete[] cmd;
	}
	else
		success=false;

#ifndef WIN32
    CSignal::unset_handler();
#endif
	delete[] action;
	return success;
}

bool CSGInterface::test()
{
	if (m_nrhs<2)
		return false;

	BYTE* vector;
	INT len;

	get_byte_vector(vector, len);
	reset_counter();
	set_byte_vector(vector, len);
	delete[] vector;

	return true;
}
#endif // !HAVE_SWIG
