#include <string.h>

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

bool CSGInterface::handle()
{
	INT len=0;

	if (!gui)
		gui=new CTextGUI(0, NULL);

	if (!gui)
		SG_SERROR("gui could not be initialized.");

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
		SG_SERROR("%s:%s", "string expected as first argument", e.get_exception_string());
	}

	if (strmatch(action, len, N_SEND_COMMAND))
	{
		parse_args(2, 0);
		CHAR* cmd=interface->get_string(len);
		gui->parse_line(cmd);
		delete[] cmd;
	}
	else
		return false;

#ifndef WIN32
    CSignal::unset_handler();
#endif
	return true;
}
