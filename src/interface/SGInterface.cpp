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

static CSGInterfaceMethod sg_methods[] =
{
	{(CHAR*) "test", (&CSGInterface::test), 0, (CHAR*) "this is a test."},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};


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
	bool success=false;

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
		SG_SERROR("String expected as first argument: %s\n", e.get_exception_string());
	}

	SG_PRINT("action: %s, nlhs %d, nrhs %d\n", action, m_nlhs, m_nrhs);
	INT i=0;
	while (sg_methods[i].action)
	{
		if (strmatch(action, len, sg_methods[i].action))
		{
			if (!(interface->*(sg_methods[i].method))())
				SG_SERROR("Usage: %s\n", sg_methods[i].usage);
			else
			{
				success=true;
				break;
			}
		}
		i++;
	}

	// FIXME: invoke old interface
	if (!gui)
		gui=new CTextGUI(0, NULL);

	if (!gui)
		SG_SERROR("GUI could not be initialized.\n");

	if(!success && strmatch(action, len, N_SEND_COMMAND))
	{
		//parse_args(2, 0);
		CHAR* cmd=interface->get_string(len);
		SG_PRINT("cmd:%s\n", cmd);
		gui->parse_line(cmd);
		delete[] cmd;
		success=true;
	}

#ifndef WIN32
    CSignal::unset_handler();
#endif

	delete[] action;
	delete gui;
	return success;
}

bool CSGInterface::test()
{
	if (m_nrhs<2)
		return false;

/*
	DREAL* vector;
	INT len;

	get_real_vector(vector, len);
	for (INT i=0; i<len; i++) SG_PRINT("data %d: %f\n", i, vector[i]);
	reset_counter();
	set_real_vector(vector, len);
	delete[] vector;
*/

	TSparse<DREAL>* matrix;
	INT num_feat, num_vec;

	get_real_sparsematrix(matrix, num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
	{
		for (INT j=0; j<num_feat; j++)
		{
			INT idx=i*num_feat+j;
			SG_PRINT("data %d, %d, index %d: %f\n", i, j, idx, matrix[idx]);
		}
	}

	reset_counter();
	set_real_sparsematrix(matrix, num_feat, num_vec);
	delete[] matrix;


	return true;
}
#endif // !HAVE_SWIG
