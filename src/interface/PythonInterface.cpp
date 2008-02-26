#include "lib/config.h"

#if defined(HAVE_PYTHON) && !defined(HAVE_SWIG)            

#include "interface/PythonInterface.h"
#include "interface/SGInterface.h"

#include "lib/ShogunException.h"
#include "lib/io.h"
#include "lib/python.h"

extern "C" {
#include <object.h>
#include <../../numarray/numpy/libnumarray.h>
#include <numpy/ndarrayobject.h>
}

extern CSGInterface* interface;

CPythonInterface::CPythonInterface(PyObject* self, PyObject* args) : CSGInterface()
{
	ASSERT(PyTuple_Check(args));
	m_rhs=args;
	m_nrhs=PyTuple_GET_SIZE(args);

	m_nlhs=0;
	Py_INCREF(Py_None);
	m_lhs=Py_None;
}

CPythonInterface::~CPythonInterface()
{
}

/** get functions - to pass data from the target interface to shogun */
void CPythonInterface::parse_args(INT num_args, INT num_default_args)
{
}


/// get type of current argument (does not increment argument counter)
IFType CPythonInterface::get_argument_type()
{
	return UNDEFINED;
}


INT CPythonInterface::get_int()
{
	const PyObject* i=get_current_arg();
	if (!i || !PyInt_Check(i))
		SG_ERROR("Expected Scalar Integer as argument %d\n", arg_counter);

	return PyInt_AS_LONG(i);
}

DREAL CPythonInterface::get_real()
{
	const PyObject* f=get_current_arg();
	if (!f || !PyFloat_Check(f))
		SG_ERROR("Expected Scalar Float as argument %d\n", arg_counter);

	arg_counter++;
	return PyFloat_AS_DOUBLE(f);
}

bool CPythonInterface::get_bool()
{
	const PyObject* b=get_current_arg();
	if (!b || !PyBool_Check(b))
		SG_ERROR("Expected Scalar Boolean as argument %d\n", arg_counter);

	arg_counter++;
	return PyInt_AS_LONG(b) != 0;
}


CHAR* CPythonInterface::get_string(INT& len)
{
	const PyObject* s=get_current_arg();
	if (!s || !PyString_Check(s))
		SG_ERROR("Expected String as argument %d\n", arg_counter);

	len=PyString_Size((PyObject*) s);
	CHAR* str=PyString_AS_STRING(s);
	ASSERT(str && len>0);

	CHAR* cstr = new CHAR[len+1];
	ASSERT(cstr);

	memcpy(cstr, str, len+1);
	cstr[len]='\0';

	arg_counter++;
	return cstr;
}

void CPythonInterface::get_byte_vector(BYTE*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CPythonInterface::get_int_vector(INT*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CPythonInterface::get_shortreal_vector(SHORTREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CPythonInterface::get_real_vector(DREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
}


void CPythonInterface::get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec)
{
}

void CPythonInterface::get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec)
{
}

void CPythonInterface::get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec)
{
}

void CPythonInterface::get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec)
{
	//has to be rewritting using numpy
}


void CPythonInterface::get_byte_sparsematrix(TSparse<BYTE>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CPythonInterface::get_int_sparsematrix(TSparse<INT>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CPythonInterface::get_shortreal_sparsematrix(TSparse<SHORTREAL>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CPythonInterface::get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
}


void CPythonInterface::get_string_list(T_STRING<CHAR>*& strings, INT& num_str)
{
}


/** set functions - to pass data from shogun to the target interface */
void CPythonInterface::create_return_values(INT num_val)
{
}

void CPythonInterface::set_byte_vector(BYTE* vec, INT len)
{
}

void CPythonInterface::set_int_vector(INT* vec, INT len)
{
}

void CPythonInterface::set_shortreal_vector(SHORTREAL* vec, INT len)
{
}

void CPythonInterface::set_real_vector(DREAL* vec, INT len)
{
}


void CPythonInterface::set_byte_matrix(BYTE* matrix, INT num_feat, INT num_vec)
{
}

void CPythonInterface::set_int_matrix(INT* matrix, INT num_feat, INT num_vec)
{
}

void CPythonInterface::set_shortreal_matrix(SHORTREAL* matrix, INT num_feat, INT num_vec)
{
}

void CPythonInterface::set_real_matrix(DREAL* matrix, INT num_feat, INT num_vec)
{
}


void CPythonInterface::set_byte_sparsematrix(TSparse<BYTE>* matrix, INT num_feat, INT num_vec)
{
}

void CPythonInterface::set_int_sparsematrix(TSparse<INT>* matrix, INT num_feat, INT num_vec)
{
}

void CPythonInterface::set_shortreal_sparsematrix(TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)
{
}

void CPythonInterface::set_real_sparsematrix(TSparse<DREAL>* matrix, INT num_feat, INT num_vec)
{
}


void CPythonInterface::set_string_list(T_STRING<CHAR>* strings, INT num_str)
{
}


void CPythonInterface::submit_return_values()
{
}

PyObject* sg(PyObject* self, PyObject* args);
void exitsg(void);

static PyMethodDef sg_pythonmethods[] = {
    {(char*) "sg",  sg, METH_VARARGS, (char*) "Shogun."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initsg(void)
{
	// initialize python interpreter
	Py_Initialize();

	// initialize threading (just in case it is needed)
	PyEval_InitThreads();

	// initialize textgui
	//gui=new CTextGUI(0, NULL) ;

    // callback to cleanup at exit
	Py_AtExit(exitsg);

	// initialize callbacks
    Py_InitModule((char*) "sg", sg_pythonmethods);
}

PyObject* sg(PyObject* self, PyObject* args)
{
	delete interface;
	interface=new CPythonInterface(self, args);

	try
	{
		if (!interface->handle())
			SG_ERROR("interface currently does not handle this command\n");
	}
	catch (ShogunException e)
	{
		Py_INCREF(Py_None);
		return Py_None;
	}

	return ((CPythonInterface*) interface)->get_return_values();
}

void exitsg(void)
{
	SG_SINFO( "quitting...\n");
	//delete gui;
}
#endif // HAVE_PYTHON && HAVE_SWIG
