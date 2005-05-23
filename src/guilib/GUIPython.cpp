#include "lib/config.h"

#ifdef HAVE_PYTHON
#include "guilib/GUIPython.h"
#include "gui/TextGUI.h"
#include "lib/common.h"

//next line is not necessary, however if disabled causes a warning
#define libnumeric_UNIQUE_SYMBOL libnumeric_API
#include <numarray/libnumarray.h>
#include <numarray/arrayobject.h>

extern CTextGUI* gui;

CGUIPython::CGUIPython()
{
	import_libnumarray();
	assert(libnumeric_API);
}

CGUIPython::~CGUIPython()
{
}

PyObject* CGUIPython::py_send_command(PyObject* self, PyObject* args)
{
	char *cmd;

	if (!PyArg_ParseTuple(args, "s", &cmd))
		return NULL;

	gui->parse_line(cmd);

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_system(PyObject* self, PyObject* args)
{
	char *cmd;

	if (!PyArg_ParseTuple(args, "s", &cmd))
		return NULL;
	::system(cmd);

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_get_kernel_matrix(PyObject* self, PyObject* args)
{

	double *result=NULL;
	int m, n;

	CKernel* k = gui->guikernel.get_kernel();

	if (!k)
		CIO::message(M_ERROR, "no kernel set\n");
	else
		result=k->get_kernel_matrix(m,n);

	if(result)
		return (PyObject*) NA_NewArray((void *)result, tFloat64, 2, m, n);
	else
		return NULL;
}

PyObject* CGUIPython::py_set_features(PyObject* self, PyObject* args)
{
	PyObject   *py_ofeat = NULL;
	PyArrayObject *py_afeat = NULL;
	char* target = NULL;

	if (PyArg_ParseTuple(args, "Os", &py_ofeat, &target))
	{
		/* Align, Byteswap, Contiguous, Typeconvert */
		py_afeat  = NA_InputArray(py_ofeat, tFloat64, NUM_C_ARRAY);
		if (py_afeat && target)
		{
			CIO::message(M_MESSAGEONLY, "nd:%d\n", py_afeat->nd);
			CIO::message(M_MESSAGEONLY, "target: %s\n", target);
			if ((py_afeat->nd == 2))
			{
				double* feat= (double*) NA_OFFSETDATA(py_afeat);
				int num_vec=py_afeat->dimensions[0];
				int num_feat=py_afeat->dimensions[1];

				if (feat)
				{
					CRealFeatures* f= new CRealFeatures((LONG) 0);
					REAL* fm=new REAL[num_vec*num_feat];
					assert(fm);
					for(int i=0; i<num_vec; i++)
					{
						for(int j=0; j<num_feat; j++)
						{
							CIO::message(M_MESSAGEONLY, "vec(%d,%d)=%f\n", i,j,feat[i*num_feat+j]);
							fm[i*num_feat+j]=feat[i*num_feat+j];
						}
					}
					((CRealFeatures*) f)->set_feature_matrix(fm, num_feat, num_vec);

					if (f && target)
					{
						if (!strncmp(target, "TRAIN", strlen("TRAIN")))
						{
							gui->guifeatures.set_train_features(f);
						}
						else if (!strncmp(target, "TEST", strlen("TEST")))
						{
							gui->guifeatures.set_test_features(f);
						}
					}
					else
						CIO::message(M_ERROR, "usage is gf('set_features', 'TRAIN|TEST', features, ...)");
				}
				else
					CIO::message(M_ERROR,"empty feats ??\n");
			}
			else
				CIO::message(M_ERROR, "set_features: arrays must have 1 dimension.\n");
		}
		else
			CIO::message(M_ERROR, "set_features: error converting array inputs.\n");
	}
	else
		CIO::message(M_ERROR, "set_features: Invalid parameters.\n");

	Py_XDECREF(py_afeat);
	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_set_kernel_matrix(PyObject* self, PyObject* args)
{
	Py_INCREF(Py_None);
	return Py_None;
}

//static PyObject * gfpython_system(PyObject *self, PyObject *args)
//{
//	char *command;
//	int sts;
//
//	if (!PyArg_ParseTuple(args, "s", &command))
//	{
//		PyErr_SetString(PyExc_RuntimeError, "du bist doch doof");
//		return NULL;
//	}
//	sts = system(command);
//	return Py_BuildValue("i", sts);
//}
//

PyObject* CGUIPython::py_test(PyObject* self, PyObject* args)
{
	PyObject   *py_ofeat = NULL;
	PyArrayObject *py_afeat = NULL;

	if (PyArg_ParseTuple(args, "O", &py_ofeat))
	{
		/* Align, Byteswap, Contiguous, Typeconvert */
		py_afeat  = NA_InputArray(py_ofeat, tFloat64, NUM_C_ARRAY);
		if (py_afeat)
		{
			if ((py_afeat->nd == 1))
			{
				double* feat= (double*) NA_OFFSETDATA(py_afeat);
				int num=py_afeat->dimensions[0];

				if (feat)
				{
					for (int i=0; i<num; i++)
						CIO::message(M_MESSAGEONLY, "%f\n", feat[i]);
				}
				else
					CIO::message(M_ERROR,"empty feats ??\n");
			}
			else
				CIO::message(M_ERROR, "set_features: arrays must have 1 dimension.\n");
		}
		else
			CIO::message(M_ERROR, "set_features: error converting array inputs.\n");
	}
	else
		CIO::message(M_ERROR, "set_features: Invalid parameters.\n");

	Py_XDECREF(py_afeat);
	Py_INCREF(Py_None);
	return Py_None;
}

#endif
