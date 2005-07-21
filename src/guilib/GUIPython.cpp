#include "lib/config.h"

#ifdef HAVE_PYTHON
#include "guilib/GUIPython.h"
#include "gui/TextGUI.h"
#include "lib/common.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "features/CharFeatures.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "kernel/WeightedDegreePositionCharKernel.h"
#include "kernel/CombinedKernel.h"

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

PyObject* CGUIPython::py_help(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_crc(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_translate_string(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_hmm(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_viterbi(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_svm(PyObject* self, PyObject* args)
{
	CSVM* svm=gui->guisvm.get_svm();

	if (svm && svm->get_num_support_vectors()>0)
	{
		PyArrayObject* py_alphas=NA_NewArray(NULL, tFloat64, 1, svm->get_num_support_vectors());
		PyArrayObject* py_sv_idx=NA_NewArray(NULL, tInt32, 1, svm->get_num_support_vectors());
		Float64 b=svm->get_bias();

		CIO::message(M_DEBUG,"num_sv: %d\n", svm->get_num_support_vectors());

		if (py_alphas && py_sv_idx)
		{
			for (int i=0; i< svm->get_num_support_vectors(); i++)
			{
				NA_set1_Float64(py_alphas, i, svm->get_alpha(i));
				NA_set1_Int64(py_sv_idx, i, svm->get_support_vector(i));
			}

			PyObject* ret=PyDict_New();
			PyDict_SetItemString(ret, "b", Py_BuildValue("f",b));
			PyDict_SetItemString(ret, "sv_idx", (PyObject*) py_sv_idx);
			PyDict_SetItemString(ret, "alpha", (PyObject*) py_alphas);

			return ret;
		}
	}

	return NULL;
}

PyObject* CGUIPython::py_get_kernel_matrix(PyObject* self, PyObject* args)
{

	Float64 *result=NULL;
	int m, n;

	CKernel* k = gui->guikernel.get_kernel();

	if (!k)
		CIO::message(M_ERROR, "no kernel set\n");
	else
		result=k->get_kernel_matrix(m,n);

	if(result)
		return (PyObject*) NA_NewArray(result, tFloat64, 2, m, n);
	else
		return NULL;
}


PyObject* CGUIPython::py_get_kernel_optimization(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_compute_by_subkernels(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_subkernels_weights(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_last_subkernel_weights(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_wd_pos_weights(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_subkernel_weights(PyObject* self, PyObject* args)
{
	CKernel* k = gui->guikernel.get_kernel() ;
	INT degree=-1;
	INT length=-1;

	if (k)
	{
		if (k->get_kernel_type() == K_WEIGHTEDDEGREE)
		{
			CWeightedDegreeCharKernel *kernel = (CWeightedDegreeCharKernel *) k;

			const REAL* weights = kernel->get_degree_weights(degree, length) ;
			if (length == 0)
				length = 1;

			PyArrayObject* py_weights=NA_NewArray(NULL, tFloat64, degree, length);

			for (int i=0; i<degree; i++)
				for (int j=0; j<length; j++)
					NA_set2_Float64(py_weights, i, j, weights[i*length+j]);

			return (PyObject*) py_weights;
		}
		else if (k->get_kernel_type() == K_WEIGHTEDDEGREEPOS)
		{
			CWeightedDegreePositionCharKernel *kernel = (CWeightedDegreePositionCharKernel *) k;

			const REAL* weights = kernel->get_degree_weights(degree, length) ;
			if (length == 0)
				length = 1;

			PyArrayObject* py_weights=NA_NewArray(NULL, tFloat64, degree, length);

			for (int i=0; i<degree; i++)
				for (int j=0; j<length; j++)
					NA_set2_Float64(py_weights, i, j, weights[i*length+j]);

			return (PyObject*) py_weights;
		}
		else if (k->get_kernel_type() == K_COMBINED)
		{
			CCombinedKernel *kernel = (CCombinedKernel *) k;
			INT num_weights = -1 ;
			const REAL* weights = kernel->get_subkernel_weights(num_weights) ;

			PyArrayObject* py_weights=NA_NewArray(NULL, tFloat64, 1, num_weights);

			for (int i=0; i<num_weights; i++)
				NA_set1_Float64(py_weights, i, weights[i]);

			return (PyObject*) py_weights;
		}
		else
			CIO::message(M_ERROR, "kernel does not have any subkernel weights\n");
	}
	else
		CIO::message(M_ERROR, "no kernel set\n");

	return NULL;
}

PyObject* CGUIPython::py_last_subkernel_weights(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_features(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_labels(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_version(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_preproc_init(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_hmm_defs(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_hmm(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_model_prob_no_b_trans(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_best_path_no_b_trans(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_best_path_trans(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_best_path_no_b(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_append_hmm(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_svm(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_kernel_parameters(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_custom_kernel(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_kernel_init(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_features(PyObject* self, PyObject* args)
{
	PyObject   *py_ofeat = NULL;
	char* target = NULL;

	if (PyArg_ParseTuple(args, "sO", &target, &py_ofeat))
	{
		if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
				(!strncmp(target, "TEST", strlen("TEST"))) ) 
		{
			CFeatures* features=set_features(py_ofeat);

			if (features && target)
			{
				if (!strncmp(target, "TRAIN", strlen("TRAIN")))
				{
					gui->guifeatures.set_train_features(features);
				}
				else if (!strncmp(target, "TEST", strlen("TEST")))
				{
					gui->guifeatures.set_test_features(features);
				}
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_features', 'TRAIN|TEST', features, ...)");
		}
		else
			CIO::message(M_ERROR, "usage is gf('set_features', 'TRAIN|TEST', features, ...)");
	}
	else
		CIO::message(M_ERROR, "set_features: Invalid parameters.\n");

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_add_features(PyObject* self, PyObject* args)
{
	PyObject   *py_ofeat = NULL;
	char* target = NULL;

	if (PyArg_ParseTuple(args, "sO", &target, &py_ofeat))
	{
		if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
				(!strncmp(target, "TEST", strlen("TEST"))) ) 
		{

			CFeatures* features=set_features(py_ofeat);

			if (features && target)
			{
				if (!strncmp(target, "TRAIN", strlen("TRAIN")))
				{
					gui->guifeatures.add_train_features(features);
				}
				else if (!strncmp(target, "TEST", strlen("TEST")))
				{
					gui->guifeatures.add_test_features(features);
				}
			}
			else
				CIO::message(M_ERROR, "usage is gf('add_features', 'TRAIN|TEST', features, ...)");
		}
		else
			CIO::message(M_ERROR, "set_features: Invalid parameters.\n");

	}

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_clean_features(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_labels(PyObject* self, PyObject* args)
{
	PyObject *arg = NULL;
	char* target = NULL;

	if (PyArg_ParseTuple(args, "sO", &target, &arg))
	{
		if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
				(!strncmp(target, "TEST", strlen("TEST"))) ) 
		{
			CLabels* labels=set_labels(arg);

			if (labels && target)
			{
				if (!strncmp(target, "TRAIN", strlen("TRAIN")))
					gui->guilabels.set_train_labels(labels);
				else if (!strncmp(target, "TEST", strlen("TEST")))
					gui->guilabels.set_test_labels(labels);
			}
			else
				CIO::message(M_ERROR, "usage is gf('set_labels', 'TRAIN|TEST', labels)");
		}
		else
			CIO::message(M_ERROR, "usage is gf('set_labels', 'TRAIN|TEST', labels)");
	}
	else
		CIO::message(M_ERROR, "set_labels: Invalid parameters.\n");

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_set_preproc_init(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_hmm_defs(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_one_class_hmm_classify(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_one_class_linear_hmm_classify(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_hmm_classify(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_one_class_hmm_classify_example(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_hmm_classify_example(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_svm_classify(PyObject* self, PyObject* args)
{
	CFeatures* f=gui->guifeatures.get_test_features();
	PyArrayObject* py_result = NULL;

	if (f && f->get_num_vectors())
	{
		int num_vec=f->get_num_vectors();

		CLabels* l=gui->guisvm.classify();

		if (l)
		{
			py_result = NA_NewArray(NULL, tFloat64, 1, num_vec);
			for (int i=0; i<num_vec; i++)
				NA_set1_Float64(py_result, i, l->get_label(i));
			delete l;
		}
	}

	return (PyObject*) py_result;
}

PyObject* CGUIPython::py_svm_classify_example(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_plugin_estimate(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_plugin_estimate(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_plugin_estimate_classify(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_plugin_estimate_classify_example(PyObject* self, PyObject* args)
{
	return NULL;
}

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
				CIO::message(M_ERROR, "py_test: arrays must have 1 dimension.\n");
		}
		else
			CIO::message(M_ERROR, "py_test: error converting array inputs.\n");
	}
	else
		CIO::message(M_ERROR, "py_test: Invalid parameters.\n");

	Py_XDECREF(py_afeat);
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


CFeatures* CGUIPython::set_features(PyObject* arg)
{
	PyArrayObject *py_afeat = NULL;
	CFeatures* features = NULL;

	/* Align, Byteswap, Contiguous, Typeconvert */
	switch (NA_NumarrayType(arg))
	{
		case tInt8:
		case tUInt8:
			py_afeat  = NA_InputArray(arg, tInt8, NUM_C_ARRAY);
			if ((py_afeat->nd == 2))
			{
				CHAR* feat= (CHAR*) NA_OFFSETDATA(py_afeat);
				int num_vec=py_afeat->dimensions[0];
				int num_feat=py_afeat->dimensions[1];

				if (feat)
				{
					// FIXME allow for other alphabets
					features= new CCharFeatures(DNA, (LONG) 0);
					CHAR* fm=new CHAR[num_vec*num_feat];
					assert(fm);
					for(int i=0; i<num_vec; i++)
					{
						for(int j=0; j<num_feat; j++)
							fm[i*num_feat+j]=feat[i*num_feat+j];
					}
					((CCharFeatures*) features)->set_feature_matrix(fm, num_feat, num_vec);
				}
				else
					CIO::message(M_ERROR,"empty feats ??\n");
			}
			else
				CIO::message(M_ERROR, "set_features: arrays must have 2 dimension.\n");
			break;
		case tFloat64:
			py_afeat  = NA_InputArray(arg, tFloat64, NUM_C_ARRAY);
			if ((py_afeat->nd == 2))
			{
				double* feat= (double*) NA_OFFSETDATA(py_afeat);
				int num_vec=py_afeat->dimensions[0];
				int num_feat=py_afeat->dimensions[1];

				if (feat)
				{
					features= new CRealFeatures((LONG) 0);
					REAL* fm=new REAL[num_vec*num_feat];
					assert(fm);
					for(int i=0; i<num_vec; i++)
					{
						for(int j=0; j<num_feat; j++)
							fm[i*num_feat+j]=feat[i*num_feat+j];
					}
					((CRealFeatures*) features)->set_feature_matrix(fm, num_feat, num_vec);
				}
				else
					CIO::message(M_ERROR,"empty feats ??\n");
			}
			else
				CIO::message(M_ERROR, "set_features: arrays must have 2 dimension.\n");
			break;
		default:
			CIO::message(M_ERROR, "Unknown nummarray type\n");
	};

	Py_XDECREF(py_afeat);
	return features;
}

CLabels* CGUIPython::set_labels(PyObject* arg)
{
	PyArrayObject *py_labels = NA_InputArray(arg, tFloat64, NUM_C_ARRAY);
	CLabels* labels = NULL;

	if (py_labels && (py_labels->nd == 1) && (py_labels->dimensions[0]>0))
	{
		labels=new CLabels(py_labels->dimensions[0]);
		Float64* lab= (double*) NA_OFFSETDATA(py_labels);

		for (int i=0; i<labels->get_num_labels(); i++)
			if (!labels->set_label(i, lab[i]))
				CIO::message(M_ERROR, "weirdo ! %d %d\n", labels->get_num_labels(), i);
	}
	Py_XDECREF(py_labels);
	return labels;
}
#endif
