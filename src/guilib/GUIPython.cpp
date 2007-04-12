/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#if defined(HAVE_PYTHON) && !defined(HAVE_SWIG)
#include "guilib/GUIPython.h"
#include "gui/TextGUI.h"
#include "base/Version.h"
#include "lib/common.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "features/CharFeatures.h"
#include "kernel/WeightedDegreeStringKernel.h"
#include "kernel/WeightedDegreePositionStringKernel.h"
#include "kernel/CombinedKernel.h"
#include "kernel/CustomKernel.h"

#include "lib/python.h"

extern "C" {
#include <object.h>
#include <../../numarray/numpy/libnumarray.h>
#include <numpy/ndarrayobject.h>
}

extern CTextGUI* gui;

CGUIPython::CGUIPython() : CSGObject()
{
	import_libnumarray();
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
	gui->parse_line("help");

	Py_INCREF(Py_None);
	return Py_None;
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

		SG_DEBUG("num_sv: %d\n", svm->get_num_support_vectors());

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

	if (!k || !k->get_lhs() || !k->get_rhs())
		SG_ERROR( "no kernel set\n");
	else
	{
		//result=k->get_kernel_matrix_real(m,n,NULL);
		
		m = k->get_lhs()->get_num_vectors();	
		n = k->get_rhs()->get_num_vectors();
		result = new DREAL[m*n];
		ASSERT(result);

		for (INT i=0; i<m; i++)
			for (INT j=0; j<n; j++)
				result[i*n+j]=k->kernel(i,j);
		
	}

	if(result)
		return (PyObject*) NA_NewArray(result, tFloat64, 2, m,n);
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
	PyObject* py_oweights = NULL;

	if (PyArg_ParseTuple(args, "O", &py_oweights))
	{
		PyArrayObject* py_weights = NA_InputArray(py_oweights, tFloat64, NUM_C_ARRAY);
		CKernel* k= gui->guikernel.get_kernel();

		if (k && py_weights)
		{
			double* w= (double*) NA_OFFSETDATA(py_weights);

			if (k->get_kernel_type() == K_WEIGHTEDDEGREE)
			{
				CWeightedDegreeStringKernel *kernel = (CWeightedDegreeStringKernel *) k;
				INT degree = kernel->get_degree() ;

				if ((py_weights->nd == 1 && py_weights->dimensions[0] == degree) ||
						(py_weights->nd == 2 && py_weights->dimensions[0] == degree))
				{
					if (py_weights->nd == 1)
						kernel->set_weights(w, py_weights->dimensions[0], 0);
					else
						kernel->set_weights(w, py_weights->dimensions[0], py_weights->dimensions[1]);

					Py_INCREF(Py_None);
					return Py_None;
				}
				else
					SG_ERROR( "dimension mismatch (should be de(seq_length | 1) x degree)\n") ;

			}
			else if (k->get_kernel_type() == K_WEIGHTEDDEGREEPOS)
			{
				CWeightedDegreePositionStringKernel *kernel = (CWeightedDegreePositionStringKernel *) k;
				INT degree = kernel->get_degree() ;

				if ((py_weights->nd == 1 && py_weights->dimensions[0] == degree) ||
						(py_weights->nd == 2 && py_weights->dimensions[0] == degree))
				{
					if (py_weights->nd == 1)
						kernel->set_weights(w, py_weights->dimensions[0], 0);
					else
						kernel->set_weights(w, py_weights->dimensions[0], py_weights->dimensions[1]);

					Py_INCREF(Py_None);
					return Py_None;
				}
				else
					SG_ERROR( "dimension mismatch (should be de(seq_length | 1) x degree)\n") ;
			}
			else
			{
				INT num_subkernels = k->get_num_subkernels() ;
				if (py_weights->nd == 1 && py_weights->dimensions[0]==num_subkernels)
				{
					k->set_subkernel_weights(w, py_weights->dimensions[0]);
					Py_INCREF(Py_None);
					return Py_None;
				}
				else
					SG_ERROR( "dimension mismatch (should be 1 x num_subkernels)\n") ;
			}

		}
	}
	else
		SG_ERROR( "expected double array");

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
			CWeightedDegreeStringKernel *kernel = (CWeightedDegreeStringKernel *) k;

			const DREAL* weights = kernel->get_degree_weights(degree, length) ;
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
			CWeightedDegreePositionStringKernel *kernel = (CWeightedDegreePositionStringKernel *) k;

			const DREAL* weights = kernel->get_degree_weights(degree, length) ;
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
			const DREAL* weights = kernel->get_subkernel_weights(num_weights) ;

			PyArrayObject* py_weights=NA_NewArray(NULL, tFloat64, 1, num_weights);

			for (int i=0; i<num_weights; i++)
				NA_set1_Float64(py_weights, i, weights[i]);

			return (PyObject*) py_weights;
		}
		else
			SG_ERROR( "kernel does not have any subkernel weights\n");
	}
	else
		SG_ERROR( "no kernel set\n");

	return NULL;
}

PyObject* CGUIPython::py_last_subkernel_weights(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_get_features(PyObject* self, PyObject* args)
{
	char* target = NULL;

	if (PyArg_ParseTuple(args, "s", &target))
	{
		CFeatures* f = NULL;

		if (!strncmp(target, "TRAIN", strlen("TRAIN")))
		{
			f=gui->guifeatures.get_train_features();
		}
		else if (!strncmp(target, "TEST", strlen("TEST")))
		{
			f=gui->guifeatures.get_test_features();
		}
		else
			SG_ERROR( "target is TRAIN|TEST");

		if (f)
		{
			switch (f->get_feature_class())
			{
				case C_SIMPLE:
					switch (f->get_feature_type())
					{
						case F_DREAL:
							{
								INT num_vec= ((CRealFeatures*) f)->get_num_vectors();
								INT num_feat= ((CRealFeatures*) f)->get_num_features();
								Float64* feat=new Float64[num_vec*num_feat];

								if (feat)
								{
									for (INT i=0; i<num_vec; i++)
									{
										INT len=0;
										bool free_vec;
										DREAL* vec=((CRealFeatures*) f)->get_feature_vector(i, len, free_vec);
										ASSERT(len==num_feat);
										for (INT j=0; j<num_feat; j++)
											feat[((CRealFeatures*) f)->get_num_vectors()*j+i]= (double) vec[j];
										((CRealFeatures*) f)->free_feature_vector(vec, i, free_vec);
									}
									return (PyObject*) NA_NewArray(feat, tFloat64, 2, num_vec, num_feat);
								}
							}
							break;
						case F_WORD:
						case F_SHORT:
						case F_CHAR:
						case F_BYTE:
						default:
							SG_ERROR( "not implemented\n");
					}
					break;
				case C_SPARSE:
				case C_STRING:
				default:
					SG_ERROR( "not implemented\n");
			}
		}

	}
	return NULL;
}

PyObject* CGUIPython::py_get_labels(PyObject* self, PyObject* args)
{
	char* target = NULL;

	if (PyArg_ParseTuple(args, "s", &target))
	{
		CLabels* labels = NULL;

		if (!strncmp(target, "TRAIN", strlen("TRAIN")))
		{
			labels=gui->guilabels.get_train_labels();
		}
		else if (!strncmp(target, "TEST", strlen("TEST")))
		{
			labels=gui->guilabels.get_test_labels();
		}
		else
			SG_ERROR( "target is TRAIN|TEST");

		Float64 *result=NULL;
		int len=0;
		if (labels)
		{
			result=labels->get_labels(len);

			if(result)
				return (PyObject*) NA_NewArray(result, tFloat64, 1, 1, len);
		}
	}
	return NULL;
}

PyObject* CGUIPython::py_get_version(PyObject* self, PyObject* args)
{
	return PyLong_FromUnsignedLong(version.get_version_revision());
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
	CSVM* svm=gui->guisvm.get_svm();

	if (svm)
	{
		PyObject* py_dict = NULL;
		if (!PyArg_ParseTuple(args, "O", &py_dict))
			return NULL;
		if (!PyDict_Check(py_dict))
			return NULL;

		PyObject* py_oalphas = PyDict_GetItemString(py_dict, "alphas");
		PyObject* py_osv_idx = PyDict_GetItemString(py_dict, "sv_idx");
		PyObject* py_bias = PyDict_GetItemString(py_dict, "b");

		if (py_oalphas && py_osv_idx && py_bias)
		{
			PyArrayObject* py_alphas = NA_InputArray(py_oalphas, tFloat64, NUM_C_ARRAY);
			PyArrayObject* py_sv_idx = NA_InputArray(py_oalphas, tInt64, NUM_C_ARRAY);

			if (py_alphas && py_sv_idx && py_bias && (py_alphas->nd == 1) && (py_alphas->dimensions[0] > 0) && NA_ShapeEqual(py_alphas, py_sv_idx))
			{
				svm->create_new_model(py_alphas->dimensions[0]);
				svm->set_bias(PyFloat_AsDouble(py_bias));

				for (int i=0; i< svm->get_num_support_vectors(); i++)
				{
					svm->set_alpha(i, NA_get_Float64(py_alphas,i));
					svm->set_support_vector(i, (int) NA_get1_Int64(py_sv_idx,i));
				}

				if (!PyErr_Occurred())
				{
					Py_INCREF(Py_None);
					return Py_None;
				}
			}
			else
				SG_ERROR( "no svm object available\n") ;

			Py_XDECREF(py_alphas);
			Py_XDECREF(py_sv_idx);
			Py_XDECREF(py_bias);
		}
		Py_XDECREF(py_dict);
	}

	return NULL;
}

PyObject* CGUIPython::py_kernel_parameters(PyObject* self, PyObject* args)
{
	return NULL;
}

PyObject* CGUIPython::py_set_custom_kernel(PyObject* self, PyObject* args)
{
	PyObject* py_okernel = NULL;
	PyArrayObject* py_akernel = NULL;
	char* target = NULL;
	bool source_is_diag = false;
	bool dest_is_diag = false;


	if (PyArg_ParseTuple(args, "Os", &py_okernel, target))
	{
		if ( (!strncmp(target, "DIAG", strlen("DIAG"))) || 
				(!strncmp(target, "FULL", strlen("FULL"))) ) 
		{
			if (!strncmp(target, "FULL2DIAG", strlen("FULL2DIAG")))
			{
				source_is_diag = false;
				dest_is_diag = true;
			}
			else if (!strncmp(target, "FULL", strlen("FULL")))
			{
				source_is_diag = false;
				dest_is_diag = false;
			}
			else if (!strncmp(target, "DIAG", strlen("DIAG")))
			{
				source_is_diag = true;
				dest_is_diag = true;
			}

			py_akernel  = NA_InputArray(py_okernel, tFloat64, NUM_C_ARRAY);

			if (py_akernel)
			{
				double* km= (double*) NA_OFFSETDATA(py_akernel);
				CCustomKernel* k=(CCustomKernel*)gui->guikernel.get_kernel();
				if  (k && k->get_kernel_type() == K_COMBINED)
				{
					SG_DEBUG( "identified combined kernel\n") ;
					k = (CCustomKernel*)((CCombinedKernel*)k)->get_last_kernel() ;
				}

				if (k && k->get_kernel_type() == K_CUSTOM)
				{
					if (source_is_diag && dest_is_diag && (py_akernel->nd == 2 && py_akernel->dimensions[0] == py_akernel->dimensions[1]) )
					{
						if (k->set_diag_kernel_matrix_from_diag(km, py_akernel->dimensions[0]))
						{
							Py_INCREF(Py_None);
							return Py_None;
						}
					}
					else if (!source_is_diag && dest_is_diag && (py_akernel->nd == 2 && py_akernel->dimensions[0] == py_akernel->dimensions[1]) ) 
					{
						if (k->set_diag_kernel_matrix_from_full(km, py_akernel->dimensions[0]))
						{
							Py_INCREF(Py_None);
							return Py_None;
						}
					}
					else if (!source_is_diag && !dest_is_diag)
					{
						if (k->set_full_kernel_matrix_from_full(km, py_akernel->dimensions[0], py_akernel->dimensions[1]))
						{
							Py_INCREF(Py_None);
							return Py_None;
						}
					}
					else
						SG_ERROR("not defined / general error\n");
				}
				else
					SG_ERROR( "not a custom kernel\n") ;
			}
			else
				SG_ERROR("kernel matrix must by given as double matrix\n");
		}
		else
			SG_ERROR( "usage is sg('set_custom_kernel',[kernelmatrix, is_upperdiag])");
	}
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
	char* cmdline = NULL;

	if ( (PyArg_ParseTuple(args, "sO", &target, &py_ofeat)) ||
			(PyArg_ParseTuple(args, "sOs", &target, &py_ofeat, &cmdline)) )
	{
		if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
				(!strncmp(target, "TEST", strlen("TEST"))) ) 
		{
			CFeatures* features=set_features(py_ofeat, cmdline);

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
				SG_ERROR( "usage is sg('set_features', 'TRAIN|TEST', features, ...)");
		}
		else
			SG_ERROR( "usage is sg('set_features', 'TRAIN|TEST', features, ...)");
	}
	else
		SG_ERROR( "set_features: Invalid parameters.\n");

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_add_features(PyObject* self, PyObject* args)
{
	PyObject   *py_ofeat = NULL;
	char* target = NULL;
	char* cmdline = NULL;

	if (PyArg_ParseTuple(args, "sOs", &target, &py_ofeat, &cmdline) ||
            PyArg_ParseTuple(args, "sO", &target, &py_ofeat))
	{
		if ( (!strncmp(target, "TRAIN", strlen("TRAIN"))) || 
				(!strncmp(target, "TEST", strlen("TEST"))) ) 
		{

			CFeatures* features=set_features(py_ofeat, cmdline);

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
				SG_ERROR( "usage is sg('add_features', 'TRAIN|TEST', features, ...)");
		}
		else
			SG_ERROR( "set_features: Invalid parameters.\n");

	}

	Py_INCREF(Py_None);
	return Py_None;
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
				SG_ERROR( "usage is sg('set_labels', 'TRAIN|TEST', labels)");
		}
		else
			SG_ERROR( "usage is sg('set_labels', 'TRAIN|TEST', labels)");
	}
	else
		SG_ERROR( "set_labels: Invalid parameters.\n");

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
			ASSERT(py_result)
			for (int i=0; i<num_vec; i++)
				NA_set1_Float64(py_result, i, l->get_label(i));
			delete l;
		}
	}

	return (PyObject*) py_result;
}

PyObject* CGUIPython::py_svm_classify_example(PyObject* self, PyObject* args)
{
	int idx = 0;

	if (PyArg_ParseTuple(args, "i", &idx))
	{
		DREAL result;
		if (!gui->guisvm.classify_example(idx, result))
			SG_ERROR( "svm_classify_example failed\n") ;
		else
			return PyFloat_FromDouble(result);
	}

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
						SG_PRINT( "%f\n", feat[i]);
				}
				else
					SG_ERROR("empty feats ??\n");
			}
			else
				SG_ERROR( "py_test: arrays must have 1 dimension.\n");
		}
		else
			SG_ERROR( "py_test: error converting array inputs.\n");
	}
	else
		SG_ERROR( "py_test: Invalid parameters.\n");

	Py_XDECREF(py_afeat);
	Py_INCREF(Py_None);
	return Py_None;
}

//static PyObject * sgpython_system(PyObject *self, PyObject *args)
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

CFeatures* CGUIPython::set_features(PyObject* arg, char* args)
{
	PyArrayObject *py_afeat = NULL;
	CFeatures* features = NULL;
	int typeno=-1;
	/* Align, Byteswap, Contiguous, Typeconvert */

	if (!NA_NumArrayCheck(arg) && !NA_NDArrayCheck(arg))
	{
		SG_ERROR( "no numpy type\n");
		return NULL;
	}

	typeno=PyArray(arg)->descr->type_num;
	//strangely this seems to be a CharacterArray...
	if (NA_NDArrayCheck(arg) && PyArray(arg)->descr->type_num == PyArray_NOTYPE)
		typeno=tUInt8;
	//SG_DEBUG("%d (%d) vs. %d : tInt8(%d),tUInt8(%d),tUInt32(%d),tInt32(%d)\n", typeno, PyArray_NOTYPE, NA_NumarrayType(arg), tInt8, tUInt8,tUInt32,tInt32);

	switch (typeno)
	{
		case tInt8:
		case tUInt8:
			py_afeat  = NA_InputArray(arg, tUInt8, NUM_C_ARRAY);
			if (NA_NDArrayCheck(arg))

			{
				if ((py_afeat->nd == 1))
				{
					CHAR* feat= (CHAR*) NA_OFFSETDATA(py_afeat);
					int num_vec=py_afeat->dimensions[0];
					int num_feat=0;
					//int num_feat=PyArray(py_afeat)->itemsize;
					SG_DEBUG( "vec: %d dim:%d\n", num_vec, num_feat);
					if (feat)
					{
						if (args)
						{
							CAlphabet* alphabet=new CAlphabet(args, strlen(args));
							features= new CCharFeatures(alphabet, 0);
							CHAR* fm=new CHAR[num_vec*num_feat];
							ASSERT(fm);
							for(int i=0; i<num_vec; i++)
							{
								for(int j=0; j<num_feat; j++)
								{
									fm[j*num_vec+i]=feat[i*num_feat+j];
								}

							}
							((CCharFeatures*) features)->set_feature_matrix(fm, num_vec, num_feat);
						}
						else
							SG_ERROR( "please specify alphabet!\n");
					}
					else
						SG_ERROR("empty feats ??\n");
				}
			}
			else
			{
				if ((py_afeat->nd == 2))
				{
					CHAR* feat= (CHAR*) NA_OFFSETDATA(py_afeat);
					int num_vec=py_afeat->dimensions[0];
					int num_feat=py_afeat->dimensions[1];
					if (feat)
					{
						CAlphabet* alpha = new CAlphabet(args, strlen(args));
						features= new CCharFeatures(alpha, 0);
						CHAR* fm=new CHAR[num_vec*num_feat];
						ASSERT(fm);
						for(int i=0; i<num_vec; i++)
						{
							for(int j=0; j<num_feat; j++)
							{
								fm[j*num_vec+i]=feat[i*num_feat+j];
							}
						}
						((CCharFeatures*) features)->set_feature_matrix(fm, num_vec, num_feat);

					}
					else
						SG_ERROR("empty feats ??\n");
				}
				else
					SG_ERROR( "set_features: arrays must have 2 dimension.\n");
			}
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
					features= new CRealFeatures(0);
					DREAL* fm=new DREAL[num_vec*num_feat];
					ASSERT(fm);

					for(int i=0; i<num_vec; i++)
					{
						for(int j=0; j<num_feat; j++)
						{
							fm[j*num_vec+i]=feat[i*num_feat+j];
						}

					}
					((CRealFeatures*) features)->set_feature_matrix(fm, num_vec, num_feat);

				}
				else
					SG_ERROR("empty feats ??\n");
			}
			else
				SG_ERROR( "set_features: arrays must have 2 dimension.\n");
			break;
		default:
			SG_ERROR( "Unknown numpy type\n");
	}

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
				SG_ERROR( "weirdo ! %d %d\n", labels->get_num_labels(), i);
	}
	Py_XDECREF(py_labels);
	return labels;
}
#endif
