/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#if defined(HAVE_PYTHON) && !defined(HAVE_SWIG)
#include "guilib/GUICommands.h"
#include "guilib/GUIPython.h"
#include "gui/Python.h"
#include "gui/TextGUI.h"

static CGUIPython sg_python;
extern CTextGUI* gui;

static PyMethodDef sg_pythonmethods[] = {
    {(char*) N_SEND_COMMAND,  (CGUIPython::py_send_command), METH_VARARGS, (char*) "Send command to TextGUI."},
    {(char*) N_EXEC,  (CGUIPython::py_system), METH_VARARGS, (char*) "Execute a shell command."},
    {(char*) N_HELP,  (CGUIPython::py_help), METH_VARARGS, (char*) "Print help message."},
    {(char*) N_CRC,  (CGUIPython::py_crc), METH_VARARGS, (char*) "Compute CRC checksum for string."},
    {(char*) N_TRANSLATE_STRING,  (CGUIPython::py_translate_string), METH_VARARGS, (char*) "Translate a string into higher order alphabet."},
    {(char*) N_GET_HMM,  (CGUIPython::py_get_hmm), METH_VARARGS, (char*) "Get HMM."},
    {(char*) N_GET_VITERBI_PATH,  (CGUIPython::py_get_viterbi), METH_VARARGS, (char*) "Get viterbi path."},
    {(char*) N_GET_SVM,  (CGUIPython::py_get_svm), METH_VARARGS, (char*) "Get SVM model"},
    {(char*) N_GET_KERNEL_INIT,  (CGUIPython::py_get_svm), METH_VARARGS, (char*) "Get kernel initialisation."},
    {(char*) N_GET_KERNEL_MATRIX,  (CGUIPython::py_get_kernel_matrix), METH_VARARGS, (char*) "Get the kernel matrix."},
    {(char*) N_GET_KERNEL_OPTIMIZATION,  (CGUIPython::py_get_kernel_optimization), METH_VARARGS, (char*) "Get kernel optimization (SVM normal vector)."},
    {(char*) N_COMPUTE_BY_SUBKERNELS,  (CGUIPython::py_compute_by_subkernels), METH_VARARGS, (char*) "Compute WD kernel for all subkernels."},
    {(char*) N_SET_SUBKERNEL_WEIGHTS,  (CGUIPython::py_set_subkernels_weights), METH_VARARGS, (char*) "Set WD kernel weights."},
    {(char*) N_SET_LAST_SUBKERNEL_WEIGHTS,  (CGUIPython::py_set_last_subkernel_weights), METH_VARARGS, (char*) "Set last subkernel weights."},
    {(char*) N_SET_WD_POS_WEIGHTS,  (CGUIPython::py_wd_pos_weights), METH_VARARGS, (char*) "Set WD kernel position weights."},
    {(char*) N_GET_SUBKERNEL_WEIGHTS,  (CGUIPython::py_get_subkernel_weights), METH_VARARGS, (char*) "Get subkernel weights."},
    {(char*) N_GET_LAST_SUBKERNEL_WEIGHTS,  (CGUIPython::py_last_subkernel_weights), METH_VARARGS, (char*) "Get last subkernel weights."},
    {(char*) N_GET_WD_POS_WEIGHTS,  (CGUIPython::py_wd_pos_weights), METH_VARARGS, (char*) "Get WD kernel position weights."},
    {(char*) N_GET_FEATURES,  (CGUIPython::py_get_features), METH_VARARGS, (char*) "Get features."},
    {(char*) N_GET_LABELS,  (CGUIPython::py_get_labels), METH_VARARGS, (char*) "Get labels."},
    {(char*) N_GET_VERSION,  (CGUIPython::py_get_version), METH_VARARGS, (char*) "Get version."},
    {(char*) N_GET_PREPROC_INIT,  (CGUIPython::py_get_preproc_init), METH_VARARGS, (char*) "Get preproc init."},
    {(char*) N_GET_HMM_DEFS,  (CGUIPython::py_get_hmm_defs), METH_VARARGS, (char*) "Get HMM definitions."},
    {(char*) N_SET_HMM,  (CGUIPython::py_set_hmm), METH_VARARGS, (char*) "Set HMM."},
    {(char*) N_MODEL_PROB_NO_B_TRANS,  (CGUIPython::py_model_prob_no_b_trans), METH_VARARGS, (char*) "HMM model probability no b trans."},
    {(char*) N_BEST_PATH_NO_B_TRANS,  (CGUIPython::py_best_path_no_b_trans), METH_VARARGS, (char*) "HMM best path no b trans."},
    {(char*) N_BEST_PATH_TRANS,  (CGUIPython::py_best_path_trans), METH_VARARGS, (char*) "HMM best path trans."},
    {(char*) N_BEST_PATH_NO_B,  (CGUIPython::py_best_path_no_b), METH_VARARGS, (char*) "HMM best path no b."},
    {(char*) N_APPEND_HMM,  (CGUIPython::py_append_hmm), METH_VARARGS, (char*) "Append HMM to current HMM model."},
    {(char*) N_SET_SVM,  (CGUIPython::py_set_svm), METH_VARARGS, (char*) "Set SVM."},
    {(char*) N_SET_CUSTOM_KERNEL,  (CGUIPython::py_set_custom_kernel), METH_VARARGS, (char*) "Set custom kernel matrix."},
    {(char*) N_SET_KERNEL_INIT,  (CGUIPython::py_set_kernel_init), METH_VARARGS, (char*) "Set kernel init."},
    {(char*) N_SET_FEATURES,  (CGUIPython::py_set_features), METH_VARARGS, (char*) "Set a feature object."},
    {(char*) N_ADD_FEATURES,  (CGUIPython::py_add_features), METH_VARARGS, (char*) "Add another feature object."},
    {(char*) N_SET_LABELS,  (CGUIPython::py_set_labels), METH_VARARGS, (char*) "Set labels.."},
    {(char*) N_SET_PREPROC_INIT,  (CGUIPython::py_set_preproc_init), METH_VARARGS, (char*) "Set preprocessor init."},
    {(char*) N_SET_HMM_DEFS,  (CGUIPython::py_set_hmm_defs), METH_VARARGS, (char*) "Set HMM definitions."},
    {(char*) N_ONE_CLASS_HMM_CLASSIFY,  (CGUIPython::py_one_class_hmm_classify), METH_VARARGS, (char*) "One class HMM classify."},
    {(char*) N_ONE_CLASS_LINEAR_HMM_CLASSIFY,  (CGUIPython::py_one_class_linear_hmm_classify), METH_VARARGS, (char*) "One class linear HMM classify."},
    {(char*) N_HMM_CLASSIFY,  (CGUIPython::py_hmm_classify), METH_VARARGS, (char*) "HMM classify."},
    {(char*) N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE,  (CGUIPython::py_one_class_hmm_classify_example), METH_VARARGS, (char*) "One class HMM classify example."},
    {(char*) N_HMM_CLASSIFY_EXAMPLE,  (CGUIPython::py_hmm_classify_example), METH_VARARGS, (char*) "HMM classify example."},
    {(char*) N_SVM_CLASSIFY,  (CGUIPython::py_svm_classify), METH_VARARGS, (char*) "SVM classify."},
    {(char*) N_SVM_CLASSIFY_EXAMPLE,  (CGUIPython::py_svm_classify_example), METH_VARARGS, (char*) "SVM classify example."},
    {(char*) N_GET_PLUGIN_ESTIMATE,  (CGUIPython::py_get_plugin_estimate), METH_VARARGS, (char*) "Get plugin estimate."},
    {(char*) N_SET_PLUGIN_ESTIMATE,  (CGUIPython::py_set_plugin_estimate), METH_VARARGS, (char*) "Set plugin estimate."},
    {(char*) N_PLUGIN_ESTIMATE_CLASSIFY,  (CGUIPython::py_plugin_estimate_classify), METH_VARARGS, (char*) "Classify using plugin estimator."},
    {(char*) N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE,  (CGUIPython::py_plugin_estimate_classify_example), METH_VARARGS, (char*) "Classify example using plugin estimator."},
    {(char*) "test",  (CGUIPython::py_test), METH_VARARGS, (char*) "Test."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* all dead
PyMODINIT_FUNC initsg(void)
{
	// initialize python interpreter
	Py_Initialize();

	// initialize threading (just in case it is needed)
	PyEval_InitThreads();

	// initialize textgui
	gui=new CTextGUI(0, NULL) ;

    // callback to cleanup at exit
	Py_AtExit(exitsg);

	// initialize callbacks
    Py_InitModule((char*) "sg", sg_pythonmethods);
}

void exitsg(void)
{
	SG_SINFO( "quitting...\n");
	delete gui;
}*/
#endif //HAVE_SWIG
