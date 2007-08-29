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
#include "guilib/GUICommands.h"
#include "guilib/GUIPython.h"
#include "gui/Python.h"
#include "gui/TextGUI.h"

static CGUIPython sg_python;
extern CTextGUI* gui;

static PyMethodDef sg_pythonmethods[] = {
    {N_SEND_COMMAND,  (CGUIPython::py_send_command), METH_VARARGS, "Send command to TextGUI."},
    {N_EXEC,  (CGUIPython::py_system), METH_VARARGS, "Execute a shell command."},
    {N_HELP,  (CGUIPython::py_help), METH_VARARGS, "Print help message."},
    {N_CRC,  (CGUIPython::py_crc), METH_VARARGS, "Compute CRC checksum for string."},
    {N_TRANSLATE_STRING,  (CGUIPython::py_translate_string), METH_VARARGS, "Translate a string into higher order alphabet."},
    {N_GET_HMM,  (CGUIPython::py_get_hmm), METH_VARARGS, "Get HMM."},
    {N_GET_VITERBI_PATH,  (CGUIPython::py_get_viterbi), METH_VARARGS, "Get viterbi path."},
    {N_GET_SVM,  (CGUIPython::py_get_svm), METH_VARARGS, "Get SVM model"},
    {N_GET_KERNEL_INIT,  (CGUIPython::py_get_svm), METH_VARARGS, "Get kernel initialisation."},
    {N_GET_KERNEL_MATRIX,  (CGUIPython::py_get_kernel_matrix), METH_VARARGS, "Get the kernel matrix."},
    {N_GET_KERNEL_OPTIMIZATION,  (CGUIPython::py_get_kernel_optimization), METH_VARARGS, "Get kernel optimization (SVM normal vector)."},
    {N_COMPUTE_BY_SUBKERNELS,  (CGUIPython::py_compute_by_subkernels), METH_VARARGS, "Compute WD kernel for all subkernels."},
    {N_SET_SUBKERNEL_WEIGHTS,  (CGUIPython::py_set_subkernels_weights), METH_VARARGS, "Set WD kernel weights."},
    {N_SET_LAST_SUBKERNEL_WEIGHTS,  (CGUIPython::py_set_last_subkernel_weights), METH_VARARGS, "Set last subkernel weights."},
    {N_SET_WD_POS_WEIGHTS,  (CGUIPython::py_wd_pos_weights), METH_VARARGS, "Set WD kernel position weights."},
    {N_GET_SUBKERNEL_WEIGHTS,  (CGUIPython::py_get_subkernel_weights), METH_VARARGS, "Get subkernel weights."},
    {N_GET_LAST_SUBKERNEL_WEIGHTS,  (CGUIPython::py_last_subkernel_weights), METH_VARARGS, "Get last subkernel weights."},
    {N_GET_WD_POS_WEIGHTS,  (CGUIPython::py_wd_pos_weights), METH_VARARGS, "Get WD kernel position weights."},
    {N_GET_FEATURES,  (CGUIPython::py_get_features), METH_VARARGS, "Get features."},
    {N_GET_LABELS,  (CGUIPython::py_get_labels), METH_VARARGS, "Get labels."},
    {N_GET_VERSION,  (CGUIPython::py_get_version), METH_VARARGS, "Get version."},
    {N_GET_PREPROC_INIT,  (CGUIPython::py_get_preproc_init), METH_VARARGS, "Get preproc init."},
    {N_GET_HMM_DEFS,  (CGUIPython::py_get_hmm_defs), METH_VARARGS, "Get HMM definitions."},
    {N_SET_HMM,  (CGUIPython::py_set_hmm), METH_VARARGS, "Set HMM."},
    {N_MODEL_PROB_NO_B_TRANS,  (CGUIPython::py_model_prob_no_b_trans), METH_VARARGS, "HMM model probability no b trans."},
    {N_BEST_PATH_NO_B_TRANS,  (CGUIPython::py_best_path_no_b_trans), METH_VARARGS, "HMM best path no b trans."},
    {N_BEST_PATH_TRANS,  (CGUIPython::py_best_path_trans), METH_VARARGS, "HMM best path trans."},
    {N_BEST_PATH_NO_B,  (CGUIPython::py_best_path_no_b), METH_VARARGS, "HMM best path no b."},
    {N_APPEND_HMM,  (CGUIPython::py_append_hmm), METH_VARARGS, "Append HMM to current HMM model."},
    {N_SET_SVM,  (CGUIPython::py_set_svm), METH_VARARGS, "Set SVM."},
    {N_SET_CUSTOM_KERNEL,  (CGUIPython::py_set_custom_kernel), METH_VARARGS, "Set custom kernel matrix."},
    {N_SET_KERNEL_INIT,  (CGUIPython::py_set_kernel_init), METH_VARARGS, "Set kernel init."},
    {N_SET_FEATURES,  (CGUIPython::py_set_features), METH_VARARGS, "Set a feature object."},
    {N_ADD_FEATURES,  (CGUIPython::py_add_features), METH_VARARGS, "Add another feature object."},
    {N_SET_LABELS,  (CGUIPython::py_set_labels), METH_VARARGS, "Set labels.."},
    {N_SET_PREPROC_INIT,  (CGUIPython::py_set_preproc_init), METH_VARARGS, "Set preprocessor init."},
    {N_SET_HMM_DEFS,  (CGUIPython::py_set_hmm_defs), METH_VARARGS, "Set HMM definitions."},
    {N_ONE_CLASS_HMM_CLASSIFY,  (CGUIPython::py_one_class_hmm_classify), METH_VARARGS, "One class HMM classify."},
    {N_ONE_CLASS_LINEAR_HMM_CLASSIFY,  (CGUIPython::py_one_class_linear_hmm_classify), METH_VARARGS, "One class linear HMM classify."},
    {N_HMM_CLASSIFY,  (CGUIPython::py_hmm_classify), METH_VARARGS, "HMM classify."},
    {N_ONE_CLASS_HMM_CLASSIFY_EXAMPLE,  (CGUIPython::py_one_class_hmm_classify_example), METH_VARARGS, "One class HMM classify example."},
    {N_HMM_CLASSIFY_EXAMPLE,  (CGUIPython::py_hmm_classify_example), METH_VARARGS, "HMM classify example."},
    {N_SVM_CLASSIFY,  (CGUIPython::py_svm_classify), METH_VARARGS, "SVM classify."},
    {N_SVM_CLASSIFY_EXAMPLE,  (CGUIPython::py_svm_classify_example), METH_VARARGS, "SVM classify example."},
    {N_GET_PLUGIN_ESTIMATE,  (CGUIPython::py_get_plugin_estimate), METH_VARARGS, "Get plugin estimate."},
    {N_SET_PLUGIN_ESTIMATE,  (CGUIPython::py_set_plugin_estimate), METH_VARARGS, "Set plugin estimate."},
    {N_PLUGIN_ESTIMATE_CLASSIFY,  (CGUIPython::py_plugin_estimate_classify), METH_VARARGS, "Classify using plugin estimator."},
    {N_PLUGIN_ESTIMATE_CLASSIFY_EXAMPLE,  (CGUIPython::py_plugin_estimate_classify_example), METH_VARARGS, "Classify example using plugin estimator."},
    {"test",  (CGUIPython::py_test), METH_VARARGS, "Test."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


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
    Py_InitModule("sg", sg_pythonmethods);
}

void exitsg(void)
{
	SG_SINFO( "quitting...\n");
	delete gui;
}
#endif //HAVE_SWIG
