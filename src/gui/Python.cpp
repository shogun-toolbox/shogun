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
#include "gui/Python.h"
#include "gui/TextGUI.h"

static CGUIPython sg_python;
extern CTextGUI* gui;

static PyMethodDef sg_pythonmethods[] = {
    {"send_command",  (CGUIPython::py_send_command), METH_VARARGS, "Send command to TextGUI."},
    {"system",  (CGUIPython::py_system), METH_VARARGS, "Execute a shell command."},
    {"help",  (CGUIPython::py_help), METH_VARARGS, "Print help message."},
    {"crc",  (CGUIPython::py_crc), METH_VARARGS, "Compute CRC checksum for string."},
    {"translate_string",  (CGUIPython::py_translate_string), METH_VARARGS, "Translate a string into higher order alphabet."},
    {"get_hmm",  (CGUIPython::py_get_hmm), METH_VARARGS, "Get HMM."},
    {"get_viterbi_path",  (CGUIPython::py_get_viterbi), METH_VARARGS, "Get viterbi path."},
    {"get_svm",  (CGUIPython::py_get_svm), METH_VARARGS, "Get SVM model"},
    {"get_kernel_init",  (CGUIPython::py_get_svm), METH_VARARGS, "Get kernel initialisation."},
    {"get_kernel_matrix",  (CGUIPython::py_get_kernel_matrix), METH_VARARGS, "Get the kernel matrix."},
    {"get_kernel_optimization",  (CGUIPython::py_get_kernel_optimization), METH_VARARGS, "Get kernel optimization (SVM normal vector)."},
    {"compute_by_subkernels",  (CGUIPython::py_compute_by_subkernels), METH_VARARGS, "Compute WD kernel for all subkernels."},
    {"set_subkernel_weights",  (CGUIPython::py_set_subkernels_weights), METH_VARARGS, "Set WD kernel weights."},
    {"set_last_subkernel_weights",  (CGUIPython::py_set_last_subkernel_weights), METH_VARARGS, "Set last subkernel weights."},
    {"set_wd_pos_weights",  (CGUIPython::py_wd_pos_weights), METH_VARARGS, "Set WD kernel position weights."},
    {"get_subkernel_weights",  (CGUIPython::py_get_subkernel_weights), METH_VARARGS, "Get subkernel weights."},
    {"get_last_subkernel_weights",  (CGUIPython::py_last_subkernel_weights), METH_VARARGS, "Get last subkernel weights."},
    {"get_wd_pos_weights",  (CGUIPython::py_wd_pos_weights), METH_VARARGS, "Get WD kernel position weights."},
    {"get_features",  (CGUIPython::py_get_features), METH_VARARGS, "Get features."},
    {"get_labels",  (CGUIPython::py_get_labels), METH_VARARGS, "Get labels."},
    {"get_version",  (CGUIPython::py_get_version), METH_VARARGS, "Get version."},
    {"get_preproc_init",  (CGUIPython::py_get_preproc_init), METH_VARARGS, "Get preproc init."},
    {"get_hmm_defs",  (CGUIPython::py_get_hmm_defs), METH_VARARGS, "Get HMM definitions."},
    {"set_hmm",  (CGUIPython::py_set_hmm), METH_VARARGS, "Set HMM."},
    {"model_prob_no_b_trans",  (CGUIPython::py_model_prob_no_b_trans), METH_VARARGS, "HMM model probability no b trans."},
    {"best_path_no_b_trans",  (CGUIPython::py_best_path_no_b_trans), METH_VARARGS, "HMM best path no b trans."},
    {"best_path_trans",  (CGUIPython::py_best_path_trans), METH_VARARGS, "HMM best path trans."},
    {"best_path_no_b",  (CGUIPython::py_best_path_no_b), METH_VARARGS, "HMM best path no b."},
    {"append_hmm",  (CGUIPython::py_append_hmm), METH_VARARGS, "Append HMM to current HMM model."},
    {"set_svm",  (CGUIPython::py_set_svm), METH_VARARGS, "Set SVM."},
    {"set_kernel_parameters",  (CGUIPython::py_kernel_parameters), METH_VARARGS, "Set kernel parameters."},
    {"set_custom_kernel",  (CGUIPython::py_set_custom_kernel), METH_VARARGS, "Set custom kernel matrix."},
    {"set_kernel_init",  (CGUIPython::py_set_kernel_init), METH_VARARGS, "Set kernel init."},
    {"set_features",  (CGUIPython::py_set_features), METH_VARARGS, "Set a feature object."},
    {"add_features",  (CGUIPython::py_add_features), METH_VARARGS, "Add another feature object."},
    {"set_labels",  (CGUIPython::py_set_labels), METH_VARARGS, "Set labels.."},
    {"set_preproc_init",  (CGUIPython::py_set_preproc_init), METH_VARARGS, "Set preprocessor init."},
    {"set_hmm_defs",  (CGUIPython::py_set_hmm_defs), METH_VARARGS, "Set HMM definitions."},
    {"one_class_hmm_classify",  (CGUIPython::py_one_class_hmm_classify), METH_VARARGS, "One class HMM classify."},
    {"one_class_linear_hmm_classify",  (CGUIPython::py_one_class_linear_hmm_classify), METH_VARARGS, "One class linear HMM classify."},
    {"hmm_classify",  (CGUIPython::py_hmm_classify), METH_VARARGS, "HMM classify."},
    {"one_class_hmm_classify_example",  (CGUIPython::py_one_class_hmm_classify_example), METH_VARARGS, "One class HMM classify example."},
    {"hmm_classify_example",  (CGUIPython::py_hmm_classify_example), METH_VARARGS, "HMM classify example."},
    {"svm_classify",  (CGUIPython::py_svm_classify), METH_VARARGS, "SVM classify."},
    {"svm_classify_example",  (CGUIPython::py_svm_classify_example), METH_VARARGS, "SVM classify example."},
    {"get_plugin_estimate",  (CGUIPython::py_get_plugin_estimate), METH_VARARGS, "Get plugin estimate."},
    {"set_plugin_estimate",  (CGUIPython::py_set_plugin_estimate), METH_VARARGS, "Set plugin estimate."},
    {"plugin_estimate_classify",  (CGUIPython::py_plugin_estimate_classify), METH_VARARGS, "Classify using plugin estimator."},
    {"plugin_estimate_classify_example",  (CGUIPython::py_plugin_estimate_classify_example), METH_VARARGS, "Classify example using plugin estimator."},
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
	CIO::message(M_INFO, "quitting...\n");
	delete gui;
}
#endif //HAVE_SWIG
