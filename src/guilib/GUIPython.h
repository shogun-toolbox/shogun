/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIPYTHON_H_
#define __GUIPYTHON_H_

#include "lib/config.h"

#if defined(HAVE_PYTHON) && !defined(HAVE_SWIG)

extern "C" {
#include <Python.h>
}

#include "features/Features.h"
#include "features/Labels.h"

class CGUIPython
{
public:
	CGUIPython();
	~CGUIPython();

	static PyObject* py_send_command(PyObject* self, PyObject* args);
	static PyObject* py_system(PyObject* self, PyObject* args);
    static PyObject* py_help(PyObject* self, PyObject* args);
    static PyObject* py_crc(PyObject* self, PyObject* args);
    static PyObject* py_translate_string(PyObject* self, PyObject* args);
    static PyObject* py_get_hmm(PyObject* self, PyObject* args);
    static PyObject* py_get_viterbi(PyObject* self, PyObject* args);
    static PyObject* py_get_svm(PyObject* self, PyObject* args);
    static PyObject* py_get_kernel_matrix(PyObject* self, PyObject* args);
    static PyObject* py_get_kernel_optimization(PyObject* self, PyObject* args);
    static PyObject* py_compute_by_subkernels(PyObject* self, PyObject* args);
    static PyObject* py_set_subkernels_weights(PyObject* self, PyObject* args);
    static PyObject* py_set_last_subkernel_weights(PyObject* self, PyObject* args);
    static PyObject* py_wd_pos_weights(PyObject* self, PyObject* args);
    static PyObject* py_get_subkernel_weights(PyObject* self, PyObject* args);
    static PyObject* py_last_subkernel_weights(PyObject* self, PyObject* args);
    static PyObject* py_get_features(PyObject* self, PyObject* args);
    static PyObject* py_get_labels(PyObject* self, PyObject* args);
    static PyObject* py_get_version(PyObject* self, PyObject* args);
    static PyObject* py_get_preproc_init(PyObject* self, PyObject* args);
    static PyObject* py_get_hmm_defs(PyObject* self, PyObject* args);
    static PyObject* py_set_hmm(PyObject* self, PyObject* args);
    static PyObject* py_model_prob_no_b_trans(PyObject* self, PyObject* args);
    static PyObject* py_best_path_no_b_trans(PyObject* self, PyObject* args);
    static PyObject* py_best_path_trans(PyObject* self, PyObject* args);
    static PyObject* py_best_path_no_b(PyObject* self, PyObject* args);
    static PyObject* py_append_hmm(PyObject* self, PyObject* args);
    static PyObject* py_set_svm(PyObject* self, PyObject* args);
    static PyObject* py_kernel_parameters(PyObject* self, PyObject* args);
    static PyObject* py_set_custom_kernel(PyObject* self, PyObject* args);
    static PyObject* py_set_kernel_init(PyObject* self, PyObject* args);
    static PyObject* py_set_features(PyObject* self, PyObject* args);
    static PyObject* py_add_features(PyObject* self, PyObject* args);
    static PyObject* py_set_labels(PyObject* self, PyObject* args);
    static PyObject* py_set_preproc_init(PyObject* self, PyObject* args);
    static PyObject* py_set_hmm_defs(PyObject* self, PyObject* args);
    static PyObject* py_one_class_hmm_classify(PyObject* self, PyObject* args);
    static PyObject* py_one_class_linear_hmm_classify(PyObject* self, PyObject* args);
    static PyObject* py_hmm_classify(PyObject* self, PyObject* args);
    static PyObject* py_one_class_hmm_classify_example(PyObject* self, PyObject* args);
    static PyObject* py_hmm_classify_example(PyObject* self, PyObject* args);
    static PyObject* py_svm_classify(PyObject* self, PyObject* args);
    static PyObject* py_svm_classify_example(PyObject* self, PyObject* args);
    static PyObject* py_get_plugin_estimate(PyObject* self, PyObject* args);
    static PyObject* py_set_plugin_estimate(PyObject* self, PyObject* args);
    static PyObject* py_plugin_estimate_classify(PyObject* self, PyObject* args);
    static PyObject* py_plugin_estimate_classify_example(PyObject* self, PyObject* args);
    static PyObject* py_test(PyObject* self, PyObject* args);

    static CFeatures* set_features(PyObject* arg, char* args);
    static CLabels* set_labels(PyObject* arg);
};
#endif //HAVE_SWIG
#endif
