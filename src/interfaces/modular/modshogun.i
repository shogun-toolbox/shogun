/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Soeren Sonnenburg
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
%define DOCSTR
"The `modshogun` module gathers all modules available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) modshogun
#undef DOCSTR

%include "modshogun_ignores.i"

/* Documentation */
%feature("autodoc","0");

#ifdef SWIGPYTHON
#include <object.h>
%{
    static int print_sgobject(PyObject *pyobj, FILE *f, int flags);
%}

%feature("python:slot", "tp_str", functype="reprfunc") shogun::CSGObject::__str__;
%feature("python:slot", "tp_repr", functype="reprfunc") shogun::CSGObject::__repr__;
/*%feature("python:slot", "tp_hash", functype="hashfunc") shogun::CSGObject::myHashFunc;*/
%feature("python:tp_print") shogun::CSGObject "print_sgobject";
/*%feature("python:slot", "tp_as_buffer", functype="PyBufferProcs*") shogun::SGObject::tp_as_buffer;
%feature("python:slot", "bf_getbuffer", functype="getbufferproc") shogun::SGObject::getbuffer;*/
#endif

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "modshogun_doxygen.i"
#endif
#endif

%include "Classifier_includes.i"
%include "Clustering_includes.i"
%include "Distance_includes.i"
%include "Distribution_includes.i"
%include "Evaluation_includes.i"
%include "Features_includes.i"
%include "IO_includes.i"
%include "Kernel_includes.i"
%include "Library_includes.i"
%include "Mathematics_includes.i"
%include "ModelSelection_includes.i"
%include "Converter_includes.i"
%include "Preprocessor_includes.i"
%include "Regression_includes.i"
%include "Structure_includes.i"
%include "Multiclass_includes.i"

%include "SGBase.i"
%include "IO.i"
%include "Library.i"
%include "Mathematics.i"
%include "Features.i"
%include "Converter.i"
%include "Preprocessor.i"
%include "Evaluation.i"
%include "Distance.i"
%include "Kernel.i"
%include "Distribution.i"
%include "Classifier.i"
%include "Regression.i"
%include "Clustering.i"
%include "ModelSelection.i"
%include "Structure.i"
%include "Multiclass.i"
