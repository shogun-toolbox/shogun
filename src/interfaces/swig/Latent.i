/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg, 2012 Viktor Gal
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::CDirectorLatentModel;
#endif

/* Remove C Prefix */
%rename(LatentModel) CLatentModel;

%rename(LinearLatentMachine) CLinearLatentMachine;

%rename(LatentSVM) CLatentSVM;

%rename(DirectorLatentModel) CDirectorLatentModel;


/* Include Class Headers to make them visible from within the target language */
%include <shogun/latent/LatentModel.h>

%include <shogun/latent/DirectorLatentModel.h>

%include <shogun/machine/LinearLatentMachine.h>

%include <shogun/latent/LatentSVM.h>
