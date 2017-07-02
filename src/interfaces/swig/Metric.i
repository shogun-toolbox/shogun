/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg, 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

/* Remove C Prefix */
%rename(LMNNStatistics) CLMNNStatistics;
%rename(LMNN) CLMNN;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/metric/LMNN.h>
