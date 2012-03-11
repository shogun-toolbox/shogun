/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(MulticlassLibLinear) CMulticlassLibLinear;
%rename(MulticlassOCAS) CMulticlassOCAS;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/multiclass/MulticlassLibLinear.h>
%include <shogun/multiclass/MulticlassOCAS.h>

