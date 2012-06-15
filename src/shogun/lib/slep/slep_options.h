/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  SLEP_OPTIONS_H_
#define  SLEP_OPTIONS_H_

#define IGNORE_IN_CLASSLIST

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
IGNORE_IN_CLASSLIST struct slep_options
{
	bool general;
	int termination;
	double tolerance;
	int max_iter;
	int restart_num;
	int n_nodes;
	int regularization;
	double* ind;
	double* G;
	double* initial_w;
	double q;

	static bool get_default_general() { return false; }
	static int get_default_termination() { return 2; }
	static double get_default_tolerance() { return 1e-3; }
	static int get_default_max_iter() { return 1000; }
	static int get_default_restart_num() { return 100; }
	static int get_default_regularization() { return 0; }
	static double get_default_q() { return 2.0; }
};
#endif
}
#endif   /* ----- #ifndef SLEP_OPTIONS_H_  ----- */


