/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GUIKNN_H__
#define _GUIKNN_H__ 

#include "classifier/KNN.h"
#include "features/Labels.h"

class CGUI;

class CGUIKNN
{

public:
	CGUIKNN(CGUI* g);
	~CGUIKNN();

	bool new_knn(CHAR* param);
	bool train(CHAR* param);
	bool test(CHAR* param);
	bool load(CHAR* param);
	bool save(CHAR* param);

 protected:
	CGUI* gui;
	CKNN* knn;
	int k;
};
#endif
