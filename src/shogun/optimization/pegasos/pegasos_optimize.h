// Distributed under GNU General Public License (see license.txt for details).
//
//  Copyright (c) 2007 Shai Shalev-Shwartz.
//  All Rights Reserved.
//=============================================================================
// File Name: pegasos_optimize.h
// header for the main optimization function of pegasos
//=============================================================================

#ifndef _SHAI_PEGASOS_OPTIMIZE_H
#define _SHAI_PEGASOS_OPTIMIZE_H

#include <shogun/features/DotFeatures.h>
#include <shogun/lib/SGVector.h>

//*****************************************************************************
// Included Files
//*****************************************************************************
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cmath>

namespace shogun
{
class CPegasos
{
public:
static SGVector<float64_t> Learn(// Input variables
	   CDotFeatures* features,
	   SGVector<float64_t> labels,
	   int dimension,
	   double lambda,int max_iter,int exam_per_iter,int num_iter_to_avg,
	   // Output variables
	   double& obj_value, double& norm_value,double& loss_value,
	   // additional parameters
	   int eta_rule_type, double eta_constant,
	   int projection_rule, double projection_constant);
};
}
#endif
