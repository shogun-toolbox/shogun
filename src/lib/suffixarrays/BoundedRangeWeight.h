/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is the Suffix Array based String Kernel.
 *
 * The Initial Developer of the Original Code is
 * Statistical Machine Learning Program (SML), National ICT Australia (NICTA).
 * Portions created by the Initial Developer are Copyright (C) 2006
 * the Initial Developer. All Rights Reserved.
 *
 * Contributor(s):
 *
 *   Choon Hui Teo <ChoonHui.Teo@rsise.anu.edu.au>
 *   S V N Vishwanathan <SVN.Vishwanathan@nicta.com.au>
 *
 * ***** END LICENSE BLOCK ***** */


// File    : sask/Code/BoundedRangeWeight.h
//
// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created : 09 Feb 2006
//
// Updated : 24 Apr 2006


#ifndef BOUNDEDRANGEWEIGHT_H
#define BOUNDEDRANGEWEIGHT_H

#include "DataType.h"
#include "ErrorCode.h"
#include "I_WeightFactory.h"


//' Bounded Range weight class
class BoundedRangeWeight : public I_WeightFactory
{

	int n;
public:

	/// Constructor
	BoundedRangeWeight(const int &n_=1): n(n_){}

	/// Destructor
	virtual ~BoundedRangeWeight(){}

	/// Compute weight
	ErrorCode ComputeWeight(const UInt32 &floor_len, const UInt32 &x_len, Real &weight);
};
#endif
