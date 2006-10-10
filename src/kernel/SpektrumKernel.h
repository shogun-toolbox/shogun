///* ***** BEGIN LICENSE BLOCK *****
// * Version: MPL 1.1
// *
// * The contents of this file are subject to the Mozilla Public License Version
// * 1.1 (the "License"); you may not use this file except in compliance with
// * the License. You may obtain a copy of the License at
// * http://www.mozilla.org/MPL/
// *
// * Software distributed under the License is distributed on an "AS IS" basis,
// * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
// * for the specific language governing rights and limitations under the
// * License.
// *
// * The Original Code is the Suffix Array based String Kernel.
// *
// * The Initial Developer of the Original Code is
// * Statistical Machine Learning Program (SML), National ICT Australia (NICTA).
// * Portions created by the Initial Developer are Copyright (C) 2006
// * the Initial Developer. All Rights Reserved.
// *
// * Contributor(s):
// *
// *   Choon Hui Teo <ChoonHui.Teo@rsise.anu.edu.au>
// *   S V N Vishwanathan <SVN.Vishwanathan@nicta.com.au>
// *
// * ***** END LICENSE BLOCK ***** */
//
//
//// File    : sask/Code/StringKernel.h
////
//// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
////           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
////
//// Created : 09 Feb 2006
////
//// Updated : 24 Apr 2006
//
//
//#ifndef STRINGKERNEL_H
//#define STRINGKERNEL_H
//
//#include "DataType.h"
//#include "ErrorCode.h"
//#include "ESA.h"
//#include "I_SAFactory.h"
//#include "I_LCPFactory.h"
//#include "W_msufsort.h"
//#include "W_kasai_lcp.h"
//#include "I_WeightFactory.h"
//#include "ConstantWeight.h"
//#include "ExpDecayWeight.h"
//#include "BoundedRangeWeight.h"
//#include "KSpectrumWeight.h"
//
//
//
//using namespace std;
//
//class StringKernel {
//
//public:
//	/// Variables
//	ESA				  *esa;
//	I_WeightFactory	  *weigher;
//	Real              *val;  //' val array. Storing precomputed val(t) values.
//	Real			  *lvs;  //' leaves array. Storing weights for leaves.
//	
//
//	/// Constructors
//	StringKernel();
//
//	//' Given contructed suffix array
//	StringKernel(ESA *esa_);
//
//	//' Given text, build suffix array for it
//	StringKernel(const UInt32 &size, SYMBOL *text, int verb=INFO);
//
//
//	/// Destructor
//	virtual ~StringKernel();
//
//	//' Methods
//
//	/// Precompute the contribution of each intervals (or internal nodes)
//	ErrorCode PrecomputeVal();
//	
//	/// Compute Kernel matrix
//	ErrorCode Compute_K(SYMBOL *xprime, const UInt32 &xprime_len, Real &value);
//
//	/// Set leaves array, lvs[]
//	ErrorCode Set_Lvs(const Real *leafWeight, const UInt32 *len, const UInt32 &m);
//
//	/// Set leaves array as lvs[i]=i for i=0 to esa->length
//	ErrorCode Set_Lvs();
//
// private:
//	
//	/// An iterative auxiliary function used in PrecomputeVal()
//	ErrorCode IterativeCompute(const UInt32 &left, const UInt32 &right);
//
//};
//#endif
