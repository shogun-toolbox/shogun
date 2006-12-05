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


// File    : sask/Code/W_kasai_lcp.cpp
//
// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created : 09 Feb 2006
//
// Updated : 24 Apr 2006


#ifndef W_KASAI_LCP_CPP
#define W_KASAI_LCP_CPP

#include "lib/suffixarray/W_kasai_lcp.h"
#include <vector>

/**
 *  Compute LCP array. Algorithm adapted from Manzini's SWAT2004 paper.
 *  Modification: array indexing changed from 1-based to 0-based.
 *
 *  \param text - (IN) The text which corresponds to SA.
 *  \param len  - (IN) Length of text.
 *  \param sa   - (IN) Suffix array.
 *  \param lcp  - (OUT) Computed LCP array.
 */

ErrorCode
W_kasai_lcp::ComputeLCP(const SYMBOL *text, const UInt32 &len, 
												const UInt32 *sa, LCP& lcp)
{  
  std::vector<UInt32>  isa(len);

	//' Step 1: Compute inverse suffix array
	for(UInt32 i=0; i<len; i++)	isa[sa[i]]=i;
  
	//' Step 2: Compute LCP values in O(n) time
	UInt32 h = 0;
  for(UInt32 i=0; i<len; i++) {
    
    UInt32 k = isa[i];
    if(k==0){
      
      //By definition lcp[0] = 0
      lcp.array[k] = 0;
      
    } else {
		
      UInt32 j = sa[k-1];
			while(i+h<len && j+h<len && text[i+h]==text[j+h]) h++;
			lcp.array[k] = h;
      
    }
		
    if(h>0) h--;
	}
  
	return NOERROR;
}
#endif
