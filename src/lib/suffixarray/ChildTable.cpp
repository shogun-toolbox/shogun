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


// File    : sask/Code/ChildTable.cpp
//
// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created : 09 Feb 2006
//
// Updated : 24 Apr 2006

#ifndef CHILDTABLE_CPP
#define CHILDTABLE_CPP

#include "lib/io.h"
#include "lib/suffixarray/ChildTable.h"

/**
 *  Return the value of idx-th "up" field of child table. 
 *   val = childtab[idx -1];
 *
 *  \param idx - (IN)  The index of child table.
 *  \param val - (OUT) The value of idx-th entry in child table's "up" field.
 */
ErrorCode 
ChildTable::up(const UInt32 &idx, UInt32 &val){

	if(idx == size()) {
		// Special case: To get the first 0-index
		val = (*this)[idx-1];
		return NOERROR;
	}

  // svnvish: BUGBUG
  // Do we need to this in production code?
	UInt32 lcp_idx = 0, lcp_prev_idx = 0;
  lcp_idx = _lcptab[idx];
	lcp_prev_idx = _lcptab[idx-1];
  ASSERT(lcp_prev_idx > lcp_idx);
  val = (*this)[idx-1];

	return NOERROR;
}

/**
 *  Return the value of idx-th "down" field of child table.  Deprecated. 
 *    Instead use val = childtab[idx];
 *
 *  \param idx - (IN)  The index of child table.
 *  \param val - (OUT) The value of idx-th entry in child table's "down" field.
 */
ErrorCode 
ChildTable::down(const UInt32 &idx, UInt32 &val){
  
	// For a l-interval, l-[i..j], childtab[i].down == childtab[j+1].up
	// If l-[i..j] is last child-interval of its parent OR 0-[0..n], 
	//   childtab[i].nextlIndex == childtab[i].down

  // svnvish: BUGBUG
  // Do we need to this in production code?
	UInt32 lcp_idx = 0, lcp_nextidx = 0;
	lcp_nextidx = _lcptab[(*this)[idx]];
	lcp_idx = _lcptab[idx];
	ASSERT(lcp_nextidx > lcp_idx);

	// childtab[i].down := childtab[i].nextlIndex
	val = (*this)[idx];
	
	return NOERROR;
}


/**
 *  Return the first l-index of a given l-[i..j] interval.
 *
 *  \param i   - (IN)  Left bound of l-[i..j]
 *  \param j   - (IN)  Right bound of l-[i..j]
 *  \param idx - (OUT) The first l-index.
 */

ErrorCode 
ChildTable::l_idx(const UInt32 &i, const UInt32 &j, UInt32 &idx){
  
	UInt32 u = (*this)[j]; 
  
	if(i < u && u <= j){
		idx = u;
  }else {
		idx = (*this)[i]; 
	}									
	return NOERROR;
}


/**
 *  Dump array elements to output stream
 *
 *  \param os - (IN) Output stream.
 *  \param ct - (IN) ChildTable object.
 */
std::ostream& 
operator << (std::ostream& os, const ChildTable& ct){
  
  for( UInt32 i = 0; i < ct.size(); i++ ){
    os << "ct[ " << i << "]: " << ct[i] << std::endl;
  }
	return os;
}

#endif
