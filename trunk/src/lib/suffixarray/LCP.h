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


// File    : sask/Code/LCP.h
//
// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created : 09 Feb 2006
//
// Updated : 24 Apr 2006


#ifndef LCP_H
#define LCP_H

#include "lib/suffixarray/DataType.h"
#include "lib/suffixarray/ErrorCode.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <iostream>

/**
 *  LCP array class
 */

class LCP
{
 private:
	/// Compacted array
  std::vector<Byte1>  _p_array;
  std::vector<UInt32> _idx_array;
  std::vector<UInt32> _val_array;

  UInt32 _size;
  
  bool _is_compact;
  
  typedef std::vector<UInt32>::const_iterator const_itr;
  
  const_itr _beg;
  const_itr _end;

	const_itr _cache;
	UInt32 _dist;
	
 public:

  /// Original array - 4bytes
  std::vector<UInt32>  array;    
	
	/// Constructors
	LCP(const UInt32 &size);

	/// Destructors
	virtual ~LCP();

	/// Methods

	/// Compact 4n bytes array into (1n+8p) bytes arrays
	ErrorCode compact(void);
  
	/// Retrieve lcp array value
	// ErrorCode lcp(const UInt32 &idx, UInt32 &value);
	
  UInt32 operator[] (const UInt32& idx); 
  
  friend std::ostream& operator << (std::ostream& os, LCP& lcp);
  
};
#endif
