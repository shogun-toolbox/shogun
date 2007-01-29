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


// File    : sask/Code/ESA.h
//
// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created : 09 Feb 2006
//
// Updated : 24 Apr 2006


#ifndef ESA_H
#define ESA_H

#include "lib/suffixarray/DataType.h"
#include "lib/suffixarray/ErrorCode.h"
#include "lib/suffixarray/LCP.h"
#include "lib/suffixarray/ChildTable.h"
#include "lib/suffixarray/I_LCPFactory.h"
#include "lib/suffixarray/I_SAFactory.h"
#include <vector>
#include <algorithm>


enum SA_VERB
{
	SA_DEBUG,
	SA_INFO,
	SA_ERROR
};

//#define SLINK


class ESA
{
  
 private:
 
  SA_VERB _verb;
  
 public:

  UInt32      size;            //' The length of #text#
	SYMBOL      *text;           //' Text corresponds to SA
  UInt32      *suftab;         //' Suffix Array
	LCP         lcptab;          //' LCP array
	ChildTable  childtab;        //' Child table (fields merged)
	UInt32      *suflink;        //' Suffix link table. Two fields: l,r
	

	//' --- for bucket table ---
	UInt32      bcktab_depth;    //' Number of char defining each bucket
	UInt32      bcktab_size;     //' size of bucket table
	UInt32      *bcktab_val;     //' value column of bucket table

	UInt32      *bcktab_key4;    //' 4-bytes key column of Bucket table
	UInt32      *coef4;
	UInt32      hash_value4;

	UInt64      *bcktab_key8;    //' 8-bytes key column of Bucket table
	UInt64      *coef8;
	UInt64      hash_value8;
	//' ---
  

	/// Constructors
	ESA(const UInt32 & size_, SYMBOL *text_, SA_VERB verb=SA_INFO);

	/// Destructor
	virtual ~ESA();

	/// Construct child table
	ErrorCode ConstructChildTable();


	/// Get suffix link interval
	ErrorCode GetSuflink(const UInt32 &i, const UInt32 &j,
											 UInt32 &sl_i, UInt32 &sl_j);


	/// Find the suffix link
	ErrorCode FindSuflink(const UInt32 &parent_i, const UInt32 &parent_j,
												const UInt32 &child_i, const UInt32 &child_j,
												UInt32 &sl_i, UInt32 &sl_j);
												
	/// Construct suffix link table
	ErrorCode ConstructSuflink();

	/// Construct bucket table
	ErrorCode ConstructBcktab(const UInt32 &alphabet_size=256);

	
	/// Get all non-singleton child-intervals
	ErrorCode GetChildIntervals(const UInt32 &lb, const UInt32 &rb, 
															std::vector<std::pair<UInt32,UInt32> > &q);

	/// Get intervals by index
	ErrorCode GetIntervalByIndex(const UInt32 &parent_i, const UInt32 &parent_j,
															 const UInt32 &start_idx, UInt32 &child_i, 
															 UInt32 &child_j);

	/// Get intervals by character
	ErrorCode GetIntervalByChar(const UInt32 &parent_i, const UInt32 &parent_j,
															const SYMBOL &start_ch, const UInt32 &depth,
															UInt32 &child_i, UInt32 &child_j);
	/// Get lcp value
	ErrorCode GetLcp(const UInt32 &i, const UInt32 &j, UInt32 &val);

	/// Compare pattern to text[suftab[idx]..length].
	ErrorCode Compare(const UInt32 &idx, const UInt32 &depth, SYMBOL *pattern, 
										const UInt32 &p_len, UInt32 &matched_len);

	/// Find longest substring of pattern in enhanced suffix array.
	ErrorCode Match(const UInt32 &i, const UInt32 &j, SYMBOL *pattern, const UInt32 p_len,
									UInt32 &lb, UInt32 &rb,	UInt32 &matched_len);

	/// Similar to Match() but returns also floor interval of [lb..rb]
	ErrorCode ExactSuffixMatch(const UInt32 &i, const UInt32 &j, const UInt32 &offset,
														 SYMBOL *pattern, const UInt32 p_len, UInt32 &lb, UInt32 &rb,	
														 UInt32 &matched_len, UInt32 &floor_lb, UInt32 &floor_rb,
														 UInt32 &floor_len);
	
};
#endif
