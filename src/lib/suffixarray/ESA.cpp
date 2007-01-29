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


// File    : sask/Code/ESA.cpp
//
// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created : 09 Feb 2006
//
// Updated : 24 Apr 2006


#ifndef ESA_CPP
#define ESA_CPP

#include "lib/io.h"
#include "lib/suffixarray/ESA.h"
#include "lib/suffixarray/W_msufsort.h"
#include "lib/suffixarray/W_kasai_lcp.h"

#include <vector>
#include <stack>
#include <queue>
#include <algorithm> 
#include <numeric>
#include <fstream>
#include <ctime>

#define MIN(x,y) ((x) < (y)) ? (x):(y)
#define showtext(i,len) std::cout << "'";                      \
                        for(UInt32 cnt=0; cnt<(len); cnt++)    \
                          std::cout<<text[(i)+cnt];            \
                        std::cout << "'" << std::endl;         


ESA::ESA(const UInt32 & size_, SYMBOL *text_, SA_VERB verb): 
  _verb(verb),
  size(size_), 
  text(text_), 
  suftab(0),
  lcptab(size_),
  childtab(size_, lcptab),
  suflink(0),
	bcktab_depth(0),
	bcktab_size(0),
	bcktab_val(0),
	bcktab_key4(0),
	coef4(0),
	bcktab_key8(0),
	coef8(0)
{  

  ErrorCode ec;
  I_SAFactory* sa_fac = 0;
  I_LCPFactory* lcp_fac = 0;

	//' input validation
	ASSERT(size > 0);
	ASSERT(text[size-1] == SENTINEL);


	//' Construct Suffix Array
	if(!sa_fac){
    sa_fac = new W_msufsort();
	}
  
	suftab = new UInt32[size];

	ec = sa_fac->ConstructSA(text, size, suftab); CHECKERROR(ec);
	if(sa_fac) { delete sa_fac; sa_fac = NULL; }

  if(_verb == SA_DEBUG){
    for(UInt32 kk=0; kk<size; kk++)
      std::cout << "SA:["<<kk<<"]:"<< &text[suftab[kk]]<<std::endl;
  }

	//' Compute LCP array
	if(!lcp_fac){
		lcp_fac = new W_kasai_lcp();
	}
	ec = lcp_fac->ComputeLCP(text, size, suftab, lcptab); CHECKERROR(ec);
	if(lcp_fac) { delete lcp_fac; lcp_fac = NULL; }

	//' Compress LCP array
	ec = lcptab.compact(); CHECKERROR(ec);


  if(_verb == SA_DEBUG)
    std::cout<< "LCP Table : " << std::endl << lcptab << std::endl;

	//' Construct Child Table
	ec = ConstructChildTable(); CHECKERROR(ec);

	if(_verb == SA_DEBUG)
    std::cout<< "Child Table : " << std::endl << childtab << std::endl;



#ifdef SLINK
	//' Construct Suffix link table
	//' The suffix link interval, (l-1)-[p..q] of interval l-[i..j] can be retrieved
	//'   by following method:
	//'   Let k be the firstlIndex of l-[i..j], p = suflink[2*k], q = suflink[2*k+1].
	suflink = new UInt32[2 * size + 2];  //' extra space for extra sentinel char!
	memset(suflink,0,sizeof(UInt32)*(2 * size +2));
	ec = ConstructSuflink(); CHECKERROR(ec);
	
  if(_verb == SA_DEBUG){
    for(UInt32 kk=0; kk< size; kk++)
      std::cout << "SL["<<kk<<"]: ("<<suflink[2*kk]<<","<<suflink[2*kk+1]<<")"<<std::endl;
    std::cout << std::endl;
  }	
#else

	//' Threshold for constructing bucket table
	if(size >= 1024) 
		ec = ConstructBcktab(); CHECKERROR(ec);

	//' Otherwise, just do plain binary search to search for suffix link interval
	
#endif
	
}


ESA::~ESA()
{
	if(suflink) { delete [] suflink; suflink=NULL; }
	if(suftab) { delete [] suftab; suftab=NULL; }
}

/// The lcp-interval structure. Used in ESA::ConstructChildTable()
class lcp_interval {
	
public:

	UInt32 lcp;
	UInt32 lb;
	UInt32 rb;
  std::vector<lcp_interval *> child;
  
	/// Constructors
	lcp_interval(){}

	lcp_interval(const UInt32 &lcp_, const UInt32 lb_,
							 const UInt32 &rb_, lcp_interval *itv) {
		lcp = lcp_;
		lb = lb_;
		rb = rb_;
    if(itv) 
      child.push_back(itv);
	}

	/// Destructor
	~lcp_interval(){
		for(UInt32 i=0; i< child.size(); i++)
			delete child[i];
		child.clear();
	}
  
};


/**
 *  Construct 3-fields-merged child table.
 */
ErrorCode
ESA::ConstructChildTable(){
	
	// Input validation
	ASSERT(text);
	ASSERT(suftab);

	
	//' stack for lcp-intervals
  std::stack<lcp_interval*> lit;

	
	//' Refer to: Abo05::Algorithm 4.5.2.
	lcp_interval *lastInterval = 0;
	lcp_interval *new_itv = 0;
	lit.push(new lcp_interval(0, 0, 0, 0));  //' root interval


  // Variables to handle 0-idx 
  bool first = true;
  UInt32 prev_0idx = 0;
  UInt32 first0idx = 0;
	  
	// Loop thro and process each index. 
	for(UInt32 idx = 1; idx < size + 1; idx++) {
    
		UInt32 tmp_lb = idx - 1;

		//svnvish: BUGBUG
    // We just assume that the lcp of size + 1 is zero. 
    // This simplifies the logic of the code
    UInt32 lcp_idx = 0;
    if(idx < size){
			lcp_idx = lcptab[idx];
    } 
    
		while (lcp_idx < lit.top()->lcp){
      
      lastInterval = lit.top(); lit.pop();
			lastInterval->rb = idx - 1;
            
      // svnvish: Begin process
			UInt32 n_child = lastInterval->child.size();
			UInt32 i = lastInterval->lb;
			UInt32 j = lastInterval->rb; // idx -1 ?


			//Step 1: Set childtab[i].down or childtab[j+1].up to first l-index
			UInt32 first_l_index = i+1;
      if(n_child && (lastInterval->child[0]->lb == i))
        first_l_index = lastInterval->child[0]->rb+1;
      

      //svnvish: BUGBUG
      // ec = childtab.Set_Up(lastInterval->rb+1, first_l_index);
      // ec = childtab.Set_Down(lastInterval->lb, first_l_index);
      
      childtab[lastInterval->rb] = first_l_index; 
      childtab[lastInterval->lb] = first_l_index;

      // Now we need to set the NextlIndex fields The main problem here
      // is that the child intervals might not be contiguous
      
      UInt32 ptr = i+1;
      UInt32 child_count = 0;
      
      while(ptr < j){
        UInt32 n_first = j;
        UInt32 n_last = j;
        
        // Get next child to process
        if(n_child - child_count){
          n_first = lastInterval->child[child_count]->lb;
          n_last = lastInterval->child[child_count]->rb;
          child_count++;
        }
        
        // Eat away singleton intervals
        while(ptr < n_first){
					childtab[ptr] = ptr + 1;
          ptr++;
        }
      
        // Handle an child interval and make appropriate entries in
        // child table
        ptr = n_last + 1;
				if(n_last < j){
					childtab[n_first] = ptr;
        }
				
      }
      

			//' Free lcp_intervals
			for(UInt32 child_cnt = 0; child_cnt < n_child; child_cnt++) {
				delete lastInterval->child[child_cnt];
				lastInterval->child[child_cnt] = 0;
      }
			// svnvish: End process

			tmp_lb = lastInterval->lb;

			if(lcp_idx <= lit.top()->lcp) {
				lit.top()->child.push_back(lastInterval);
        lastInterval = 0; 
			}
      
		}// while
    
    
		if(lcp_idx > lit.top()->lcp) {
			new_itv = new lcp_interval(lcp_idx, tmp_lb,0, lastInterval);
		  lit.push(new_itv);
			new_itv = 0;
			lastInterval = 0;
		}
    
    // Handle the 0-indices.
    // 0-indices := { i | LCP[i]=0, \forall i = 0,...,n-1}
    if((idx < size) && (lcp_idx == 0)) {
      // svnvish: BUGBUG
      // ec = childtab.Set_NextlIndex(prev_0_index,k);
      childtab[prev_0idx] = idx;
      prev_0idx = idx;
      // Handle first 0-index specially
      // Store in childtab[(size-1)+1].up
      if(first){
        // svnvish: BUGBUG
        // ec = childtab.Set_Up(size,k); CHECKERROR(ec);
        first0idx = idx;
        first = false;
      }
    }
  } // for
  childtab[size-1] = first0idx;



  // svnvish: All remaining elements in the stack are ignored.
	// chteo: Free all remaining elements in the stack.
	while(!lit.empty()) {
		lastInterval = lit.top();
		delete lastInterval;
		lit.pop();
	}

	ASSERT(lit.empty());
  return NOERROR;
}

#ifdef SLINK

/**
 *  Get suffix link interval, [sl_i..sl_j], of a given interval, [i..j].
 *
 *  \param i    - (IN) Left bound of interval [i..j]
 *  \param j    - (IN) Right bound of interval [i..j]
 *  \param sl_i - (OUT) Left bound of suffix link interval [sl_i..sl_j]
 *  \param sl_j - (OUT) Right bound of suffix link interval [sl_i..sl_j]
 */
ErrorCode
ESA::GetSuflink(const UInt32 &i, const UInt32 &j,	UInt32 &sl_i, UInt32 &sl_j)
{
	ErrorCode ec;

	//' Input validation
	ASSERT(i<j);
	ASSERT(i>=0 && j<size);

	UInt32 idx;

	//' Step 1: Get first l-index
	ec = childtab.l_idx(i,j,idx); CHECKERROR(ec);
	

	//' Step 2: Get suffix link interval
	sl_i = suflink[idx+idx];
	sl_j = suflink[idx+idx+1];

	//' Validate suffix link interval
	ASSERT(sl_i < sl_j && (sl_j-sl_i) >= (j-i));

	return NOERROR;
}


#elif defined(LSEARCH)
/**
 *  "Linear" Search version of GetSuflink. Suffix link intervals are not stored
 *    explicitly but searched when needed.
 *
 *  Note: Slow!!! especially in the case of long and similar texts.
 */
ErrorCode 
ESA::GetSuflink(const UInt32 &i, const UInt32 &j,
									UInt32 &sl_i, UInt32 &sl_j)
{
	ErrorCode ec;
	
	//' Variables
	SYMBOL ch;
	UInt32 lcp=0;
	UInt32 final_lcp = 0;
	UInt32 lb = 0, rb = size-1;  //' root interval
	
	//' First suflink interval char := Second char of original interval
	ch = text[suftab[i]+1];
	
	
	//' lcp of suffix link interval := lcp of original interval - 1
	final_lcp = 0;
	ec = GetLcp(i,j,final_lcp); CHECKERROR(ec);
	final_lcp = (final_lcp > 0) ? final_lcp-1 : 0;

	
	//' Searching for suffix link interval
	sl_i = lb;
	sl_j = rb;

	while(lcp < final_lcp) {
		ec = GetIntervalByChar(lb,rb,ch,lcp,sl_i, sl_j); CHECKERROR(ec);
		ec = GetLcp(sl_i, sl_j, lcp); CHECKERROR(ec);

		lb = sl_i;
		rb = sl_j;
		ch = text[suftab[i]+lcp+1];
	}
	
	ASSERT(sl_j > sl_i);
	ASSERT((sl_j-sl_i) >= (j-i));
	
	return NOERROR;
}

#else

/**
 *  Construct bucket table.
 *
 *  \param alpahabet_size - Size of alphabet set
 */
ErrorCode
ESA::ConstructBcktab(const UInt32 &alphabet_size)
{

	UInt32 MAX_DEPTH = 8;  //' when alphabet_size is 256
	UInt32 sizeof_uint4 = 4;  //' 4 bytes integer
	UInt32 sizeof_uint8 = 8;  //' 8 bytes integer
	UInt32 sizeof_key = sizeof_uint8;
	
 
	//' Step 1: Determine the bcktab_depth
	for(bcktab_depth = MAX_DEPTH; bcktab_depth >0; bcktab_depth--) {
		bcktab_size = 0;
		
		for(UInt32 i=0; i < size; i++)
			if(lcptab[i] < bcktab_depth) 
				bcktab_size++;
		
		if(bcktab_depth <= 4) 
			sizeof_key = sizeof_uint4;

		if(bcktab_size <= size/(sizeof_key + sizeof_uint4)) 
			break;
	}


	//' Step 2: Allocate memory for bcktab_key and bcktab_val.		
	//' Step 3: Precompute coefficients for computing hash values of prefixes later.
	//' Step 4: Collect the prefixes with lcp <= bcktab_depth and 
	//'           convert them into hash value.
 	if(sizeof_key == sizeof_uint4) {
		//' (2)
		bcktab_key4 = new UInt32[bcktab_size];
		bcktab_val = new UInt32[bcktab_size];
		ASSERT(bcktab_key4 && bcktab_val);	

		//' (3)
		coef4 =  new UInt32[4];
		coef4[0] = 1;
		for(UInt32 i=1; i < 4; i++)
			coef4[i] = coef4[i-1]*alphabet_size;


		//' (4)
		for(UInt32 i=0, k=0; i < size; i++) {
			if(lcptab[i] < bcktab_depth) {
				UInt32 c = MIN((size-suftab[i]), bcktab_depth);
				hash_value4 = 0;
				for(UInt32 j=0; j < c; j++)
					hash_value4 += text[suftab[i]+j]*coef4[bcktab_depth-1-j];
				
				bcktab_key4[k] = hash_value4;
				bcktab_val[k] = i;
				k++;
			}
		}
	}
	else {
		//' (2)
		bcktab_key8 = new UInt64[bcktab_size];
		bcktab_val = new UInt32[bcktab_size];
		ASSERT(bcktab_key8 && bcktab_val);	

		//' (3)
		coef8 =  new UInt64[9];
		coef8[0] = 1;
		for(UInt32 i=1; i < 9; i++)
			coef8[i] = coef8[i-1]*alphabet_size;
		
		//' (4)
		for(UInt32 i=0, k=0; i < size; i++) {
			if(lcptab[i] < bcktab_depth) {
				UInt32 c = MIN( (size-suftab[i]), bcktab_depth);
				hash_value8 = 0;
				for(UInt32 j=0; j < c; j++)
					hash_value8 += text[suftab[i]+j]*coef8[bcktab_depth-1-j];
				
				bcktab_key8[k] = hash_value8;
				bcktab_val[k] = i;
				k++;
			}
		}
	}
	

	//' check if bcktab in ascending order
	for(UInt32 ii=1; ii<bcktab_size; ii++) {
		if(bcktab_val[ii] <= bcktab_val[ii-1]) {
			for(UInt32 jj=0; jj<4; jj++)
				std::cout << int(text[bcktab_val[ii]+jj]) << " ";
			std::cout << std::endl;
		}
		

		//if(bcktab_key4 && bcktab_key4[ii] <= bcktab_key4[ii-1])

	}

	return NOERROR;
}



/**
 *  O(P log N) "Binary" Search version of GetSuflink. Suffix link intervals are not stored
 *    explicitly but bin-searched when needed. 
 *
 */
ErrorCode 
ESA::GetSuflink(const UInt32 &i, const UInt32 &j,
								UInt32 &sl_i, UInt32 &sl_j)
{
	ErrorCode ec;
	
	ASSERT(j-i >= 1); //' the interval [i..j] must has at least 2 suffixes.


	//' Variables
	UInt32 left=0, mid=0, right=0, tmp_right=0;
	UInt32 llcp=0, mlcp=0, rlcp=0;
	UInt32 orig_lcp = 0;
	UInt32 c = 0;
	UInt32 offset = 0;

	ec = GetLcp(i, j, orig_lcp); CHECKERROR(ec);

	if(orig_lcp <= 1) {
		sl_i = 0;
		sl_j = size-1;
		return NOERROR;
	}

	//' Default
	left = 0;
	right = size-1;

	//' Make use of bcktab here. Maximum lcp value is always 1 less than bcktab_depth.
	//'   This is because including lcp values equal to bcktab_depth will violate
	//'   the constraint of  prefix uniqueness.
	offset = MIN(orig_lcp-1, bcktab_depth);

	ASSERT(offset>=0);

	if(bcktab_key4) {
		hash_value4 = 0;
		for(UInt32 cnt=0; cnt < offset; cnt++)
			hash_value4 += coef4[bcktab_depth-1-cnt]*text[suftab[i]+1+cnt];

		
		//' lower bound return the exact position of of target, if found one
		UInt32 *p = std::lower_bound(bcktab_key4, bcktab_key4+bcktab_size, hash_value4);
		left = bcktab_val[p - bcktab_key4];
		
		
		//' this hash value is used to find the right bound of target interval
		hash_value4 += coef4[bcktab_depth-offset];

		
		//' upper bound return the smallest value > than target.
		UInt32 *q = std::upper_bound(p, bcktab_key4+bcktab_size, hash_value4);
		if(q == bcktab_key4+bcktab_size)
			right = size-1;
		else 
			right = bcktab_val[q - bcktab_key4] - 1;
	}
	else if(bcktab_key8) {
		hash_value8 = 0;
		for(UInt32 cnt=0; cnt < offset; cnt++)
			hash_value8 += coef8[bcktab_depth-1-cnt]*text[suftab[i]+1+cnt];

		//' lower bound return the exact position of of target, if found one
		UInt64 *p = std::lower_bound(bcktab_key8, bcktab_key8+bcktab_size, hash_value8);
		left = bcktab_val[p - bcktab_key8];

		//' this hash value is used to find the right bound of target interval
		hash_value8 += coef8[bcktab_depth-offset];

		//' upper bound return the smallest value > than target.
		UInt64 *q = std::upper_bound(p, bcktab_key8+bcktab_size, hash_value8);
		if(q == bcktab_key8+bcktab_size)
			right = size-1;
		else
			right = bcktab_val[q - bcktab_key8] - 1;
	}
	tmp_right = right;

	ASSERT(right <= size-1);
	ASSERT(right > left);


	offset = 0;
	//' Compute LEFT boundary of suflink interval
	ec = Compare(left, offset, &text[suftab[i]+1+offset], orig_lcp-1-offset, llcp); 
	CHECKERROR(ec);
	llcp += offset;

	if(llcp < orig_lcp-1) {
		ec = Compare(right, offset, &text[suftab[i]+1+offset], orig_lcp-1-offset, rlcp); 
		CHECKERROR(ec);
		rlcp += offset;

		c = MIN(llcp,rlcp);

	
		while(right-left > 1){
			mid = (left + right)/2;		
			ec = Compare(mid, c, &text[suftab[i]+1+c], orig_lcp-1-c, mlcp); CHECKERROR(ec);
			mlcp += c;
			
			//' if target not found yet...
			if(mlcp < orig_lcp-1) {
				if(text[suftab[mid]+mlcp] < text[suftab[i]+mlcp+1]) {
					left = mid;
					llcp = mlcp;
				}
				else {
					right = mid;
					rlcp = mlcp;
				}
			}
			else {
				//' mlcp == orig_lcp-1
				ASSERT(mlcp == orig_lcp-1);
				//' target found, but want to make sure it is the LEFTmost...
				right = mid;	
				rlcp = mlcp;
			}
			c = MIN(llcp, rlcp);	
		}
		
		sl_i = right;
		llcp = rlcp;
	}
	else {
		sl_i = left;
	}

	

	//' Compute RIGHT boundary of suflink interval
	right = tmp_right;
	left = sl_i;
	ec = Compare(right, offset, &text[suftab[i]+1+offset], orig_lcp-1-offset, rlcp);
	CHECKERROR(ec);
	rlcp += offset;

	if(rlcp < orig_lcp-1) {
		c = MIN(llcp,rlcp);

	
		while(right-left > 1){
			mid = (left + right)/2;		
			ec = Compare(mid, c, &text[suftab[i]+1+c], orig_lcp-1-c, mlcp); CHECKERROR(ec);
			mlcp += c;

			//' if target not found yet...
			if(mlcp < orig_lcp-1) {
				if(text[suftab[mid]+mlcp] < text[suftab[i]+mlcp+1]) {
					//' target is on the right half
					left = mid;
					llcp = mlcp;
				}
				else {
					//' target is on the left half
					right = mid;
					rlcp = mlcp;
				}
			}
			else {
				//' mlcp == orig_lcp-1
				ASSERT(mlcp == orig_lcp-1);
				//' target found, but want to make sure it is the RIGHTmost...
				left = mid;	
				llcp = mlcp;
			}
			c = MIN(llcp, rlcp);	
		}
		
		sl_j = left;
	}
	else {
		sl_j = right;
	}

	ASSERT(sl_i < sl_j);
	return NOERROR;
}

#endif


/**
 *  Find suffix link interval, [p..q], for a child interval, [c_i..c_j], given its
 *    parent interval [p_i..p_j].
 *
 *  Pre : 1. Suffix link interval for parent interval has been computed.
 *        2. [child_i..child_j] is not a singleton interval.
 *  
 *
 *  \param parent_i - (IN) Left bound of parent interval.
 *  \param parent_j - (IN) Right bound of parent interval.
 *  \param child_i  - (IN) Left bound of child interval.
 *  \param child_j  - (IN) Right bound of child interval.
 *  \param sl_i     - (OUT) Left bound of suffix link interval of child interval
 *  \param sl_j     - (OUT) Right bound of suffix link interval of child interval
 */
ErrorCode
ESA::FindSuflink(const UInt32 &parent_i, const UInt32 &parent_j,
								 const UInt32 &child_i, const UInt32 &child_j,
								 UInt32 &sl_i, UInt32 &sl_j)
{
	ErrorCode ec;

	ASSERT(child_i != child_j);

	//' Variables
	SYMBOL ch;
	UInt32 tmp_i = 0;
	UInt32 tmp_j = 0;
	UInt32 lcp_child = 0;
	UInt32 lcp_parent = 0;
	UInt32 lcp_sl = 0;
	

 	//' Step 1: Get suffix link interval of parent interval and its lcp value.
	//'      2: Get lcp values of parent and child intervals.

	//' Shortcut!
	if(parent_i ==0 && parent_j == size-1) {  //' this is root interval
		//' (1)
		sl_i = 0;
		sl_j = size-1;
		lcp_sl = 0;

		//' (2)
		lcp_parent = 0;
		ec = GetLcp(child_i,child_j,lcp_child); CHECKERROR(ec);		
		ASSERT(lcp_child  >  0);
	}
	else {
		//' (1)
		ec = GetSuflink(parent_i,parent_j,sl_i,sl_j); CHECKERROR(ec);
		ec = GetLcp(sl_i, sl_j, lcp_sl); CHECKERROR(ec);

		//' (2)
		ec = GetLcp(parent_i,parent_j,lcp_parent); CHECKERROR(ec);
		ec = GetLcp(child_i,child_j,lcp_child); CHECKERROR(ec);				
		ASSERT(lcp_child > 0);
	}


	//' Traversing down the subtree of [sl_i..sl_j] and looking for
	//'   the suffix link interval of child interval.
	
	while (lcp_sl < lcp_child-1) {
		
		//' The character that we want to look for in suflink interval.
		ch = text[suftab[child_i]+lcp_sl+1];
		
		tmp_i = sl_i;
		tmp_j = sl_j;

		
		ec = GetIntervalByChar(tmp_i, tmp_j, ch, lcp_sl, sl_i, sl_j); CHECKERROR(ec);
		ASSERT(sl_i<sl_j);  //' There must be a suflink interval for every interval.

		ec = GetLcp(sl_i, sl_j, lcp_sl); CHECKERROR(ec);

		ASSERT(lcp_sl <= lcp_child-1);
	}
	
	return NOERROR;
}



/**
 *  Construct suffix link table.
 *
 *  Reference: Abo05::pg90::Method 3
 *
 *  Time complexity : O(n)
 *  Space complexity: O(n)
 */
ErrorCode
ESA::ConstructSuflink()
{
	ErrorCode ec;
	
	//' Breadth-first traversal, need to keep a queue structure to store the 
	//'   interval-to-explore. Interval-to-explore := (i,j) where [i..j] is an 
	//'   lcp-interval.
	
	//' Variables
  std::queue< std::pair<UInt32,UInt32> > q;   //' The interval queue
	std::pair<UInt32,UInt32> interval;    

	//' Step 0: Push root onto queue. And define itself as its suflink.
	q.push(std::make_pair(0,size-1)); 

	UInt32 idx = 0; 
	ec = childtab.l_idx(0,size-1,idx); CHECKERROR(ec);

	suflink[idx+idx] = 0;          //' left bound of suffix link interval
	suflink[idx+idx+1] = size-1;   //' right bound of suffix link interval
	

	//' Step 1: Breadth first traversal.
  while (!q.empty()) {
		//' Step 1.1: Pop interval from queue.
		interval = q.front(); q.pop();
		
		//' Step 1.2: For each non-singleton child-intervals, [p..q], "find" its 
		//'             suffix link interval and then "push" it onto the interval queue.
		
		UInt32 i=0,j=0, sl_i=0, sl_j=0, start_idx=interval.first;
		do {
			//' Notes: interval.first := left bound of suffix link interval
			//'        interval.second := right bound of suffix link interval
			
			ASSERT(interval.first>=0 && interval.second < size);
			ec = GetIntervalByIndex(interval.first, interval.second, start_idx, i, j); 
			CHECKERROR(ec);
			
			if(j > i) {
				//' [i..j] is non-singleton interval
				ec = FindSuflink(interval.first, interval.second, i,j, sl_i, sl_j); CHECKERROR(ec);
				
				ASSERT(!(sl_i == i && sl_j == j));

				//' Store suflink of [i..j]
				idx=0;
				ec = childtab.l_idx(i, j, idx); CHECKERROR(ec);
				
				suflink[idx+idx] = sl_i;
				suflink[idx+idx+1] = sl_j;

				//' Push suflink interval onto queue
				q.push(std::make_pair(i,j));
			}

			start_idx = j+1;  //' prepare to get next child interval
		}while(start_idx < interval.second);
			
	}

	return NOERROR;
}



/** 
 *  Get all child-intervals, including singletons.
 *  Store all non-singleton intervals onto #q#, where interval is defined as
 *    (i,j) where i and j are left and right bounds.
 *
 *  \param lb - (IN) Left bound of current interval.
 *  \param rb - (IN) Right bound of current interval.
 *  \param q  - (OUT) Storage for intervals.
 */
ErrorCode
ESA::GetChildIntervals(const UInt32 &lb, const UInt32 &rb, 
											 std::vector<std::pair<UInt32,UInt32> > &q)
{
	ErrorCode ec;

	//' Variables
	UInt32 k=0;       //' general index
	UInt32 i=0,j=0;   //' for interval [i..j]


	//' Input validation
	ASSERT(rb-lb >= 1);

	k = lb;
	do {
		ASSERT(lb>=0 && rb<size);
		ec = GetIntervalByIndex(lb,rb,k,i,j); CHECKERROR(ec);
		if(j-i> 0) {
			//' Non-singleton interval
			q.push_back(std::make_pair(i,j));
		}
		k = j+1;
	}while(k < rb);

	return NOERROR;

}



/**
 *  Given an l-interval, l-[i..j] and a starting index, idx \in [i..j],
 *    return the child-interval, k-[p..q], of l-[i..j] where p == idx.
 *
 *  Reference: Abo05::algorithm 4.6.4
 * 
 *  Pre: #start_idx# is a l-index or equal to parent_i.
 *
 *  \param parent_i  - (IN) Left bound of parent interval.
 *  \param parent_j  - (IN) Right bound of parent interval.
 *  \param start_idx - (IN) Predefined left bound of child interval.
 *  \param child_i   - (OUT) Left bound of child interval.
 *  \param child_j   - (OUT) Right bound of child interval.
 *
 *  Time complexity: O(|alphabet set|)
 */
ErrorCode 
ESA::GetIntervalByIndex(const UInt32 &parent_i, const UInt32 &parent_j,
												const UInt32 &start_idx, UInt32 &child_i, UInt32 &child_j)
{
	ErrorCode ec;

	//' Variables
	UInt32 lcp_child_i = 0;
	UInt32 lcp_child_j = 0;
  
	//' Input validation
	ASSERT( (parent_i < parent_j) && (parent_i >= 0) &&  
          (parent_j < size) && (start_idx >= parent_i) &&  
          (start_idx < parent_j));
	
	child_i = start_idx;

	//' #start_idx# is not and l-index, i.e. #start_idx == #parent_i#
	if(child_i == parent_i) {
		ec = childtab.l_idx(parent_i,parent_j,child_j); CHECKERROR(ec);
		child_j--;

		return NOERROR;
	}

	//' #start_idx# is a l-index
	// svnvish:BUGBUG
  child_j = childtab[child_i];
	lcp_child_i = lcptab[child_i]; 
	lcp_child_j = lcptab[child_j]; 

	if(child_i < child_j  &&  lcp_child_i == lcp_child_j)
		child_j--;
	else 	{
		//' child_i is the left bound of last child interval
		child_j = parent_j;
	}
	
	return NOERROR;
}



/**
 *  Given an l-interval, l-[i..j] and a starting character, ch \in alphabet set,
 *    return the child-interval, k-[p..q], of l-[i..j] such that text[sa[p]+depth] == ch.
 *
 *  Reference: Abo05::algorithm 4.6.4
 *
 *  Post: Return [i..j]. If interval was found, i<=j, otherwise, i>j.
 *
 *  \param parent_i  - (IN) Left bound of parent interval.
 *  \param parent_j  - (IN) Right bound of parent interval.
 *  \param ch        - (IN) Starting character of left bound (suffix) of child interval.
 *  \param depth     - (IN) The position where #ch# is located in #text# 
 *                            i.e. ch = text[suftab[parent_i]+depth].
 *  \param child_i   - (OUT) Left bound of child interval.
 *  \param child_j   - (OUT) Right bound of child interval.
 *
 *  Time complexity: O(|alphabet set|)
 */
ErrorCode 
ESA::GetIntervalByChar(const UInt32 &parent_i, const UInt32 &parent_j,
											 const SYMBOL &ch, const UInt32 &depth,
											 UInt32 &child_i, UInt32 &child_j)
{
 
	ErrorCode ec;

	//' Input validation
	ASSERT(parent_i < parent_j  &&  parent_i >= 0  &&  parent_j < size);


	//' Variables
	UInt32 idx = 0;
	UInt32 idx_next = 0;
	UInt32 lcp_idx = 0;
	UInt32 lcp_idx_next = 0;
	UInt32 lcp = 0;


	//' #depth# is actually equal to the following statement!
	//ec = GetLcp(parent_i, parent_j, lcp); CHECKERROR(ec);
	lcp = depth;

	//' Step 1: Check if #ch# falls in the initial range.
	if(text[suftab[parent_i]+lcp] > ch  ||  text[suftab[parent_j]+lcp] < ch) {
		//' No child interval starts with #ch#, so, return undefined interval.
		child_i = 1;
		child_j = 0;
		return NOERROR;
	}


	//' Step 2: #ch# is in the initial range, but not necessarily exists in the range.
	//' Step 2.1: Get first l-index
	ec = childtab.l_idx(parent_i, parent_j, idx); CHECKERROR(ec);

	ASSERT(idx > parent_i && idx <= parent_j);

	if(text[suftab[idx-1]+lcp] == ch) {
		child_i = parent_i;
		child_j = idx-1;
		return NOERROR;
	}


	//' Step 3: Look for child interval which starts with #ch#
	// svnvish: BUGBUG
  //ec = childtab.NextlIndex(idx, idx_next); CHECKERROR(ec);
  idx_next = childtab[idx];
	lcp_idx = lcptab[idx];
  lcp_idx_next = lcptab[idx_next];
  
	while(idx < idx_next  &&  lcp_idx == lcp_idx_next  &&  text[suftab[idx]+lcp] < ch) {
		idx = idx_next;
		// svnvish: BUGBUG
    // ec = childtab.NextlIndex(idx, idx_next); CHECKERROR(ec);
    idx_next = childtab[idx];
		lcp_idx = lcptab[idx];
    lcp_idx_next = lcptab[idx_next];
	}

 	if(text[suftab[idx]+lcp] == ch) {
		child_i = idx;
		
		if(idx < idx_next && lcp_idx == lcp_idx_next)
			child_j = idx_next - 1;
		else
			child_j = parent_j;

		return NOERROR;
	}

	//' Child interval starts with #ch# not found
	child_i = 1;
	child_j = 0;

  return NOERROR;
}



/**
 *  Return the lcp value of a given interval, l-[i..j].
 *
 *  Pre: [i..j] \subseteq [0..size]
 *
 *  \param i   - (IN) Left bound of the interval.
 *  \param j   - (IN) Right bound of the interval.
 *  \param val - (OUT) The lcp value of the interval.
 */
ErrorCode
ESA::GetLcp(const UInt32 &i, const UInt32 &j, UInt32 &val)
{
	ErrorCode ec;

	
	//' Input validation
	ASSERT(i < j  &&  i >= 0  &&  j < size);

	//' Variables
	UInt32 up, down;
	
	//' 0-[0..size-1]. This is a shortcut!
	if(i == 0 && j == size) {
		val = 0;
	}
	else {
		ec = childtab.up(j+1,up);  CHECKERROR(ec);

		if( (i < up) && (up <= j)) {
			val = lcptab[up];
		}
		else {
			ec = childtab.down(i,down); CHECKERROR(ec);
			val = lcptab[down];
		}
	}

	return NOERROR;
}



/**
 *  Compare #pattern# string to text[suftab[#idx#]..size] and return the
 *    length of the substring matched.
 *
 *  \param idx         - (IN) The index of esa.
 *  \param depth       - (IN) The start position of matching mechanism. 
 *  \param pattern     - (IN) The pattern string.
 *  \param p_len       - (IN) The length of #pattern#.
 *  \param matched_len - (OUT) The length of matched substring.
 */
ErrorCode
ESA::Compare(const UInt32 &idx, const UInt32 &depth, SYMBOL *pattern, 
						 const UInt32 &p_len, UInt32 &matched_len)
{
	//' Variables
	UInt32 min=0;

	min = (p_len < size-(suftab[idx]+depth)) ? p_len : size-(suftab[idx]+depth);

	matched_len = 0;
	for(UInt32 k=0; k < min; k++) {
		if(text[suftab[idx]+depth+k] == pattern[k])
			matched_len++;
		else
			break;
	}

	return NOERROR;
}




/**
 *  Find the longest matching of text and pattern.
 *
 *  Note: undefinded interval := [i..j] where i>j
 *
 *  Post: Return "floor" and "ceil" of longest substring of pattern that exists in text.
 *          Otherwise, that is, no substring of pattern ever exists in text,
 *          return the starting interval, [i..j].
 *
 *  \param i           - (IN) Left bound of the starting interval.
 *  \param j           - (IN) Rgiht bound of the starting interval.
 *  \param offset      - (IN) The number of characters between the head of suffix and the
 *                              position to start matching.
 *  \param pattern     - (IN) The pattern string to match to esa.
 *  \param p_len       - (IN) The length of #pattern#
 *  \param lb          - (OUT) The left bound of the interval containing 
 *                               longest matched suffix.
 *  \param rb          - (OUT) The right bound of the interval containing 
 *                               longest matched suffix.
 *  \param matched_len - (OUT) The length of the longest matched suffix.
 *  \param floor_lb    - (OUT) Left bound of floor interval of [lb..rb].
 *  \param floor_rb    - (OUT) Right bound of floor interval of [lb..rb].
 *  \param floor_len   - (OUT) The lcp value of floor interval.
 */
ErrorCode 
ESA::ExactSuffixMatch(const UInt32 &i, const UInt32 &j, const UInt32 &offset,
											SYMBOL *pattern, const UInt32 p_len, UInt32 &lb, UInt32 &rb, 
											UInt32 &matched_len, UInt32 &floor_lb, UInt32 &floor_rb, 
											UInt32 &floor_len) 
{
	ErrorCode ec;

	//' Input validation
	ASSERT(i != j);

	
	//' Variables
	UInt32 min, lcp;
	bool queryFound = true;
	

	//' Initial setting.
	floor_lb = lb = i;
	floor_rb = rb = j;

	matched_len = offset;


	//' Step 1: Get lcp of floor/starting interval.
	ec = GetLcp(floor_lb, floor_rb, lcp); CHECKERROR(ec);
	floor_len = lcp;
	
	//' Step 2: Skipping #offset# characters
	while(lcp < matched_len) {
		floor_lb = lb;
		floor_rb = rb;
		floor_len = lcp;

		ec = GetIntervalByChar(floor_lb, floor_rb, pattern[lcp], lcp, lb, rb); CHECKERROR(ec);
		ASSERT(lb <= rb);
				
		if(lb == rb)
			break;
		
		ec = GetLcp(lb, rb, lcp); CHECKERROR(ec);
	}
	
	//' Step 3: Continue matching from the point (either an interval or singleton) we stopped.
	while( (lb<=rb) && queryFound ) {
		if(lb != rb) {
			ec = GetLcp(lb, rb, lcp); CHECKERROR(ec);

			min = (lcp < p_len) ? lcp : p_len;

			while(matched_len < min) {
				queryFound = (text[suftab[lb]+matched_len] == pattern[matched_len]);
 
				if(queryFound) 
					matched_len++;
				else
					return NOERROR;
			}

			ASSERT(matched_len == min);

			//' Full pattern found!
			if(matched_len == p_len) return NOERROR;
			
			floor_lb = lb;
			floor_rb = rb;
			floor_len = lcp;
			ec = GetIntervalByChar(floor_lb, floor_rb,pattern[matched_len],matched_len,lb,rb);
			CHECKERROR(ec);
			
		}else { 
			//' lb == rb, i.e. singleton interval.
			min = (p_len < size-suftab[lb]) ? p_len : size-suftab[lb];

			while(matched_len<min) {
				queryFound = (text[suftab[lb]+matched_len] == pattern[matched_len]);

				if(queryFound)
					matched_len++;
				else 
					return NOERROR;
			}

			return NOERROR;
		}
	}
	
	//' If the while loop was broken because of undefined interval,
	//'   set [i..j] back to previous interval.
	if(lb > rb) {
		lb = floor_lb;
		rb = floor_rb;
 	}

	return NOERROR;
}



#endif
