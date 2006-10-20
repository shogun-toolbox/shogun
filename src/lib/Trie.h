/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _TRIE_H___
#define _TRIE_H___

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

#define NO_CHILD ((INT)-1) 
//#define TRIE_CHECK_EVERYTHING

#ifdef TRIE_CHECK_EVERYTHING
#define TRIE_ASSERT_EVERYTHING(x) ASSERT(x)
#else
#define TRIE_ASSERT_EVERYTHING(x) 
#endif

#define TRIE_ASSERT(x) ASSERT(x)
//#define TRIE_ASSERT(x) 

class CTrie
{
public:
	struct Trie
	{
		DREAL weight;
#ifdef TRIE_CHECK_EVERYTHING
		bool has_seq ;
		bool has_floats ;
#endif		
		union 
		{
			SHORTREAL child_weights[4];
			INT children[4];
			BYTE seq[16] ;
		}; 
	};
	
public:
	CTrie(INT d) ;
	~CTrie() ;
	void destroy() ;
	void create(INT len) ;
	void delete_trees();
	void add_to_trie(int i, INT * vec, float alpha, DREAL *weights) ;
	DREAL compute_abs_weights_tree(INT tree) ;
	DREAL *compute_abs_weights(int &len) ;
	
	DREAL compute_by_tree_helper(INT* vec, INT len, INT pos, DREAL * weights) ;
	void compute_by_tree_helper(INT* vec, INT len, INT pos, DREAL* LevelContrib, DREAL factor, INT mkl_stepsize, DREAL * weights) ;
	void compute_scoring_helper(INT tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result) ;
	void add_example_to_tree_mismatch_recursion(INT tree,  INT i, DREAL alpha, INT *vec, INT len_rem, INT degree_rec, INT mismatch_rec, INT max_mismatch, DREAL * weights) ;
	
	inline INT get_node() 
	{
		INT ret = TreeMemPtr++;
		check_treemem() ;
		for (INT q=0; q<4; q++)
			TreeMem[ret].children[q]=NO_CHILD ;
#ifdef TRIE_CHECK_EVERYTHING
		TreeMem[ret].has_seq=false ;
		TreeMem[ret].has_floats=false ;
#endif
		TreeMem[ret].weight=0.0; 
		return ret ;
	} ;
	
	inline void check_treemem()
	{
		if (TreeMemPtr+10>=TreeMemPtrMax) 
		{
			CIO::message(M_DEBUG, "Extending TreeMem from %i to %i elements\n", TreeMemPtrMax, (INT) ((double)TreeMemPtrMax*1.2)) ;
			TreeMemPtrMax = (INT) ((double)TreeMemPtrMax*1.2) ;
			TreeMem = (struct Trie *)realloc(TreeMem,TreeMemPtrMax*sizeof(struct Trie)) ;
			
			if (!TreeMem)
				CIO::message(M_ERROR, "out of memory\n");
		}
	}
	
protected:
	INT length ;
	INT * trees ;
	bool tree_initialized ;
	
	INT degree ;
	DREAL * position_weights ;
	
	
	struct Trie* TreeMem ;
	INT TreeMemPtr ;
	INT TreeMemPtrMax ;
} ;

/* computes the simple kernel */
inline DREAL CTrie::compute_by_tree_helper(INT* vec, INT len, INT pos, DREAL* weights)
{
  DREAL sum=0 ;
  //fprintf(stderr, "pos=%i\n", pos) ;
  
  INT tree = trees[pos] ;
  
  for (INT j=0; pos+j < len; j++)
  {
	  TRIE_ASSERT((vec[pos+j]<4) && (vec[pos+j]>=0)) ;

	  if ((j<degree-1) && (TreeMem[tree].children[vec[pos+j]]!=NO_CHILD))
	  {
		  TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
		  
		  if (TreeMem[tree].children[vec[pos+j]]<0)
		  {
			  tree = - TreeMem[tree].children[vec[pos+j]];
			  TRIE_ASSERT(tree>=0) ;
			  TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
			  DREAL this_weight=0.0 ;
			  for (INT k=0; (j+k<degree) && (pos+j+k<length); k++)
			  {
				  TRIE_ASSERT((vec[pos+j+k]<4) && (vec[pos+j+k]>=0)) ;
				  if (TreeMem[tree].seq[k]!=vec[pos+j+k])
					  break ;
				  this_weight += weights[j+k] ;
			  }
			  sum += TreeMem[tree].weight * this_weight ;
			  break ;
		  }
		  else
		  {
			  tree = TreeMem[tree].children[vec[pos+j]];
			  
			  TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
			  sum += TreeMem[tree].weight;
		  }
	  }
	  else
	  {
		  TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
		  if (j==degree-1)
		  {
			  TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats) ;
			  sum += TreeMem[tree].child_weights[vec[pos+j]];
		  }
		  else
			  TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
		  
		  break;
	  }
  } 
  
  if (position_weights!=NULL)
	  return sum*position_weights[pos] ;
  else
	  return sum ;
}
 
inline void CTrie::compute_by_tree_helper(INT* vec, INT len, INT pos, DREAL* LevelContrib, DREAL factor, INT mkl_stepsize, DREAL * weights) 
{
	INT tree = trees[pos] ;
	if (factor==0)
		return ;
	
	if (position_weights!=NULL)
    {
		factor *= position_weights[pos] ;
		if (factor==0)
			return ;
		if (length==0) // with position_weigths, weights is a vector (1 x degree)
		{
			for (INT j=0; pos+j<len; j++)
			{
				if ((j<degree-1) && (TreeMem[tree].children[vec[pos+j]]!=NO_CHILD))
				{
					if (TreeMem[tree].children[vec[pos+j]]<0)
					{
						tree = -TreeMem[tree].children[vec[pos+j]];
						TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
						for (INT k=0; (j+k<degree) && (pos+j+k<length); k++)
						{
							if (TreeMem[tree].seq[k]!=vec[pos+j+k])
								break ;
							LevelContrib[pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k] ;
						}
						break ;
					}
					else
					{
						tree = TreeMem[tree].children[vec[pos+j]];
						TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
						LevelContrib[pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j] ;
					}
				} 
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					if (j==degree-1)
						LevelContrib[pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[pos+j]]*weights[j] ;
				}
			}
		} 
		else // with position_weigths, weights is a matrix (len x degree)
		{
			for (INT j=0; pos+j<len; j++)
			{
				if ((j<degree-1) && (TreeMem[tree].children[vec[pos+j]]!=NO_CHILD))
				{
					if (TreeMem[tree].children[vec[pos+j]]<0)
					{
						tree = -TreeMem[tree].children[vec[pos+j]];
						TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
						for (INT k=0; (j+k<degree) && (pos+j+k<length); k++)
						{
							if (TreeMem[tree].seq[k]!=vec[pos+j+k])
								break ;
							LevelContrib[pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k+pos*degree] ;
						}
						break ;
					}
					else
					{
						tree = TreeMem[tree].children[vec[pos+j]];
						TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
						LevelContrib[pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+pos*degree] ;
					}
				} 
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					if (j==degree-1)
						LevelContrib[pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[pos+j]]*weights[j+pos*degree] ;
					
					break ;
				}
			}
		} 
    }
	else if (length==0) // no position_weigths, weights is a vector (1 x degree)
    {
		for (INT j=0; pos+j<len; j++)
		{
			if ((j<degree-1) && (TreeMem[tree].children[vec[pos+j]]!=NO_CHILD))
			{
				if (TreeMem[tree].children[vec[pos+j]]<0)
				{
					tree = -TreeMem[tree].children[vec[pos+j]];
					TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
					for (INT k=0; (j+k<degree) && (pos+j+k<length); k++)
					{
						if (TreeMem[tree].seq[k]!=vec[pos+j+k])
							break ;
						LevelContrib[(j+k)/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k] ;
					}
					break ;
				}
				else
				{
					tree = TreeMem[tree].children[vec[pos+j]] ;
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j] ;
				}
			}
			else
			{
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
				if (j==degree-1)
					LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[pos+j]]*weights[j] ;
				break ;
			}
		} 
    } 
	else // no position_weigths, weights is a matrix (len x degree)
	{
		for (INT j=0; pos+j<len; j++)
		{
			if ((j<degree-1) && (TreeMem[tree].children[vec[pos+j]]!=NO_CHILD))
			{
				if (TreeMem[tree].children[vec[pos+j]]<0)
				{
					tree = -TreeMem[tree].children[vec[pos+j]];
					TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
					for (INT k=0; (j+k<degree) && (pos+j+k<length); k++)
					{
						if (TreeMem[tree].seq[k]!=vec[pos+j+k])
							break ;
						LevelContrib[(j+k+degree*pos)/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k+pos*degree] ;
					}
					break ;
				}
				else
				{
					tree = TreeMem[tree].children[vec[pos+j]];
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					LevelContrib[(j+degree*pos)/mkl_stepsize] += factor * TreeMem[tree].weight * weights[j+pos*degree] ;
				}
			}
			else
			{
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
				if (j==degree-1)
					LevelContrib[(j+degree*pos)/mkl_stepsize] += factor * TreeMem[tree].child_weights[vec[pos+j]] * weights[j+pos*degree] ;
				break ;
			}
		} 
    }
}



#endif
