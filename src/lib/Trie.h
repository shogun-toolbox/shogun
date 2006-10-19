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

#ifdef USE_TREEMEM
#define NO_CHILD ((INT)-1) 
#else
#define NO_CHILD NULL
#endif

class CTrie
{
public:
  struct Trie
  {
    DREAL weight;
    bool has_seq ;
    
    union 
    {
      SHORTREAL child_weights[4];
#ifdef USE_TREEMEM
      INT children[4];
#else
      struct Trie *children[4];
#endif
      BYTE seq[16] ;
    }; 
  };
  
 public:
  CTrie(INT d) ;
  ~CTrie() ;
  void destroy() ;
  void create(INT len) ;
  void delete_tree(struct Trie * p_tree=NULL);
  void add_to_trie(int i, INT * vec, float alpha, DREAL *weights) ;
  DREAL compute_abs_weights_tree(struct Trie* p_tree) ;
  DREAL *compute_abs_weights(int &len) ;
  
  DREAL compute_by_tree_helper(INT* vec, INT len, INT pos, DREAL * weights) ;
  void compute_by_tree_helper(INT* vec, INT len, INT pos, DREAL* LevelContrib, DREAL factor, INT mkl_stepsize, DREAL * weights) ;
  void compute_scoring_helper(struct Trie *, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result) ;
  void add_example_to_tree_mismatch_recursion(struct Trie *tree,  INT i, DREAL alpha, INT *vec, INT len_rem, INT degree_rec, INT mismatch_rec, INT max_mismatch, DREAL * weights) ;
  
#ifdef USE_TREEMEM
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
#endif
  
 protected:
  INT length ;
  struct Trie** trees ;
  bool tree_initialized ;
  
  INT degree ;
  DREAL * position_weights ;
  
  
#ifdef USE_TREEMEM
  struct Trie* TreeMem ;
  INT TreeMemPtr ;
  INT TreeMemPtrMax ;
#endif
} ;

/* computes the simple kernel */
inline DREAL CTrie::compute_by_tree_helper(INT* vec, INT len, INT pos, DREAL* weights)
{
  DREAL sum=0 ;
  
  struct Trie *tree = trees[pos] ;
  ASSERT(tree!=NULL) ;
  
  for (INT j=0; pos+j < len; j++)
    {
      if ((j<degree-1) && (tree->children[vec[pos+j]]!=NO_CHILD))
	{
	  if (tree->children[vec[pos+j]]<0)
	    {
	      DREAL this_weight=0.0 ;
	      for (INT k=0; (j+k<degree) && (pos+j+k<length); k++)
		{
		  if (tree->seq[k]!=vec[pos+j+k])
		    break ;
		  this_weight += weights[j+k] ;
		}
	      sum += tree->weight * this_weight ;
	      break ;
	    }
	  else
	    {
#ifdef USE_TREEMEM
	      tree=&TreeMem[tree->children[vec[pos+j]]];
#else
	      tree=tree->children[vec[pos+j]];
#endif
	      sum += tree->weight;
	    }
	}
      else
	{
	  if (j==degree-1)
	    sum += tree->child_weights[vec[pos+j]];
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
	struct Trie *tree = trees[pos] ;
	ASSERT(tree!=NULL) ;
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
				if ((j<degree-1) && (tree->children[vec[pos+j]]!=NO_CHILD))
				{
#ifdef USE_TREEMEM
					tree=&TreeMem[tree->children[vec[pos+j]]];
#else
					tree=tree->children[vec[pos+j]];
#endif
					LevelContrib[pos/mkl_stepsize] += factor*tree->weight*weights[j] ;
				} 
				else
				{
					if (j==degree-1)
						LevelContrib[pos/mkl_stepsize] += factor*tree->child_weights[vec[pos+j]]*weights[j] ;
				}
			}
		} 
		else // with position_weigths, weights is a matrix (len x degree)
		{
			for (INT j=0; pos+j<len; j++)
			{
				if ((j<degree-1) && (tree->children[vec[pos+j]]!=NO_CHILD))
				{
#ifdef USE_TREEMEM
					tree=&TreeMem[tree->children[vec[pos+j]]];
#else
					tree=tree->children[vec[pos+j]];
#endif
					LevelContrib[pos/mkl_stepsize] += factor*tree->weight*weights[j+pos*degree] ;
				} 
				else
				{
					if (j==degree-1)
						LevelContrib[pos/mkl_stepsize] += factor*tree->child_weights[vec[pos+j]]*weights[j+pos*degree] ;

					break ;
				}
			}
		} 
	}
	else if (length==0) // no position_weigths, weights is a vector (1 x degree)
	{
		for (INT j=0; pos+j<len; j++)
		{
			if ((j<degree-1) && (tree->children[vec[pos+j]]!=NO_CHILD))
			{
#ifdef USE_TREEMEM
				tree=&TreeMem[tree->children[vec[pos+j]]];
#else
				tree=tree->children[vec[pos+j]];
#endif
				LevelContrib[j/mkl_stepsize] += factor*tree->weight*weights[j] ;
			}
			else
			{
				if (j==degree-1)
					LevelContrib[j/mkl_stepsize] += factor*tree->child_weights[vec[pos+j]]*weights[j] ;
				break ;
			}
		} 
	} 
	else // no position_weigths, weights is a matrix (len x degree)
	{
		for (INT j=0; pos+j<len; j++)
		{
			if ((j<degree-1) && (tree->children[vec[pos+j]]!=NO_CHILD))
			{
#ifdef USE_TREEMEM
				tree=&TreeMem[tree->children[vec[pos+j]]];
#else
				tree=tree->children[vec[pos+j]];
#endif
				LevelContrib[(j+degree*pos)/mkl_stepsize] += factor * tree->weight * weights[j+pos*degree] ;
			}
			else
			{
				if (j==degree-1)
					LevelContrib[(j+degree*pos)/mkl_stepsize] += factor * tree->child_weights[vec[pos+j]] * weights[j+pos*degree] ;
				break ;
			}
		} 
	}
}



#endif
