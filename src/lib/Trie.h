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

#include <string.h>
#include "lib/common.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

//#define NO_CHILD ((INT)-2147483648)
#define NO_CHILD ((INT)-1073741824) 

//#define WEIGHTS_IN_TRIE 
//#define TRIE_CHECK_EVERYTHING

#ifdef TRIE_CHECK_EVERYTHING
#define TRIE_ASSERT_EVERYTHING(x) ASSERT(x)
#else
#define TRIE_ASSERT_EVERYTHING(x) 
#endif

//#define TRIE_ASSERT(x) ASSERT(x)
#define TRIE_ASSERT(x) 

#define TRIE_TERMINAL_CHARACTER  7

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
	
	struct TreeParseInfo {
		INT num_sym;
		INT num_feat;
		INT p;
		INT k;
		INT* nofsKmers;
		DREAL* margFactors;
		INT* x;
		INT* substrs;
		INT y0;
		DREAL* C_k;
		DREAL* L_k;
		DREAL* R_k;
	};
	
public:
	CTrie(INT d, INT p_use_compact_terminal_nodes=true) ;
	CTrie(const CTrie & to_copy) ;
	~CTrie() ;

	const CTrie & operator=(const CTrie & to_copy) ;
	bool compare_traverse(INT node, const CTrie & other, INT other_node) ;
	bool compare(const CTrie & other) ;
	bool find_node(INT node, INT * trace, INT &trace_len) const ;
	INT find_deepest_node(INT start_node, INT &deepest_node) const ;
	void display_node(INT node) const ;
	void destroy() ;
	void create(INT len, INT p_use_compact_terminal_nodes=true) ;
	void delete_trees(INT p_use_compact_terminal_nodes=true);
	void add_to_trie(int i, INT seq_offset, INT * vec, float alpha, DREAL *weights, bool degree_times_position_weights) ;
	DREAL compute_abs_weights_tree(INT tree, INT depth) ;
	DREAL *compute_abs_weights(int &len) ;
	
	DREAL compute_by_tree_helper(INT* vec, INT len, INT seq_pos, INT tree_pos, INT weight_pos, DREAL * weights, bool degree_times_position_weights) ;
	void compute_by_tree_helper(INT* vec, INT len, INT seq_pos, INT tree_pos, INT weight_pos, DREAL* LevelContrib, DREAL factor, INT mkl_stepsize, DREAL * weights, bool degree_times_position_weights) ;
	void compute_scoring_helper(INT tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result) ;
	void add_example_to_tree_mismatch_recursion(INT tree,  INT i, DREAL alpha, INT *vec, INT len_rem, INT degree_rec, INT mismatch_rec, INT max_mismatch, DREAL * weights) ;
	void traverse( INT tree, const INT p, struct TreeParseInfo info, const INT depth, INT* const x, const INT k ) ;
	void count( const DREAL w, const INT depth, const struct TreeParseInfo info, const INT p, INT* x, const INT k ) ;
	INT compact_nodes(INT start_node, INT depth, DREAL * weights) ;
	
	bool get_use_compact_terminal_nodes()
		{
			return use_compact_terminal_nodes ;
		}
	void set_use_compact_terminal_nodes(bool p_use_compact_terminal_nodes)
		{
			use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
		}
	
	inline INT get_num_used_nodes()
		{
			return TreeMemPtr ;
		}
	
	inline void set_position_weights(const DREAL * p_position_weights)
		{
			position_weights=p_position_weights ;
		}
	

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
	DREAL const *  position_weights ;
	
	struct Trie* TreeMem ;
	INT TreeMemPtr ;
	INT TreeMemPtrMax ;
	bool use_compact_terminal_nodes ;
} ;

inline void CTrie::add_to_trie(int i, INT seq_offset, INT * vec, float alpha, DREAL *weights, bool degree_times_position_weights)
{
	INT tree = trees[i] ;
	//ASSERT(seq_offset==0) ;
	
	INT max_depth = 0 ;
#ifdef WEIGHTS_IN_TRIE
	if (degree_times_position_weights)
	{
		for (INT j=0; (j<degree) && (i+j<length); j++)
			if (CMath::abs(weights[j+i*degree]*alpha)>0) // FIXME: i should be weight_pos
				max_depth = j+1 ;
	}
	else
	{
		for (INT j=0; (j<degree) && (i+j<length); j++)
			if (CMath::abs(weights[j]*alpha)>0)
				max_depth = j+1 ;
	}
#else
	// don't use the weights
	max_depth=degree ;
#endif

	for (INT j=0; (j<max_depth) && (i+j+seq_offset<length); j++)
    {
		TRIE_ASSERT((vec[i+j+seq_offset]>=0) && (vec[i+j+seq_offset]<4)) ;
		if ((j<degree-1) && (TreeMem[tree].children[vec[i+j+seq_offset]]!=NO_CHILD))
		{
			if (TreeMem[tree].children[vec[i+j+seq_offset]]<0)
			{
				// special treatment of the next nodes
				TRIE_ASSERT(j >= degree-16) ;
				// get the right element
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
				INT node = - TreeMem[tree].children[vec[i+j+seq_offset]] ;

				TRIE_ASSERT((node>=0) && (node<=TreeMemPtrMax)) ;
				TRIE_ASSERT_EVERYTHING(TreeMem[node].has_seq) ;
				TRIE_ASSERT_EVERYTHING(!TreeMem[node].has_floats) ;
				
                // check whether the same string is stored
				INT mismatch_pos = -1 ;
				INT k ;
				for (k=0; (j+k<max_depth) && (i+j+seq_offset+k<length); k++)
				{
					TRIE_ASSERT((vec[i+j+seq_offset+k]>=0) && (vec[i+j+seq_offset+k]<4)) ;
					// ###
					if ((TreeMem[node].seq[k]>=4) && (TreeMem[node].seq[k]!=TRIE_TERMINAL_CHARACTER))
						fprintf(stderr, "+++i=%i j=%i seq[%i]=%i\n", i, j, k, TreeMem[node].seq[k]) ;
					TRIE_ASSERT((TreeMem[node].seq[k]<4) || (TreeMem[node].seq[k]==TRIE_TERMINAL_CHARACTER)) ;
					TRIE_ASSERT(k<16) ;
					if (TreeMem[node].seq[k]!=vec[i+j+seq_offset+k])
					{
						mismatch_pos=k ;
						break ;
					}
				}
				// what happens when the .seq sequence is longer than vec? should we branch???

				if (mismatch_pos==-1)
					// if so, then just increase the weight by alpha and stop
					TreeMem[node].weight+=alpha ;
				else
					// otherwise
					// 1. replace current node with new node
					// 2. create new nodes until mismatching positon
					// 2. add a branch with old string (old node) and the new string (new node)
				{
					// replace old node
					INT last_node=tree ;
					
					// create new nodes until mismatch
					INT k ;
					for (k=0; k<mismatch_pos; k++)
					{
						TRIE_ASSERT((vec[i+j+seq_offset+k]>=0) && (vec[i+j+seq_offset+k]<4)) ;
						TRIE_ASSERT(vec[i+j+seq_offset+k]==TreeMem[node].seq[k]) ;
						
						INT tmp=get_node() ;
						TreeMem[last_node].children[vec[i+j+seq_offset+k]]=tmp ;
						last_node=tmp ;
#ifdef WEIGHTS_IN_TRIE
						TreeMem[last_node].weight = (TreeMem[node].weight+alpha)*weights[j+k] ;
#else
						TreeMem[last_node].weight = (TreeMem[node].weight+alpha) ;
#endif
						TRIE_ASSERT(j+k!=degree-1) ;
					}
					if ((TreeMem[node].seq[mismatch_pos]>=4) && (TreeMem[node].seq[mismatch_pos]!=TRIE_TERMINAL_CHARACTER))
						fprintf(stderr, "**i=%i j=%i seq[%i]=%i\n", i, j, k, TreeMem[node].seq[mismatch_pos]) ;
					ASSERT((TreeMem[node].seq[mismatch_pos]<4) || (TreeMem[node].seq[mismatch_pos]==TRIE_TERMINAL_CHARACTER)) ;
					TRIE_ASSERT(vec[i+j+seq_offset+mismatch_pos]!=TreeMem[node].seq[mismatch_pos]) ;
					
					if (j+k==degree-1)
					{
						for (INT q=0; q<4; q++)
							TreeMem[last_node].child_weights[q]=0.0 ;
#ifdef WEIGHTS_IN_TRIE
						if (TreeMem[node].seq[mismatch_pos]<4) // i.e. !=TRIE_TERMINAL_CHARACTER
							TreeMem[last_node].child_weights[TreeMem[node].seq[mismatch_pos]]+=TreeMem[node].weight*weights[degree-1] ;
						TreeMem[last_node].child_weights[vec[i+j+seq_offset+k]] += alpha*weights[degree-1] ;
#else
						if (TreeMem[node].seq[mismatch_pos]<4) // i.e. !=TRIE_TERMINAL_CHARACTER
							TreeMem[last_node].child_weights[TreeMem[node].seq[mismatch_pos]]=TreeMem[node].weight ;
						TreeMem[last_node].child_weights[vec[i+j+seq_offset+k]] = alpha ;
#endif
						
#ifdef TRIE_CHECK_EVERYTHING
						TreeMem[last_node].has_floats=true ;
#endif
					}
					else
					{
						// the branch for the existing string
						if (TreeMem[node].seq[mismatch_pos]<4) // i.e. !=TRIE_TERMINAL_CHARACTER
						{
							TreeMem[last_node].children[TreeMem[node].seq[mismatch_pos]] = -node ;

							// move string by mismatch_pos positions
							for (INT q=0; q<16; q++)
							{
								if ((j+q+mismatch_pos<degree) && (i+j+seq_offset+q+mismatch_pos<length))
									TreeMem[node].seq[q] = TreeMem[node].seq[q+mismatch_pos] ;
								else
									TreeMem[node].seq[q] = TRIE_TERMINAL_CHARACTER ;
							}
#ifdef TRIE_CHECK_EVERYTHING
							TreeMem[node].has_seq=true ;
#endif
						}
						
						// the new branch
						TRIE_ASSERT((vec[i+j+seq_offset+mismatch_pos]>=0) && (vec[i+j+seq_offset+mismatch_pos]<4)) ;
						{
							INT tmp = get_node() ;
							TreeMem[last_node].children[vec[i+j+seq_offset+mismatch_pos]] = -tmp ;
							last_node=tmp ;
						}
						TreeMem[last_node].weight = alpha ;
#ifdef TRIE_CHECK_EVERYTHING
						TreeMem[last_node].has_seq = true ;
#endif
						memset(TreeMem[last_node].seq, TRIE_TERMINAL_CHARACTER, 16) ;
						for (INT q=0; (j+q+mismatch_pos<degree) && (i+j+seq_offset+q+mismatch_pos<length); q++)
							TreeMem[last_node].seq[q] = vec[i+j+seq_offset+mismatch_pos+q] ;
					}
				}
				break ;
			} 
			else
			{
				tree=TreeMem[tree].children[vec[i+j+seq_offset]] ;
				TRIE_ASSERT((tree>=0) && (tree<TreeMemPtrMax)) ;
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
#ifdef WEIGHTS_IN_TRIE
				TreeMem[tree].weight += alpha*weights[j];
#else
				TreeMem[tree].weight += alpha ;
#endif
			}
		}
		else if (j==degree-1)
		{
			// special treatment of the last node
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
			/*if (!TreeMem[tree].has_floats)
			{
				fprintf(stderr, "%i\n", TreeMem[tree].children[vec[i+j+seq_offset]]) ;
				if (TreeMem[tree].children[vec[i+j+seq_offset]]>=0)
				{
#ifdef WEIGHTS_IN_TRIE
					TreeMem[TreeMem[tree].children[vec[i+j+seq_offset]]].weight += alpha*weights[j] ;
#else
					TreeMem[TreeMem[tree].children[vec[i+j+seq_offset]]].weight += alpha;
#endif
				}
				else
				{
#ifdef WEIGHTS_IN_TRIE
					TreeMem[-TreeMem[tree].children[vec[i+j+seq_offset]]].weight += alpha*weights[j] ;
#else
					TreeMem[-TreeMem[tree].children[vec[i+j+seq_offset]]].weight += alpha;
#endif
				}
				
			}
			else
			{*/
				TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats) ;
#ifdef WEIGHTS_IN_TRIE
				TreeMem[tree].child_weights[vec[i+j+seq_offset]] += alpha*weights[j] ;
#else
				TreeMem[tree].child_weights[vec[i+j+seq_offset]] += alpha;
#endif
				//}
			break ;
		}
		else
		{
			bool use_seq = use_compact_terminal_nodes && (j>degree-16) ;
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;

			INT tmp = get_node() ;
			if (use_seq)
				TreeMem[tree].children[vec[i+j+seq_offset]] = -tmp ;
			else
				TreeMem[tree].children[vec[i+j+seq_offset]] = tmp ;
			tree=tmp ;
			
			TRIE_ASSERT((tree>=0) && (tree<TreeMemPtrMax)) ;
#ifdef TRIE_CHECK_EVERYTHING
			TreeMem[tree].has_seq = use_seq ;
#endif
			if (use_seq)
			{
				TreeMem[tree].weight = alpha ;
				// important to have the terminal characters (see ###)
				memset(TreeMem[tree].seq, TRIE_TERMINAL_CHARACTER, 16) ;
				for (INT q=0; (j+q<degree) && (i+j+seq_offset+q<length); q++)
				{
					TRIE_ASSERT(q<16) ;
					TreeMem[tree].seq[q]=vec[i+j+seq_offset+q] ;
				}
				break ;
			}
			else
			{
#ifdef WEIGHTS_IN_TRIE
				TreeMem[tree].weight = alpha*weights[j] ;
#else
				TreeMem[tree].weight = alpha ;
#endif
				if (j==degree-2)
				{
#ifdef TRIE_CHECK_EVERYTHING
					TreeMem[tree].has_floats = true ;
#endif
					for (INT k=0; k<4; k++)
						TreeMem[tree].child_weights[k]=0;
				}
				else
				{
					for (INT k=0; k<4; k++)
						TreeMem[tree].children[k]=NO_CHILD;
				}
			}
		}
    }
}

inline DREAL CTrie::compute_by_tree_helper(INT* vec, INT len, INT seq_pos, 
										   INT tree_pos,
										   INT weight_pos, DREAL* weights, 
										   bool degree_times_position_weights)
{
	INT tree = trees[tree_pos] ;
	
	if ((position_weights!=NULL) && (position_weights[weight_pos]==0))
		return 0.0;
	
	DREAL *weights_column=NULL ;
	if (degree_times_position_weights)
    { // weights is a vector (degree x length)
		weights_column=&weights[weight_pos*degree] ;
	    /*if (!position_mask)
		  {		
		  position_mask = new bool[len] ;
		  for (INT i=0; i<len; i++)
		  {
		  position_mask[i]=false ;
		  
		  for (INT j=0; j<degree; j++)
		  if (weights[i*degree+j]!=0.0)
		  {
		  position_mask[i]=true ;
		  break ;
		  }
		  }
		  }
		  if (!position_mask[weight_pos])
		  return 0 ;*/
    }
	else // weights is a vector (1 x degree)
		weights_column=weights ;
	
	DREAL sum=0 ;
	for (INT j=0; seq_pos+j < len; j++)
    {
		TRIE_ASSERT((vec[seq_pos+j]<4) && (vec[seq_pos+j]>=0)) ;
		
		if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
		{
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
			if (TreeMem[tree].children[vec[seq_pos+j]]<0)
			{
				tree = - TreeMem[tree].children[vec[seq_pos+j]];
				TRIE_ASSERT(tree>=0) ;
				TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
				DREAL this_weight=0.0 ;
				for (INT k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
				{
					TRIE_ASSERT((vec[seq_pos+j+k]<4) && (vec[seq_pos+j+k]>=0)) ;
					if (TreeMem[tree].seq[k]!=vec[seq_pos+j+k])
						break ;
					this_weight += weights_column[j+k] ;
				}
				sum += TreeMem[tree].weight * this_weight ;
				break ;
			}
			else
			{
				tree=TreeMem[tree].children[vec[seq_pos+j]];
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
#ifdef WEIGHTS_IN_TRIE
				sum += TreeMem[tree].weight ;
#else
				sum += TreeMem[tree].weight * weights_column[j] ;
#endif
			} ;
		}
		else
		{
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
			if (j==degree-1)
			{
/*				if (TreeMem[tree].has_floats)
				{*/
					TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats) ;
#ifdef WEIGHTS_IN_TRIE
					sum += TreeMem[tree].child_weights[vec[seq_pos+j]] ;
#else
					sum += TreeMem[tree].child_weights[vec[seq_pos+j]] * weights_column[j] ;
#endif
/*				}
				else
				{
					if (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD)
					{
						if (TreeMem[tree].children[vec[seq_pos+j]]<0)
						{
							fprintf(stderr, "node=%i\n", TreeMem[tree].children[vec[seq_pos+j]]) ;
#ifdef WEIGHTS_IN_TRIE
							sum += TreeMem[-TreeMem[tree].children[vec[seq_pos+j]]].weight ;
#else
							sum += TreeMem[-TreeMem[tree].children[vec[seq_pos+j]]].weight * weights_column[j] ;
#endif
						}
						else
						{
							fprintf(stderr, "node=%i\n", TreeMem[tree].children[vec[seq_pos+j]]) ;
#ifdef WEIGHTS_IN_TRIE
							sum += TreeMem[TreeMem[tree].children[vec[seq_pos+j]]].weight ;
#else
							sum += TreeMem[TreeMem[tree].children[vec[seq_pos+j]]].weight * weights_column[j] ;
#endif
						}
					}
				}
*/
			}
			else
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
			
			break;
		}
    } 
	
	if (position_weights!=NULL)
		return sum*position_weights[weight_pos] ;
	else
		return sum ;
}

inline void CTrie::compute_by_tree_helper(INT* vec, INT len,
										  INT seq_pos, INT tree_pos, 
										  INT weight_pos, 
										  DREAL* LevelContrib, DREAL factor, 
										  INT mkl_stepsize, 
										  DREAL * weights, 
										  bool degree_times_position_weights) 
{
	INT tree = trees[tree_pos] ;
	if (factor==0)
		return ;
	
	if (position_weights!=NULL)
    {
		factor *= position_weights[weight_pos] ;
		if (factor==0)
			return ;
		if (!degree_times_position_weights) // with position_weigths, weights is a vector (1 x degree)
		{
			for (INT j=0; seq_pos+j<len; j++)
			{
				if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
				{
					if (TreeMem[tree].children[vec[seq_pos+j]]<0)
					{
						tree = -TreeMem[tree].children[vec[seq_pos+j]];
						TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
						for (INT k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
						{
							if (TreeMem[tree].seq[k]!=vec[seq_pos+j+k])
								break ;
#ifdef WEIGHTS_IN_TRIE
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
#else
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k] ;
#endif
						}
						break ;
					}
					else
					{
						tree=TreeMem[tree].children[vec[seq_pos+j]];
						TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
#ifdef WEIGHTS_IN_TRIE
						LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
#else
						LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j] ;
#endif
					}
				} 
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					if (j==degree-1)
					{
#ifdef WEIGHTS_IN_TRIE
						LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]] ;
#else
						LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]]*weights[j] ;
#endif
					}
				}
			}
		} 
		else // with position_weigths, weights is a matrix (len x degree)
		{
			for (INT j=0; seq_pos+j<len; j++)
			{
				if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
				{
					if (TreeMem[tree].children[vec[seq_pos+j]]<0)
					{
						tree = -TreeMem[tree].children[vec[seq_pos+j]];
						TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
						for (INT k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
						{
							if (TreeMem[tree].seq[k]!=vec[seq_pos+j+k])
								break ;
#ifdef WEIGHTS_IN_TRIE
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
#else
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k+weight_pos*degree] ;
#endif
						}
						break ;
					}
					else
					{
						tree=TreeMem[tree].children[vec[seq_pos+j]];
						TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
#ifdef WEIGHTS_IN_TRIE
						LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
#else
						LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+weight_pos*degree] ;
#endif
					}
				} 
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					if (j==degree-1)
					{
#ifdef WEIGHTS_IN_TRIE
						LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]] ;
#else
						LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]]*weights[j+weight_pos*degree] ;
#endif
					}		  
					break ;
				}
			}
		} 
    }
	else if (!degree_times_position_weights) // no position_weigths, weights is a vector (1 x degree)
    {
		for (INT j=0; seq_pos+j<len; j++)
		{
			if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
			{
				if (TreeMem[tree].children[vec[seq_pos+j]]<0)
				{
					tree = -TreeMem[tree].children[vec[seq_pos+j]];
					TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
					for (INT k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
					{
						if (TreeMem[tree].seq[k]!=vec[seq_pos+j+k])
							break ;
#ifdef WEIGHTS_IN_TRIE
						LevelContrib[(j+k)/mkl_stepsize] += factor*TreeMem[tree].weight ;
#else
						LevelContrib[(j+k)/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k] ;
#endif
					}
					break ;
				}
				else
				{
					tree=TreeMem[tree].children[vec[seq_pos+j]];
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
#ifdef WEIGHTS_IN_TRIE
					LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].weight ;
#else
					LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j] ;
#endif
				}
			}
			else
			{
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
				if (j==degree-1)
				{
#ifdef WEIGHTS_IN_TRIE
					LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]] ;
#else
					LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]]*weights[j] ;
#endif
				}
				break ;
			}
		} 
    } 
	else // no position_weigths, weights is a matrix (len x degree)
    {
		/*if (!position_mask)
		  {		
		  position_mask = new bool[len] ;
		  for (INT i=0; i<len; i++)
		  {
		  position_mask[i]=false ;
		  
		  for (INT j=0; j<degree; j++)
		  if (weights[i*degree+j]!=0.0)
		  {
		  position_mask[i]=true ;
		  break ;
		  }
		  }
		  }
		  if (position_mask[weight_pos]==0)
		  return ;*/
		
		for (INT j=0; seq_pos+j<len; j++)
		{
			if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
			{
				if (TreeMem[tree].children[vec[seq_pos+j]]<0)
				{
					tree = -TreeMem[tree].children[vec[seq_pos+j]];
					TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq) ;
					for (INT k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
					{
						if (TreeMem[tree].seq[k]!=vec[seq_pos+j+k])
							break ;
#ifdef WEIGHTS_IN_TRIE
						LevelContrib[(j+k+degree*weight_pos)/mkl_stepsize] += factor*TreeMem[tree].weight ;
#else
						LevelContrib[(j+k+degree*weight_pos)/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k+weight_pos*degree] ;
#endif
					}
					break ;
				}
				else
				{
					tree=TreeMem[tree].children[vec[seq_pos+j]];
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
#ifdef WEIGHTS_IN_TRIE
					LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].weight ;
#else
					LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].weight * weights[j+weight_pos*degree] ;
#endif
				} 
			}
			else
			{
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
				if (j==degree-1)
				{
#ifdef WEIGHTS_IN_TRIE
					LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].child_weights[vec[seq_pos+j]] ;
#else
					LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].child_weights[vec[seq_pos+j]] * weights[j+weight_pos*degree] ;
#endif
				}
				break ;
			}
		} 
    }
}


#endif
