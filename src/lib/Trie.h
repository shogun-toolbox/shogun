/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _TRIE_H___
#define _TRIE_H___

#include <string.h>
#include "lib/common.h"
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "base/SGObject.h"

//#define NO_CHILD ((INT)-2147483648)
#define NO_CHILD ((INT)-1073741824) 

#define WEIGHTS_IN_TRIE 
//#define TRIE_CHECK_EVERYTHING

#ifdef TRIE_CHECK_EVERYTHING
#define TRIE_ASSERT_EVERYTHING(x) ASSERT(x)
#else
#define TRIE_ASSERT_EVERYTHING(x) 
#endif

//#define TRIE_ASSERT(x) ASSERT(x)
#define TRIE_ASSERT(x) 

#define TRIE_TERMINAL_CHARACTER  7

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

class CTrie : public CSGObject
{
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
	void set_degree(INT d);
	void create(INT len, INT p_use_compact_terminal_nodes=true) ;
	void delete_trees(INT p_use_compact_terminal_nodes=true);
	void add_to_trie(int i, INT seq_offset, INT * vec, float alpha, DREAL *weights, bool degree_times_position_weights) ;
	DREAL compute_abs_weights_tree(INT tree, INT depth) ;
	DREAL* compute_abs_weights(int &len) ;

	DREAL compute_by_tree_helper(INT* vec, INT len, INT seq_pos, INT tree_pos, INT weight_pos, DREAL * weights, bool degree_times_position_weights) ;
	void compute_by_tree_helper(INT* vec, INT len, INT seq_pos, INT tree_pos, INT weight_pos, DREAL* LevelContrib, DREAL factor, INT mkl_stepsize, DREAL * weights, bool degree_times_position_weights) ;
	void compute_scoring_helper(INT tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result) ;
	void add_example_to_tree_mismatch_recursion(INT tree,  INT i, DREAL alpha, INT *vec, INT len_rem, INT degree_rec, INT mismatch_rec, INT max_mismatch, DREAL * weights) ;
	void traverse( INT tree, const INT p, struct TreeParseInfo info, const INT depth, INT* const x, const INT k ) ;
	void count( const DREAL w, const INT depth, const struct TreeParseInfo info, const INT p, INT* x, const INT k ) ;
	INT compact_nodes(INT start_node, INT depth, DREAL * weights) ;

	inline bool get_use_compact_terminal_nodes()
	{
		return use_compact_terminal_nodes ;
	}
	inline void set_use_compact_terminal_nodes(bool p_use_compact_terminal_nodes)
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
	}

	inline void check_treemem()
	{
		if (TreeMemPtr+10 < TreeMemPtrMax)
			return;
		SG_DEBUG( "Extending TreeMem from %i to %i elements\n",
				TreeMemPtrMax, (INT) ((double)TreeMemPtrMax*1.2));
		TreeMemPtrMax = (INT) ((double)TreeMemPtrMax*1.2);
		TreeMem = (struct Trie *)realloc(TreeMem,
				TreeMemPtrMax*sizeof(struct Trie));
		if (!TreeMem)
			SG_ERROR( "out of memory\n");
	}

	inline void set_weights_in_tree(bool weights_in_tree_)
	{
		weights_in_tree = weights_in_tree_ ;
	}

	inline bool set_weights_in_tree()
	{
		return weights_in_tree ;
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

	bool weights_in_tree ;
};

#endif
