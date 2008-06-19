/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _TRIE_H___
#define _TRIE_H___

#include <string.h>
#include "lib/common.h"
#include "lib/io.h"
#include "lib/DynamicArray.h"
#include "lib/Mathematics.h"
#include "base/SGObject.h"

// sentinel is 0xFFFFFFFC or float -2
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

/** consensus entry */
struct ConsensusEntry
{
	/** string */
	ULONG string;
	/** score */
	SHORTREAL score;
	/** bt */
	INT bt;
};

/** POIM trie */
struct POIMTrie
{
	/** weight */
	DREAL weight;
#ifdef TRIE_CHECK_EVERYTHING
	/** has sequence */
	bool has_seq;
	/** has floats */
	bool has_floats;
#endif		
	union
	{
		/** child weights */
		SHORTREAL child_weights[4];
		/** children */
		INT children[4];
		/** sequence */
		BYTE seq[16] ;
	}; 

	/** super_string_score */
	DREAL S;
	/** left_partial_overlap_score */
	DREAL L;
	/** right_partial_overlap_score */
	DREAL R;
};

/** DNA trie */
struct DNATrie
{
	/** weight */
	DREAL weight;
#ifdef TRIE_CHECK_EVERYTHING
	/** has sequence */
	bool has_seq;
	/** has floats */
	bool has_floats;
#endif
	union
	{
		/** child weights */
		SHORTREAL child_weights[4];
		/** children */
		INT children[4];
		/** sequence */
		BYTE seq[16] ;
	}; 
};

/** tree parse info */
struct TreeParseInfo {
	/** number of symbols */
	INT num_sym;
	/** number of features */
	INT num_feat;
	/** p */
	INT p;
	/** k */
	INT k;
	/** nofsKmers */
	INT* nofsKmers;
	/** margFactors */
	DREAL* margFactors;
	/** x */
	INT* x;
	/** substrs */
	INT* substrs;
	/** y0 */
	INT y0;
	/** C k */
	DREAL* C_k;
	/** L k */
	DREAL* L_k;
	/** R k */
	DREAL* R_k;
};


template <class Trie> class CTrie;

/** Template class Trie implements a suffix trie, i.e. a tree in which all
 * suffixes up to a certain length are stored. It is excessively used in the
 * CWeightedDegreeStringKernel and CWeightedDegreePositionStringKernel to
 * construct the whole features space \f$\Phi(x)\f$ and enormously helps here to
 * speed up SVM training and evaluation.
 *
 * Note that depending on the underlying structure used, a single symbol in the
 * tree requires 20 bytes (DNATrie). It is also used to do the efficient
 * recursion in computing positional oligomer importance matrices (POIMs) where
 * the structure requires * 20+3*8 (POIMTrie) bytes.
 *
 * Finally note that this try may use compact internal nodes (for strings that
 * appear without modifications, thus not requiring further branches), which
 * may save a lot of memory on higher degree tries.
 *
 */
template <class Trie> class CTrie : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param d degree
		 * @param p_use_compact_terminal_nodes if compact terminal nodes shall
		 *                                     be used
		 */
		CTrie(INT d, bool p_use_compact_terminal_nodes=true);

		/** copy constructor */
		CTrie(const CTrie & to_copy);
		~CTrie();

		/** overload operator = */
		const CTrie & operator=(const CTrie & to_copy);

		/** compare traverse
		 *
		 * @param node node
		 * @param other other trie
		 * @param other_node other node
		 * @return if comparison was successful
		 */
		bool compare_traverse(INT node, const CTrie & other, INT other_node);

		/** compare
		 *
		 * @param other other trie
		 * @return if comparison was successful
		 */
		bool compare(const CTrie & other);

		/** find node
		 *
		 * @param node node to find
		 * @param trace trace
		 * @param trace_len length of trace
		 */
		bool find_node(INT node, INT * trace, INT &trace_len) const;

		/** find deepest node
		 *
		 * @param start_node start node
		 * @param deepest_node deepest node will be stored in here
		 * @return depth of deepest node
		 */
		INT find_deepest_node(INT start_node, INT &deepest_node) const;

		/** display node
		 *
		 * @param node node to display
		 */
		void display_node(INT node) const;

		/** destroy */
		void destroy();

		/** set degree
		 *
		 * @param d new degree
		 */
		void set_degree(INT d);

		/** create
		 *
		 * @param len length of new trie
		 * @param p_use_compact_terminal_nodes if compact terminal nodes shall
		 *                                     be used
		 */
		void create(INT len, bool p_use_compact_terminal_nodes=true);

		/** delete trees
		 *
		 * @param p_use_compact_terminal_nodes if compact terminal nodes shall
		 *                                     be used
		 */
		void delete_trees(bool p_use_compact_terminal_nodes=true);

		/** add to trie
		 *
		 * @param i i
		 * @param seq_offset sequence offset
		 * @param vec vector
		 * @param alpha alpha
		 * @param weights weights
		 * @param degree_times_position_weights if degree times position
		 *                                      weights shall be applied
		 */
		void add_to_trie(int i, INT seq_offset, INT * vec, float alpha, DREAL *weights, bool degree_times_position_weights);

		/** compute absolute weights tree
		 *
		 * @param tree tree to compute for
		 * @param depth depth
		 * @return computed absolute weights tree
		 */
		DREAL compute_abs_weights_tree(INT tree, INT depth);

		/** compute absolute weights
		 *
		 * @param len length
		 * @return computed absolute weights
		 */
		DREAL* compute_abs_weights(int &len);

		/** compute by tree helper
		 *
		 * @param vec vector
		 * @param len length
		 * @param seq_pos sequence position
		 * @param tree_pos tree position
		 * @param weight_pos weight position
		 * @param weights
		 * @param degree_times_position_weights if degree times position
		 *                                      weights shall be applied
		 * @return a computed value
		 */
		DREAL compute_by_tree_helper(INT* vec, INT len, INT seq_pos, INT tree_pos, INT weight_pos, DREAL * weights, bool degree_times_position_weights) ;

		/** compute by tree helper
		 *
		 * @param vec vector
		 * @param len length
		 * @param seq_pos sequence position
		 * @param tree_pos tree position
		 * @param weight_pos weight position
		 * @param LevelContrib level contribution
		 * @param factor factor
		 * @param mkl_stepsize MKL stepsize
		 * @param weights
		 * @param degree_times_position_weights if degree times position
		 *                                      weights shall be applied
		 */
		void compute_by_tree_helper(INT* vec, INT len, INT seq_pos, INT tree_pos, INT weight_pos, DREAL* LevelContrib, DREAL factor, INT mkl_stepsize, DREAL * weights, bool degree_times_position_weights);

		/** compute scoring helper
		 *
		 * @param tree tree
		 * @param i i
		 * @param j j
		 * @param weight weight
		 * @param d degree
		 * @param max_degree maximum degree
		 * @param num_feat number of features
		 * @param num_sym number of symbols
		 * @param sym_offset symbol offset
		 * @param offs offsets
		 * @param result result
		 */
		void compute_scoring_helper(INT tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result);

		/** add example to tree mismatch recursion
		 *
		 * @param tree tree
		 * @param i i
		 * @param alpha alpha
		 * @param vec vector
		 * @param len_rem length of rem
		 * @param degree_rec degree rec
		 * @param mismatch_rec mismatch rec
		 * @param max_mismatch maximum mismatch
		 * @param weights weights
		 */
		void add_example_to_tree_mismatch_recursion(INT tree,  INT i, DREAL alpha, INT *vec, INT len_rem, INT degree_rec, INT mismatch_rec, INT max_mismatch, DREAL * weights);

		/** traverse
		 *
		 * @param tree tree
		 * @param p p
		 * @param info tree parse info
		 * @param depth depth
		 * @param x x
		 * @param k k
		 */
		void traverse(INT tree, const INT p, struct TreeParseInfo info, const INT depth, INT* const x, const INT k);

		/** count
		 *
		 * @param w w
		 * @param depth depth
		 * @param info tree parse info
		 * @param p p
		 * @param x x
		 * @param k
		 */
		void count(const DREAL w, const INT depth, const struct TreeParseInfo info, const INT p, INT* x, const INT k);

		/** compact nodes
		 *
		 * @param start_node start node
		 * @param depth depth
		 * @param weights weights
		 */
		INT compact_nodes(INT start_node, INT depth, DREAL * weights);

		/** get cumulative score
		 *
		 * @param pos position
		 * @param seq sequence
		 * @param deg degree
		 * @param weights weights
		 * @return cumulative score
		 */
		DREAL get_cumulative_score(INT pos, ULONG seq, INT deg, DREAL* weights);

		/** fill backtracking table recursion
		 *
		 * @param tree tree
		 * @param depth depth
		 * @param seq sequence
		 * @param value value
		 * @param table table of concensus entries
		 * @param weights weights
		 */
		void fill_backtracking_table_recursion(Trie* tree, INT depth, ULONG seq, DREAL value, CDynamicArray<ConsensusEntry>* table, DREAL* weights);

		/** fill backtracking table
		 *
		 * @param pos position
		 * @param prev previous concencus entry
		 * @param cur current concensus entry
		 * @param cumulative if is cumulative
		 * @param weights weights
		 */
		void fill_backtracking_table(INT pos, CDynamicArray<ConsensusEntry>* prev, CDynamicArray<ConsensusEntry>* cur, bool cumulative, DREAL* weights);

		/** POIMs extract W
		 *
		 * @param W W
		 * @param K K
		 */
		void POIMs_extract_W(DREAL* const* const W, const INT K);

		/** POIMs precalc SLR
		 *
		 * @param distrib distribution
		 */
		void POIMs_precalc_SLR(const DREAL* const distrib);

		/** POIMs get SLR
		 *
		 * @param parentIdx parent index
		 * @param sym symbol
		 * @param depth depth
		 * @param S will point to S
		 * @param L will point to L
		 * @param R will point to R
		 */
		void POIMs_get_SLR(const INT parentIdx, const INT sym, const int depth, DREAL* S, DREAL* L, DREAL* R);

		/** POIMs add SLR
		 *
		 * @param poims POIMs
		 * @param K K
		 * @param debug debug level
		 */
		void POIMs_add_SLR(DREAL* const* const poims, const INT K, const INT debug);

		/** get use compact terminal nodes
		 *
		 * @return if compact terminal nodes are used
		 */
		inline bool get_use_compact_terminal_nodes()
		{
			return use_compact_terminal_nodes ;
		}

		/** set use compact terminal nodes
		 *
		 * @param p_use_compact_terminal_nodes if compact terminal nodes shall
		 *                                     be used
		 */
		inline void set_use_compact_terminal_nodes(bool p_use_compact_terminal_nodes)
		{
			use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
		}

		/** get number of used nodes
		 *
		 * @return number of used nodes
		 */
		inline INT get_num_used_nodes()
		{
			return TreeMemPtr;
		}

		/** set position weights
		 *
		 * @param p_position_weights new position weights
		 */
		inline void set_position_weights(const DREAL * p_position_weights)
		{
			position_weights=p_position_weights;
		}

		/** get node
		 *
		 * @return node
		 */
		inline INT get_node(bool last_node=false)
		{
			INT ret = TreeMemPtr++;
			check_treemem() ;

			if (last_node)
			{
				for (INT q=0; q<4; q++)
					TreeMem[ret].child_weights[q]=0.0;
			}
			else
			{
				for (INT q=0; q<4; q++)
					TreeMem[ret].children[q]=NO_CHILD;
			}
#ifdef TRIE_CHECK_EVERYTHING
			TreeMem[ret].has_seq=false ;
			TreeMem[ret].has_floats=false ;
#endif
			TreeMem[ret].weight=0.0;
			return ret ;
		}

		/** check tree memory usage */
		inline void check_treemem()
		{
			if (TreeMemPtr+10 < TreeMemPtrMax)
				return;
			SG_DEBUG( "Extending TreeMem from %i to %i elements\n",
					TreeMemPtrMax, (INT) ((double)TreeMemPtrMax*1.2));
			TreeMemPtrMax = (INT) ((double)TreeMemPtrMax*1.2);
			TreeMem = (Trie*) realloc(TreeMem,
					TreeMemPtrMax*sizeof(Trie));
			if (!TreeMem)
				SG_ERROR( "out of memory\n");
		}

		/** set weights in tree
		 *
		 * @param weights_in_tree_ if weights shall be in tree
		 */
		inline void set_weights_in_tree(bool weights_in_tree_)
		{
			weights_in_tree = weights_in_tree_;
		}

		/** get weights in tree
		 *
		 * @return if weights are in tree
		 */
		inline bool get_weights_in_tree()
		{
			return weights_in_tree;
		}

		/** POIMs extract W helper
		 *
		 * @param nodeIdx node index
		 * @param depth depth
		 * @param offset offset
		 * @param y0 y0
		 * @param W W
		 * @param K K
		 */
		void POIMs_extract_W_helper(const INT nodeIdx, const int depth, const INT offset, const INT y0, DREAL* const* const W, const INT K);

		/** POIMs calc SLR helper
		 *
		 * @param distrib distribution
		 * @param i i
		 * @param nodeIdx node index
		 * @param left_tries_idx left tries index
		 * @param depth depth
		 * @param lastSym last symbol
		 * @param S S
		 * @param L L
		 * @param R R
		 */
		void POIMs_calc_SLR_helper1(const DREAL* const distrib, const INT i, const INT nodeIdx, INT left_tries_idx[4], const int depth, INT const lastSym, DREAL* S, DREAL* L, DREAL* R);

		/** POIMs calc SLR helper 2
		 * @param distrib distribution
		 * @param i i
		 * @param nodeIdx node index
		 * @param left_tries_idx left tries index
		 * @param depth depth
		 * @param S S
		 * @param L L
		 * @param R R
		 */
		void POIMs_calc_SLR_helper2(const DREAL* const distrib, const INT i, const INT nodeIdx, INT left_tries_idx[4], const int depth, DREAL* S, DREAL* L, DREAL* R);

		/** POIMs add SLR helper 1
		 *
		 * @param nodeIdx node index
		 * @param depth depth
		 * @param i i
		 * @param y0 y0
		 * @param poims POIMs
		 * @param K K
		 * @param debug debug level
		 */
		void POIMs_add_SLR_helper1(const INT nodeIdx, const int depth, const INT i, const INT y0, DREAL* const* const poims, const INT K, const INT debug);

		/** POIMs add SLR helper 2
		 *
		 * @param poims POIMs
		 * @param K K
		 * @param k k
		 * @param i i
		 * @param y y
		 * @param valW value W
		 * @param valS value S
		 * @param valL value L
		 * @param valR value R
		 * @param debug debug level
		 */
		void POIMs_add_SLR_helper2(DREAL* const* const poims, const int K, const int k, const INT i, const INT y, const DREAL valW, const DREAL valS, const DREAL valL, const DREAL valR, const INT debug);

	public:
		/** number of symbols */
		INT NUM_SYMS;

	protected:
		/** length */
		INT length;
		/** trees */
		INT * trees;

		/** degree */
		INT degree;
		/** position weights */
		DREAL const *  position_weights;

		/** tree memory */
		Trie* TreeMem;
		/** tree memory pointer */
		INT TreeMemPtr;
		/** tree memory pointer maximum */
		INT TreeMemPtrMax;
		/** if compact terminal nodes are used */
		bool use_compact_terminal_nodes;

		/** if weights are in tree */
		bool weights_in_tree;

		/** nofsKmers */
		INT* nofsKmers;
};

	template <class Trie>
	CTrie<Trie>::CTrie(INT d, bool p_use_compact_terminal_nodes)
	: CSGObject(), degree(d), position_weights(NULL),
		use_compact_terminal_nodes(p_use_compact_terminal_nodes),
		weights_in_tree(true)
	{
		TreeMemPtrMax=1024*1024/sizeof(Trie);
		TreeMemPtr=0;
		TreeMem=(Trie*)malloc(TreeMemPtrMax*sizeof(Trie));

		length=0;
		trees=NULL;

		NUM_SYMS=4;
	}

	template <class Trie>
	CTrie<Trie>::CTrie(const CTrie & to_copy)
	: CSGObject(to_copy), degree(to_copy.degree), position_weights(NULL),
		use_compact_terminal_nodes(to_copy.use_compact_terminal_nodes)
	{
		if (to_copy.position_weights!=NULL)
		{
			position_weights = to_copy.position_weights;
			/*new DREAL[to_copy.length];
			  for (INT i=0; i<to_copy.length; i++)
			  position_weights[i]=to_copy.position_weights[i]; */
		}
		else
			position_weights=NULL;

		TreeMemPtrMax=to_copy.TreeMemPtrMax;
		TreeMemPtr=to_copy.TreeMemPtr;
		TreeMem=(Trie*)malloc(TreeMemPtrMax*sizeof(Trie));
		memcpy(TreeMem, to_copy.TreeMem, TreeMemPtrMax*sizeof(Trie));

		length=to_copy.length;
		trees=new INT[length];
		for (INT i=0; i<length; i++)
			trees[i]=to_copy.trees[i];

		NUM_SYMS=4;
	}

	template <class Trie>
const CTrie<Trie> &CTrie<Trie>::operator=(const CTrie<Trie> & to_copy)
{
	degree=to_copy.degree ;
	use_compact_terminal_nodes=to_copy.use_compact_terminal_nodes ;

	delete[] position_weights ;
	position_weights=NULL ;
	if (to_copy.position_weights!=NULL)
	{
		position_weights=to_copy.position_weights ;
		/*position_weights = new DREAL[to_copy.length] ;
		  for (INT i=0; i<to_copy.length; i++)
		  position_weights[i]=to_copy.position_weights[i] ;*/
	}
	else
		position_weights=NULL ;

	TreeMemPtrMax=to_copy.TreeMemPtrMax ;
	TreeMemPtr=to_copy.TreeMemPtr ;
	free(TreeMem) ;
	TreeMem = (Trie*)malloc(TreeMemPtrMax*sizeof(Trie)) ;
	memcpy(TreeMem, to_copy.TreeMem, TreeMemPtrMax*sizeof(Trie)) ;

	length = to_copy.length ;
	if (trees)
		delete[] trees ;
	trees=new INT[length] ;		
	for (INT i=0; i<length; i++)
		trees[i]=to_copy.trees[i] ;

	return *this ;
}

template <class Trie>
INT CTrie<Trie>::find_deepest_node(INT start_node, INT& deepest_node) const 
{
#ifdef TRIE_CHECK_EVERYTHING
	INT ret=0 ;
	SG_DEBUG("start_node=%i\n", start_node) ;

	if (start_node==NO_CHILD) 
	{
		for (INT i=0; i<length; i++)
		{
			INT my_deepest_node ;
			INT depth=find_deepest_node(i, my_deepest_node) ;
			SG_DEBUG("start_node %i depth=%i\n", i, depth) ;
			if (depth>ret)
			{
				deepest_node=my_deepest_node ;
				ret=depth ;
			}
		}
		return ret ;
	}

	if (TreeMem[start_node].has_seq)
	{
		for (INT q=0; q<16; q++)
			if (TreeMem[start_node].seq[q]!=TRIE_TERMINAL_CHARACTER)
				ret++ ;
		deepest_node=start_node ;
		return ret ;
	}
	if (TreeMem[start_node].has_floats)
	{
		deepest_node=start_node ;
		return 1 ;
	}

	for (INT q=0; q<4; q++)
	{
		INT my_deepest_node ;
		if (TreeMem[start_node].children[q]==NO_CHILD)
			continue ;
		INT depth=find_deepest_node(abs(TreeMem[start_node].children[q]), my_deepest_node) ;
		if (depth>ret)
		{
			deepest_node=my_deepest_node ;
			ret=depth ;
		}
	}
	return ret ;
#else
	SG_ERROR( "not implemented\n") ;
	return 0 ;
#endif
}

	template <class Trie>
INT CTrie<Trie>::compact_nodes(INT start_node, INT depth, DREAL * weights) 
{
	SG_ERROR( "code buggy\n") ;

	INT ret=0 ;

	if (start_node==NO_CHILD) 
	{
		for (INT i=0; i<length; i++)
			compact_nodes(i,1, weights) ;
		return 0 ;
	}
	if (start_node<0)
		return -1 ;

	if (depth==degree-1)
	{
		TRIE_ASSERT_EVERYTHING(TreeMem[start_node].has_floats) ;
		INT num_used=0 ;
		for (INT q=0; q<4; q++)
			if (TreeMem[start_node].child_weights[q]!=0.0)
				num_used++ ;
		if (num_used>1)
			return -1 ;
		return 1 ;
	}
	TRIE_ASSERT_EVERYTHING(!TreeMem[start_node].has_floats) ;

	INT num_used = 0 ;
	INT q_used=-1 ;

	for (INT q=0; q<4; q++)
	{
		if (TreeMem[start_node].children[q]==NO_CHILD)
			continue ;
		num_used++ ;
		q_used=q ;
	}
	if (num_used>1)
	{
		if (depth>=degree-2)
			return -1 ;
		for (INT q=0; q<4; q++)
		{
			if (TreeMem[start_node].children[q]==NO_CHILD)
				continue ;
			INT num=compact_nodes(abs(TreeMem[start_node].children[q]), depth+1, weights) ;
			if (num<=2)
				continue ;
			INT node=get_node() ;

			INT last_node=TreeMem[start_node].children[q] ;
			if (weights_in_tree)
			{
				ASSERT(weights[depth]!=0.0) ;
				TreeMem[node].weight=TreeMem[last_node].weight/weights[depth] ;
			}
			else
				TreeMem[node].weight=TreeMem[last_node].weight ;

#ifdef TRIE_CHECK_EVERYTHING
			TreeMem[node].has_seq=true ;
#endif
			memset(TreeMem[node].seq, TRIE_TERMINAL_CHARACTER, 16) ;
			for (INT n=0; n<num; n++)
			{
				ASSERT(depth+n+1<=degree-1) ;
				ASSERT(last_node!=NO_CHILD) ;
				if (depth+n+1==degree-1)
				{
					TRIE_ASSERT_EVERYTHING(TreeMem[last_node].has_floats) ;
					INT  k ;
					for (k=0; k<4; k++)
						if (TreeMem[last_node].child_weights[k]!=0.0)
							break ;
					if (k==4)
						break ;
					TreeMem[node].seq[n]=k ;
					break ;
				}
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[last_node].has_floats) ;
					INT k ;
					for (k=0; k<4; k++)
						if (TreeMem[last_node].children[k]!=NO_CHILD)
							break ;
					if (k==4)
						break ;
					TreeMem[node].seq[n]=k ;
					last_node=TreeMem[last_node].children[k] ;
				}
			}
			TreeMem[start_node].children[q]=-node ;
		}
		return -1 ;
	}
	if (num_used==0)
		return 0 ;

	ret=compact_nodes(abs(TreeMem[start_node].children[q_used]), depth+1, weights) ;
	if (ret<0)
		return ret ;
	return ret+1 ;
}


	template <class Trie>
bool CTrie<Trie>::compare_traverse(INT node, const CTrie<Trie> & other, INT other_node) 
{
	SG_DEBUG("checking nodes %i and %i\n", node, other_node) ;
	if (fabs(TreeMem[node].weight-other.TreeMem[other_node].weight)>=1e-5)
	{
		SG_DEBUG( "CTrie::compare: TreeMem[%i].weight=%f!=other.TreeMem[%i].weight=%f\n", node, TreeMem[node].weight, other_node,other.TreeMem[other_node].weight) ;
		SG_DEBUG( ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
		display_node(node) ;
		SG_DEBUG( "============================================================\n") ;			
		other.display_node(other_node) ;
		SG_DEBUG( "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
		return false ;
	}

#ifdef TRIE_CHECK_EVERYTHING
	if (TreeMem[node].has_seq!=other.TreeMem[other_node].has_seq)
	{
		SG_DEBUG( "CTrie::compare: TreeMem[%i].has_seq=%i!=other.TreeMem[%i].has_seq=%i\n", node, TreeMem[node].has_seq, other_node,other.TreeMem[other_node].has_seq) ;
		SG_DEBUG( ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
		display_node(node) ;
		SG_DEBUG( "============================================================\n") ;			
		other.display_node(other_node) ;
		SG_DEBUG( "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;
		return false ;
	}
	if (TreeMem[node].has_floats!=other.TreeMem[other_node].has_floats)
	{
		SG_DEBUG( "CTrie::compare: TreeMem[%i].has_floats=%i!=other.TreeMem[%i].has_floats=%i\n", node, TreeMem[node].has_floats, other_node, other.TreeMem[other_node].has_floats) ;
		return false ;
	}
	if (other.TreeMem[other_node].has_floats)
	{
		for (INT q=0; q<4; q++)
			if (fabs(TreeMem[node].child_weights[q]-other.TreeMem[other_node].child_weights[q])>1e-5)
			{
				SG_DEBUG( "CTrie::compare: TreeMem[%i].child_weights[%i]=%e!=other.TreeMem[%i].child_weights[%i]=%e\n", node, q,TreeMem[node].child_weights[q], other_node,q,other.TreeMem[other_node].child_weights[q]) ;
				SG_DEBUG( ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
				display_node(node) ;
				SG_DEBUG( "============================================================\n") ;			
				other.display_node(other_node) ;
				SG_DEBUG( "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
				return false ;
			}
	}
	if (other.TreeMem[other_node].has_seq)
	{
		for (INT q=0; q<16; q++)
			if ((TreeMem[node].seq[q]!=other.TreeMem[other_node].seq[q]) && ((TreeMem[node].seq[q]<4)||(other.TreeMem[other_node].seq[q]<4)))
			{
				SG_DEBUG( "CTrie::compare: TreeMem[%i].seq[%i]=%i!=other.TreeMem[%i].seq[%i]=%i\n", node,q,TreeMem[node].seq[q], other_node,q,other.TreeMem[other_node].seq[q]) ;
				SG_DEBUG( ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
				display_node(node) ;
				SG_DEBUG( "============================================================\n") ;			
				other.display_node(other_node) ;
				SG_DEBUG( "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;			
				return false ;
			}
	}
	if (!other.TreeMem[other_node].has_seq && !other.TreeMem[other_node].has_floats)
	{
		for (INT q=0; q<4; q++)
		{
			if ((TreeMem[node].children[q]==NO_CHILD) && (other.TreeMem[other_node].children[q]==NO_CHILD))
				continue ;
			if ((TreeMem[node].children[q]==NO_CHILD)!=(other.TreeMem[other_node].children[q]==NO_CHILD))
			{
				SG_DEBUG( "CTrie::compare: TreeMem[%i].children[%i]=%i!=other.TreeMem[%i].children[%i]=%i\n", node,q,TreeMem[node].children[q], other_node,q,other.TreeMem[other_node].children[q]) ;
				SG_DEBUG( ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n") ;			
				display_node(node) ;
				SG_DEBUG( "============================================================\n") ;			
				other.display_node(other_node) ;
				SG_DEBUG( "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n") ;
				return false ;
			}
			if (!compare_traverse(abs(TreeMem[node].children[q]), other, abs(other.TreeMem[other_node].children[q])))
				return false ;
		}
	}
#else
	SG_ERROR( "not implemented\n") ;
#endif

	return true ;
}

	template <class Trie>
bool CTrie<Trie>::compare(const CTrie<Trie> & other)
{
	bool ret=true ;
	for (INT i=0; i<length; i++)
		if (!compare_traverse(trees[i], other, other.trees[i]))
			return false ;
		else
			SG_DEBUG("two tries at %i identical\n", i) ;

	return ret ;
}

template <class Trie>
bool CTrie<Trie>::find_node(INT node, INT * trace, INT& trace_len) const 
{
#ifdef TRIE_CHECK_EVERYTHING
	ASSERT(trace_len-1>=0) ;
	ASSERT((trace[trace_len-1]>=0) && (trace[trace_len-1]<TreeMemPtrMax))
		if (TreeMem[trace[trace_len-1]].has_seq)
			return false ;
	if (TreeMem[trace[trace_len-1]].has_floats)
		return false ;

	for (INT q=0; q<4; q++)
	{
		if (TreeMem[trace[trace_len-1]].children[q]==NO_CHILD)
			continue ;
		INT tl=trace_len+1 ;
		if (TreeMem[trace[trace_len-1]].children[q]>=0)
			trace[trace_len]=TreeMem[trace[trace_len-1]].children[q] ;
		else
			trace[trace_len]=-TreeMem[trace[trace_len-1]].children[q] ;

		if (trace[trace_len]==node)
		{
			trace_len=tl ;
			return true ;
		}
		if (find_node(node, trace, tl))
		{
			trace_len=tl ;
			return true ;
		}
	}
	trace_len=0 ;
	return false ;
#else
	SG_ERROR( "not implemented\n") ;
	return false ;
#endif
}

template <class Trie>
void CTrie<Trie>::display_node(INT node) const
{
#ifdef TRIE_CHECK_EVERYTHING
	INT * trace=new INT[2*degree] ;
	INT trace_len=-1 ;
	bool found = false ;
	INT tree=-1 ;
	for (tree=0; tree<length; tree++)
	{
		trace[0]=trees[tree] ;
		trace_len=1 ;
		found=find_node(node, trace, trace_len) ;
		if (found)
			break ;
	}
	ASSERT(found) ;
	SG_PRINT( "position %i  trace: ", tree) ;

	for (INT i=0; i<trace_len-1; i++)
	{
		INT branch=-1 ;
		for (INT q=0; q<4; q++)
			if (abs(TreeMem[trace[i]].children[q])==trace[i+1])
			{
				branch=q;
				break ;
			}
		ASSERT(branch!=-1) ;
		char acgt[5]="ACGT" ;
		SG_PRINT( "%c", acgt[branch]) ;
	}
	SG_PRINT( "\nnode=%i\nweight=%f\nhas_seq=%i\nhas_floats=%i\n", node, TreeMem[node].weight, TreeMem[node].has_seq, TreeMem[node].has_floats) ;
	if (TreeMem[node].has_floats)
	{
		for (INT q=0; q<4; q++)
			SG_PRINT( "child_weighs[%i] = %f\n", q, TreeMem[node].child_weights[q]) ;
	}
	if (TreeMem[node].has_seq)
	{
		for (INT q=0; q<16; q++)
			SG_PRINT( "seq[%i] = %i\n", q, TreeMem[node].seq[q]) ;
	}
	if (!TreeMem[node].has_seq && !TreeMem[node].has_floats)
	{
		for (INT q=0; q<4; q++)
		{
			if (TreeMem[node].children[q]!=NO_CHILD)
			{
				SG_PRINT( "children[%i] = %i -> \n", q, TreeMem[node].children[q]) ;
				display_node(abs(TreeMem[node].children[q])) ;
			}
			else
				SG_PRINT( "children[%i] = NO_CHILD -| \n", q, TreeMem[node].children[q]) ;
		}

	}

	delete[] trace ;
#else
	SG_ERROR( "not implemented\n") ;
#endif
}


template <class Trie> CTrie<Trie>::~CTrie()
{
	destroy() ;

	free(TreeMem) ;
}

template <class Trie> void CTrie<Trie>::destroy()
{
	if (trees!=NULL)
	{
		delete_trees();
		for (INT i=0; i<length; i++)
			trees[i] = NO_CHILD;
		delete[] trees;

		TreeMemPtr=0;
		length=0;
		trees=NULL;
	}
}

template <class Trie> void CTrie<Trie>::set_degree(INT d)
{
	delete_trees(get_use_compact_terminal_nodes());
	degree=d;
}

template <class Trie> void CTrie<Trie>::create(INT len, bool p_use_compact_terminal_nodes)
{
	destroy();

	trees=new INT[len] ;		
	TreeMemPtr=0 ;
	for (INT i=0; i<len; i++)
		trees[i]=get_node(degree==1);
	length = len ;

	use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
}


template <class Trie> void CTrie<Trie>::delete_trees(bool p_use_compact_terminal_nodes)
{
	if (trees==NULL)
		return;

	TreeMemPtr=0 ;
	for (INT i=0; i<length; i++)
		trees[i]=get_node(degree==1);

	use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
} 

	template <class Trie>
DREAL CTrie<Trie>::compute_abs_weights_tree(INT tree, INT depth) 
{
	DREAL ret=0 ;

	if (tree==NO_CHILD)
		return 0 ;
	TRIE_ASSERT(tree>=0) ;

	if (depth==degree-2)
	{
		ret+=(TreeMem[tree].weight) ;

		for (INT k=0; k<4; k++)
			ret+=(TreeMem[tree].child_weights[k]) ;

		return ret ;
	}

	ret+=(TreeMem[tree].weight) ;

	for (INT i=0; i<4; i++)
		if (TreeMem[tree].children[i]!=NO_CHILD)
			ret += compute_abs_weights_tree(TreeMem[tree].children[i], depth+1)  ;

	return ret ;
}


	template <class Trie>
DREAL *CTrie<Trie>::compute_abs_weights(int &len) 
{
	DREAL * sum=new DREAL[length*4] ;
	for (INT i=0; i<length*4; i++)
		sum[i]=0 ;
	len=length ;

	for (INT i=0; i<length; i++)
	{
		TRIE_ASSERT(trees[i]!=NO_CHILD) ;
		for (INT k=0; k<4; k++)
		{
			sum[i*4+k]=compute_abs_weights_tree(TreeMem[trees[i]].children[k], 0) ;
		}
	}

	return sum ;
}

	template <class Trie>
void CTrie<Trie>::add_example_to_tree_mismatch_recursion(INT tree,  INT i, DREAL alpha,
		INT *vec, INT len_rem, 
		INT degree_rec, INT mismatch_rec, 
		INT max_mismatch, DREAL * weights) 
{
	if (tree==NO_CHILD)
		tree=trees[i] ;
	TRIE_ASSERT(tree!=NO_CHILD) ;

	if ((len_rem<=0) || (mismatch_rec>max_mismatch) || (degree_rec>degree))
		return ;
	const INT other[4][3] = {	{1,2,3},{0,2,3},{0,1,3},{0,1,2} } ;

	INT subtree = NO_CHILD ;

	if (degree_rec==degree-1)
	{
		TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats) ;
		if (weights_in_tree)
			TreeMem[tree].child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec];
		else
			if (weights[degree_rec]!=0.0)
				TreeMem[tree].child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec]/weights[degree_rec];
		if (mismatch_rec+1<=max_mismatch)
			for (INT o=0; o<3; o++)
			{
				if (weights_in_tree)
					TreeMem[tree].child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
				else
					if (weights[degree_rec]!=0.0)
						TreeMem[tree].child_weights[other[vec[0]][o]] += alpha*weights[degree_rec+degree*(mismatch_rec+1)]/weights[degree_rec];
			}
		return ;
	}
	else
	{
		TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
		if (TreeMem[tree].children[vec[0]]!=NO_CHILD)
		{
			subtree=TreeMem[tree].children[vec[0]] ;
			if (weights_in_tree)
				TreeMem[subtree].weight += alpha*weights[degree_rec+degree*mismatch_rec];
			else
				if (weights[degree_rec]!=0.0)
					TreeMem[subtree].weight += alpha*weights[degree_rec+degree*mismatch_rec]/weights[degree_rec];
		}
		else 
		{
			INT tmp = get_node(degree_rec==degree-2);
			ASSERT(tmp>=0) ;
			TreeMem[tree].children[vec[0]]=tmp ;
			subtree=tmp ;
#ifdef TRIE_CHECK_EVERYTHING
			if (degree_rec==degree-2)
				TreeMem[subtree].has_floats=true ;
#endif
			if (weights_in_tree)
				TreeMem[subtree].weight = alpha*weights[degree_rec+degree*mismatch_rec] ;
			else
				if (weights[degree_rec]!=0.0)
					TreeMem[subtree].weight = alpha*weights[degree_rec+degree*mismatch_rec]/weights[degree_rec] ;
				else
					TreeMem[subtree].weight = 0.0 ;
		}
		add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
				&vec[1], len_rem-1, 
				degree_rec+1, mismatch_rec, max_mismatch, weights) ;

		if (mismatch_rec+1<=max_mismatch)
		{
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;
			for (INT o=0; o<3; o++)
			{
				INT ot = other[vec[0]][o] ;
				if (TreeMem[tree].children[ot]!=NO_CHILD)
				{
					subtree=TreeMem[tree].children[ot] ;
					if (weights_in_tree)
						TreeMem[subtree].weight += alpha*weights[degree_rec+degree*(mismatch_rec+1)];
					else
						if (weights[degree_rec]!=0.0)
							TreeMem[subtree].weight += alpha*weights[degree_rec+degree*(mismatch_rec+1)]/weights[degree_rec];
				}
				else 
				{
					INT tmp = get_node(degree_rec==degree-2);
					ASSERT(tmp>=0) ;
					TreeMem[tree].children[ot]=tmp ;
					subtree=tmp ;
#ifdef TRIE_CHECK_EVERYTHING
					if (degree_rec==degree-2)
						TreeMem[subtree].has_floats=true ;
#endif

					if (weights_in_tree)
						TreeMem[subtree].weight = alpha*weights[degree_rec+degree*(mismatch_rec+1)] ;
					else
						if (weights[degree_rec]!=0.0)
							TreeMem[subtree].weight = alpha*weights[degree_rec+degree*(mismatch_rec+1)]/weights[degree_rec] ;
						else
							TreeMem[subtree].weight = 0.0 ;
				}

				add_example_to_tree_mismatch_recursion(subtree,  i, alpha,
						&vec[1], len_rem-1, 
						degree_rec+1, mismatch_rec+1, max_mismatch, weights) ;
			}
		}
	}
}

	template <class Trie>
void CTrie<Trie>::compute_scoring_helper(INT tree, INT i, INT j, DREAL weight, INT d, INT max_degree, INT num_feat, INT num_sym, INT sym_offset, INT offs, DREAL* result)
{
	if (i+j<num_feat)
	{
		DREAL decay=1.0; //no decay by default
		//if (j>d)
		//	decay=pow(0.5,j); //marginalize out lower order matches

		if (j<degree-1)
		{
			for (INT k=0; k<num_sym; k++)
			{
				if (TreeMem[tree].children[k]!=NO_CHILD)
				{
					INT child=TreeMem[tree].children[k];
					//continue recursion if not yet at max_degree, else add to result
					if (d<max_degree-1)
						compute_scoring_helper(child, i, j+1, weight+decay*TreeMem[child].weight, d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
					else
						result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight+decay*TreeMem[child].weight;

					////do recursion starting from this position
					if (d==0)
						compute_scoring_helper(child, i, j+1, 0.0, 0, max_degree, num_feat, num_sym, sym_offset, offs, result);
				}
			}
		}
		else if (j==degree-1)
		{
			for (INT k=0; k<num_sym; k++)
			{
				//continue recursion if not yet at max_degree, else add to result
				if (d<max_degree-1 && i<num_feat-1)
					compute_scoring_helper(trees[i+1], i+1, 0, weight+decay*TreeMem[tree].child_weights[k], d+1, max_degree, num_feat, num_sym, sym_offset, num_sym*offs+k, result);
				else
					result[sym_offset*(i+j-max_degree+1)+num_sym*offs+k] += weight+decay*TreeMem[tree].child_weights[k];
			}
		}
	}
}

	template <class Trie>
void CTrie<Trie>::traverse( INT tree, const INT p, struct TreeParseInfo info, const INT depth, INT* const x, const INT k )
{
	const INT num_sym = info.num_sym;
	const INT y0 = info.y0;
	const INT y1 = (k==0) ? 0 : y0 - ( (depth<k) ? 0 : info.nofsKmers[k-1] * x[depth-k] );
	//const INT temp = info.substrs[depth]*num_sym - ( (depth<=k) ? 0 : info.nofsKmers[k] * x[depth-k-1] );
	//if( !( info.y0 == temp ) ) {
	//  printf( "\n temp=%d y0=%d k=%d depth=%d \n", temp, info.y0, k, depth );
	//}
	//ASSERT( info.y0 == temp );
	INT sym;
	ASSERT( depth < degree );
	//ASSERT( 0 <= info.substrs[depth] && info.substrs[depth] < info.nofsKmers[k] );
	if (depth<degree-1)
	{
		for( sym=0; sym<num_sym; ++sym ) {
			const INT childNum = TreeMem[tree].children[ sym ];
			if( childNum != NO_CHILD ) {
				INT child = childNum ;
				x[depth] = sym;
				info.substrs[depth+1] = y0 + sym;
				info.y0 = (k==0) ? 0 : (y1+sym)*num_sym;
				//ASSERT( info.y0 == ( info.substrs[depth+1]*num_sym - ( (depth<k) ? 0 : info.nofsKmers[k] * x[depth-k] ) ) );
				count( TreeMem[child].weight, depth, info, p, x, k );
				traverse( child, p, info, depth+1, x, k );
				x[depth] = -1;
			}
		}
	}
	else if( depth == degree-1 )
	{
		for( sym=0; sym<num_sym; ++sym ) {
			const DREAL w = TreeMem[tree].child_weights[ sym ];
			if( w != 0.0 ) {
				x[depth] = sym;
				info.substrs[depth+1] = y0 + sym;
				info.y0 = (k==0) ? 0 : (y1+sym)*num_sym;
				//ASSERT( info.y0 == ( info.substrs[depth+1]*num_sym - ( (depth<k) ? 0 : info.nofsKmers[k] * x[depth-k] ) ) );
				count( w, depth, info, p, x, k );
				x[depth] = -1;
			}
		}
	}
	//info.substrs[depth+1] = -1;
	//info.y0 = temp;
}

	template <class Trie>
void CTrie<Trie>::count( const DREAL w, const INT depth, const struct TreeParseInfo info, const INT p, INT* x, const INT k )
{
	ASSERT( fabs(w) < 1e10 );
	ASSERT( x[depth] >= 0 );
	ASSERT( x[depth+1] < 0 );
	if ( depth < k ) {
		return;
	}
	//ASSERT( info.margFactors[ depth-k ] == pow( 0.25, depth-k ) );
	const INT nofKmers = info.nofsKmers[k];
	const DREAL margWeight =  w * info.margFactors[ depth-k ];
	const INT m_a = depth - k + 1;
	const INT m_b = info.num_feat - p;
	const INT m = ( m_a < m_b ) ? m_a : m_b;
	// all proper k-substrings
	const INT offset0 = nofKmers * p;
	register INT i;
	register INT offset;
	offset = offset0;
	for( i = 0; i < m; ++i ) {
		const INT y = info.substrs[i+k+1];
		info.C_k[ y + offset ] += margWeight;
		offset += nofKmers;
	}
	if( depth > k ) {
		// k-prefix
		const INT offsR = info.substrs[k+1] + offset0;
		info.R_k[offsR] += margWeight;
		// k-suffix
		if( p+depth-k < info.num_feat ) {
			const INT offsL = info.substrs[depth+1] + nofKmers * (p+depth-k);
			info.L_k[offsL] += margWeight; 
		}
	}
	//    # N.x = substring represented by N
	//    # N.d = length of N.x
	//    # N.s = starting position of N.x
	//    # N.w = weight for feature represented by N
	//    if( N.d >= k )
	//      margContrib = w / 4^(N.d-k)
	//      for i = 1 to (N.d-k+1)
	//        y = N.x[i:(i+k-1)]  # overlapped k-mer
	//        C_k[ N.s+i-1, y ] += margContrib
	//      end;
	//      if( N.d > k )
	//        L_k[ N.s+N.d-k, N.x[N.d-k+(1:k)] ] += margContrib  # j-suffix of N.x
	//        R_k[ N.s,       N.x[1:k]         ] += margContrib  # j-prefix of N.x
	//      end;
	//    end;
}

	template <class Trie>
void CTrie<Trie>::add_to_trie(int i, INT seq_offset, INT * vec, float alpha, DREAL *weights, bool degree_times_position_weights)
{
	INT tree = trees[i] ;
	//ASSERT(seq_offset==0) ;

	INT max_depth = 0 ;
	DREAL* weights_column ;
	if (degree_times_position_weights)
		weights_column = &weights[(i+seq_offset)*degree] ;
	else
		weights_column = weights ;

	if (weights_in_tree)
	{
		for (INT j=0; (j<degree) && (i+j<length); j++)
			if (CMath::abs(weights_column[j]*alpha)>0)
				max_depth = j+1 ;
	}
	else
		// don't use the weights
		max_depth=degree ;

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
				{
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

						INT tmp=get_node();
						TreeMem[last_node].children[vec[i+j+seq_offset+k]]=tmp ;
						last_node=tmp ;
						if (weights_in_tree)
							TreeMem[last_node].weight = (TreeMem[node].weight+alpha)*weights_column[j+k] ;
						else
							TreeMem[last_node].weight = (TreeMem[node].weight+alpha) ;
						TRIE_ASSERT(j+k!=degree-1) ;
					}
					if ((TreeMem[node].seq[mismatch_pos]>=4) && (TreeMem[node].seq[mismatch_pos]!=TRIE_TERMINAL_CHARACTER))
						fprintf(stderr, "**i=%i j=%i seq[%i]=%i\n", i, j, k, TreeMem[node].seq[mismatch_pos]) ;
					ASSERT((TreeMem[node].seq[mismatch_pos]<4) || (TreeMem[node].seq[mismatch_pos]==TRIE_TERMINAL_CHARACTER)) ;
					TRIE_ASSERT(vec[i+j+seq_offset+mismatch_pos]!=TreeMem[node].seq[mismatch_pos]) ;

					if (j+k==degree-1)
					{
						// init child weights with zero if after dropping out
						// of the k<mismatch_pos loop we are one level below degree
						// (keep this even after get_node() change!)
						for (INT q=0; q<4; q++)
							TreeMem[last_node].child_weights[q]=0.0 ;
						if (weights_in_tree)
						{
							if (TreeMem[node].seq[mismatch_pos]<4) // i.e. !=TRIE_TERMINAL_CHARACTER
								TreeMem[last_node].child_weights[TreeMem[node].seq[mismatch_pos]]+=TreeMem[node].weight*weights_column[degree-1] ;
							TreeMem[last_node].child_weights[vec[i+j+seq_offset+k]] += alpha*weights_column[degree-1] ;
						}
						else
						{
							if (TreeMem[node].seq[mismatch_pos]<4) // i.e. !=TRIE_TERMINAL_CHARACTER
								TreeMem[last_node].child_weights[TreeMem[node].seq[mismatch_pos]]=TreeMem[node].weight ;
							TreeMem[last_node].child_weights[vec[i+j+seq_offset+k]] = alpha ;
						}

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
						INT tmp = get_node() ;
						TreeMem[last_node].children[vec[i+j+seq_offset+mismatch_pos]] = -tmp ;
						last_node=tmp ;

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
				if (weights_in_tree)
					TreeMem[tree].weight += alpha*weights_column[j];
				else
					TreeMem[tree].weight += alpha ;
			}
		}
		else if (j==degree-1)
		{
			// special treatment of the last node
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
			TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats) ;
			if (weights_in_tree)
				TreeMem[tree].child_weights[vec[i+j+seq_offset]] += alpha*weights_column[j] ;
			else
				TreeMem[tree].child_weights[vec[i+j+seq_offset]] += alpha;

			break;
		}
		else
		{
			bool use_seq = use_compact_terminal_nodes && (j>degree-16) ;
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats) ;

			INT tmp = get_node((j==degree-2) && (!use_seq));
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
				if (weights_in_tree)
					TreeMem[tree].weight = alpha*weights_column[j] ;
				else
					TreeMem[tree].weight = alpha ;
#ifdef TRIE_CHECK_EVERYTHING
				if (j==degree-2)
					TreeMem[tree].has_floats = true ;
#endif
			}
		}
	}
}

	template <class Trie>
DREAL CTrie<Trie>::compute_by_tree_helper(INT* vec, INT len, INT seq_pos, 
		INT tree_pos,
		INT weight_pos, DREAL* weights, 
		bool degree_times_position_weights)
{
	INT tree = trees[tree_pos] ;

	if ((position_weights!=NULL) && (position_weights[weight_pos]==0))
		return 0.0;

	DREAL *weights_column=NULL ;
	if (degree_times_position_weights)
		weights_column=&weights[weight_pos*degree] ;
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
				if (weights_in_tree)
					sum += TreeMem[tree].weight ;
				else
					sum += TreeMem[tree].weight * weights_column[j] ;
			} ;
		}
		else
		{
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
			if (j==degree-1)
			{
				TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats) ;
				if (weights_in_tree)
					sum += TreeMem[tree].child_weights[vec[seq_pos+j]] ;
				else
					sum += TreeMem[tree].child_weights[vec[seq_pos+j]] * weights_column[j] ;
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

	template <class Trie>
void CTrie<Trie>::compute_by_tree_helper(INT* vec, INT len,
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
							if (weights_in_tree)
								LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
							else
								LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k] ;
						}
						break ;
					}
					else
					{
						tree=TreeMem[tree].children[vec[seq_pos+j]];
						TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
						if (weights_in_tree)
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
						else
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j] ;
					}
				} 
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					if (j==degree-1)
					{
						if (weights_in_tree)
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]] ;
						else
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]]*weights[j] ;
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
							if (weights_in_tree)
								LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
							else
								LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k+weight_pos*degree] ;
						}
						break ;
					}
					else
					{
						tree=TreeMem[tree].children[vec[seq_pos+j]];
						TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
						if (weights_in_tree)
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
						else
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+weight_pos*degree] ;
					}
				} 
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					if (j==degree-1)
					{
						if (weights_in_tree)
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]] ;
						else
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]]*weights[j+weight_pos*degree] ;
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
						if (weights_in_tree)
							LevelContrib[(j+k)/mkl_stepsize] += factor*TreeMem[tree].weight ;
						else
							LevelContrib[(j+k)/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k] ;
					}
					break ;
				}
				else
				{
					tree=TreeMem[tree].children[vec[seq_pos+j]];
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					if (weights_in_tree)
						LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].weight ;
					else
						LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j] ;
				}
			}
			else
			{
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
				if (j==degree-1)
				{
					if (weights_in_tree)
						LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]] ;
					else
						LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].child_weights[vec[seq_pos+j]]*weights[j] ;
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
						if (weights_in_tree)
							LevelContrib[(j+k+degree*weight_pos)/mkl_stepsize] += factor*TreeMem[tree].weight ;
						else
							LevelContrib[(j+k+degree*weight_pos)/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+k+weight_pos*degree] ;
					}
					break ;
				}
				else
				{
					tree=TreeMem[tree].children[vec[seq_pos+j]];
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
					if (weights_in_tree)
						LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].weight ;
					else
						LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].weight * weights[j+weight_pos*degree] ;
				} 
			}
			else
			{
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq) ;
				if (j==degree-1)
				{
					if (weights_in_tree)
						LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].child_weights[vec[seq_pos+j]] ;
					else
						LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].child_weights[vec[seq_pos+j]] * weights[j+weight_pos*degree] ;
				}
				break ;
			}
		} 
	}
}

	template <class Trie>
void CTrie<Trie>::fill_backtracking_table_recursion(Trie* tree, INT depth, ULONG seq, DREAL value, CDynamicArray<ConsensusEntry>* table, DREAL* weights)
{
	DREAL w=1.0;

	if (weights_in_tree || depth==0)
		value+=tree->weight;
	else
	{
		w=weights[degree-1];
		value+=weights[depth-1]*tree->weight;
	}

	if (degree-1==depth)
	{
		for (INT sym=0; sym<4; sym++)
		{
			DREAL v=w*tree->child_weights[sym];
			if (v!=0.0)
			{
				ConsensusEntry entry;
				entry.bt=-1;
				entry.score=value+v;
				entry.string=seq | ((ULONG) sym) << (2*(degree-depth-1));

				table->append_element(entry);
			}
		}
	}
	else
	{
		for (INT sym=0; sym<4; sym++)
		{
			ULONG str=seq | ((ULONG) sym) << (2*(degree-depth-1));
			if (tree->children[sym] != NO_CHILD)
				fill_backtracking_table_recursion(&TreeMem[tree->children[sym]], depth+1, str, value, table, weights);
		}
	}
}

	template <class Trie>
DREAL CTrie<Trie>::get_cumulative_score(INT pos, ULONG seq, INT deg, DREAL* weights)
{
	DREAL result=0.0;

	//SG_PRINT("pos:%i length:%i deg:%i seq:0x%0llx...\n", pos, length, deg, seq);

	for (INT i=pos; i<pos+deg && i<length; i++)
	{
		//SG_PRINT("loop %d\n", i);
		Trie* tree = &TreeMem[trees[i]];

		for (INT d=0; d<deg-i+pos; d++)
		{
			//SG_PRINT("loop degree %d shit: %d\n", d, (2*(deg-1-d-i+pos)));
			ASSERT(d-1<degree);
			INT sym = (INT) (seq >> (2*(deg-1-d-i+pos)) & 3);

			DREAL w=1.0;
			if (!weights_in_tree)
				w=weights[d];

			ASSERT(tree->children[sym] != NO_CHILD);
			tree=&TreeMem[tree->children[sym]];
			result+=w*tree->weight;
		}
	}
	//SG_PRINT("cum: %f\n", result);
	return result;
}

	template <class Trie>
void CTrie<Trie>::fill_backtracking_table(INT pos, CDynamicArray<ConsensusEntry>* prev, CDynamicArray<ConsensusEntry>* cur, bool cumulative, DREAL* weights)
{
	ASSERT(pos>=0 && pos<length);
	ASSERT(!use_compact_terminal_nodes);

	Trie* t = &TreeMem[trees[pos]];

	fill_backtracking_table_recursion(t, 0, (ULONG) 0, 0.0, cur, weights);


	if (cumulative)
	{
		INT num_cur=cur->get_num_elements();
		for (INT i=0; i<num_cur; i++)
		{
			ConsensusEntry entry=cur->get_element(i);
			entry.score+=get_cumulative_score(pos+1, entry.string, degree-1, weights);
			cur->set_element(entry,i);
			//SG_PRINT("cum: str:0%0llx sc:%f bt:%d\n",entry.string,entry.score,entry.bt);
		}
	}

	//if previous tree exists find maximum scoring path
	//for each element in cur and update bt table
	if (prev)
	{
		INT num_cur=cur->get_num_elements();
		INT num_prev=prev->get_num_elements();

		for (INT i=0; i<num_cur; i++)
		{
			//ULONG str_cur_old= cur->get_element(i).string;
			ULONG str_cur= cur->get_element(i).string >> 2;
			//SG_PRINT("...cur:0x%0llx cur_noprfx:0x%0llx...\n", str_cur_old, str_cur);

			INT bt=-1;
			DREAL max_score=0.0;

			for (INT j=0; j<num_prev; j++)
			{
				//ULONG str_prev_old= prev->get_element(j).string;
				ULONG mask=((((ULONG)0)-1) ^ (((ULONG) 3) << (2*(degree-1))));
				ULONG str_prev=  mask & prev->get_element(j).string;
				//SG_PRINT("...prev:0x%0llx prev_nosfx:0x%0llx mask:%0llx...\n", str_prev_old, str_prev,mask);

				if (str_cur == str_prev)
				{
					DREAL sc=prev->get_element(j).score+cur->get_element(i).score;
					if (bt==-1 || sc>max_score)
					{
						bt=j;
						max_score=sc;

						//SG_PRINT("new max[%i,%i] = %f\n", j,i, max_score);
					}
				}
			}

			ASSERT(bt!=-1);
			ConsensusEntry entry;
			entry.bt=bt;
			entry.score=max_score;
			entry.string=cur->get_element(i).string;
			cur->set_element(entry, i);
			//SG_PRINT("entry[%d]: str:0%0llx sc:%f bt:%d\n",i, entry.string,entry.score,entry.bt);
		}
	}
}
#endif // _TRIE_H___
