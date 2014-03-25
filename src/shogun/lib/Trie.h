/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2009 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _TRIE_H___
#define _TRIE_H___

#include <shogun/lib/config.h>
#include <string.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/DynArray.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS

// sentinel is 0xFFFFFFFC or float -2
#define NO_CHILD ((int32_t)-1073741824)

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
	uint64_t string;
	/** score */
	float32_t score;
	/** bt */
	int32_t bt;
};

/** POIM trie */
struct POIMTrie
{
	/** weight */
	float64_t weight;
#ifdef TRIE_CHECK_EVERYTHING
	/** has sequence */
	bool has_seq;
	/** has floats */
	bool has_floats;
#endif
	union
	{
		/** child weights */
		float32_t child_weights[4];
		/** children */
		int32_t children[4];
		/** sequence */
		uint8_t seq[16] ;
	};

	/** super_string_score */
	float64_t S;
	/** left_partial_overlap_score */
	float64_t L;
	/** right_partial_overlap_score */
	float64_t R;
};

/** DNA trie */
struct DNATrie
{
	/** weight */
	float64_t weight;
#ifdef TRIE_CHECK_EVERYTHING
	/** has sequence */
	bool has_seq;
	/** has floats */
	bool has_floats;
#endif
	union
	{
		/** child weights */
		float32_t child_weights[4];
		/** children */
		int32_t children[4];
		/** sequence */
		uint8_t seq[16] ;
	};
};

/** tree parse info */
struct TreeParseInfo {
	/** number of symbols */
	int32_t num_sym;
	/** number of features */
	int32_t num_feat;
	/** p */
	int32_t p;
	/** k */
	int32_t k;
	/** nofsKmers */
	int32_t* nofsKmers;
	/** margFactors */
	float64_t* margFactors;
	/** x */
	int32_t* x;
	/** substrs */
	int32_t* substrs;
	/** y0 */
	int32_t y0;
	/** C k */
	float64_t* C_k;
	/** L k */
	float64_t* L_k;
	/** R k */
	float64_t* R_k;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

template <class Trie> class CTrie;

#define IGNORE_IN_CLASSLIST
/** @brief Template class Trie implements a suffix trie, i.e. a tree in which all
 * suffixes up to a certain length are stored.
 *
 * It is excessively used in the CWeightedDegreeStringKernel and
 * CWeightedDegreePositionStringKernel to construct the whole features space
 * \f$\Phi(x)\f$ and enormously helps here to speed up SVM training and
 * evaluation.
 *
 * Note that depending on the underlying structure used, a single symbol in the
 * tree requires 20 bytes (DNATrie). It is also used to do the efficient
 * recursion in computing positional oligomer importance matrices (POIMs) where
 * the structure requires * 20+3*8 (POIMTrie) bytes.
 *
 * Finally note that this tree may use compact internal nodes (for strings that
 * appear without modifications, thus not requiring further branches), which
 * may save a lot of memory on higher degree tries.
 *
 */
IGNORE_IN_CLASSLIST template <class Trie> class CTrie : public CSGObject
{
	public:
		/** default constructor  */
		CTrie();

		/** constructor
		 *
		 * @param d degree
		 * @param p_use_compact_terminal_nodes if compact terminal nodes shall
		 *                                     be used
		 */
		CTrie(int32_t d, bool p_use_compact_terminal_nodes=true);

		/** copy constructor */
		CTrie(const CTrie & to_copy);
		virtual ~CTrie();

		/** overload operator = */
		const CTrie & operator=(const CTrie & to_copy);

		/** compare traverse
		 *
		 * @param node node
		 * @param other other trie
		 * @param other_node other node
		 * @return if comparison was successful
		 */
		bool compare_traverse(
			int32_t node, const CTrie & other, int32_t other_node);

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
		bool find_node(int32_t node, int32_t * trace, int32_t &trace_len) const;

		/** find deepest node
		 *
		 * @param start_node start node
		 * @param deepest_node deepest node will be stored in here
		 * @return depth of deepest node
		 */
		int32_t find_deepest_node(
			int32_t start_node, int32_t &deepest_node) const;

		/** display node
		 *
		 * @param node node to display
		 */
		void display_node(int32_t node) const;

		/** destroy */
		void destroy();

		/** set degree
		 *
		 * @param d new degree
		 */
		void set_degree(int32_t d);

		/** create
		 *
		 * @param len length of new trie
		 * @param p_use_compact_terminal_nodes if compact terminal nodes shall
		 *                                     be used
		 */
		void create(int32_t len, bool p_use_compact_terminal_nodes=true);

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
		void add_to_trie(
			int32_t i, int32_t seq_offset, int32_t* vec, float32_t alpha,
			float64_t *weights, bool degree_times_position_weights);

		/** compute absolute weights tree
		 *
		 * @param tree tree to compute for
		 * @param depth depth
		 * @return computed absolute weights tree
		 */
		float64_t compute_abs_weights_tree(int32_t tree, int32_t depth);

		/** compute absolute weights
		 *
		 * @param len length
		 * @return computed absolute weights
		 */
		float64_t* compute_abs_weights(int32_t &len);

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
		float64_t compute_by_tree_helper(
			int32_t* vec, int32_t len, int32_t seq_pos, int32_t tree_pos,
			int32_t weight_pos, float64_t * weights,
			bool degree_times_position_weights) ;

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
		void compute_by_tree_helper(
			int32_t* vec, int32_t len, int32_t seq_pos, int32_t tree_pos,
			int32_t weight_pos, float64_t* LevelContrib, float64_t factor,
			int32_t mkl_stepsize, float64_t * weights,
			bool degree_times_position_weights);

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
		void compute_scoring_helper(
			int32_t tree, int32_t i, int32_t j, float64_t weight, int32_t d,
			int32_t max_degree, int32_t num_feat, int32_t num_sym,
			int32_t sym_offset, int32_t offs, float64_t* result);

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
		void add_example_to_tree_mismatch_recursion(
			int32_t tree,  int32_t i, float64_t alpha, int32_t *vec,
			int32_t len_rem, int32_t degree_rec, int32_t mismatch_rec,
			int32_t max_mismatch, float64_t * weights);

		/** traverse
		 *
		 * @param tree tree
		 * @param p p
		 * @param info tree parse info
		 * @param depth depth
		 * @param x x
		 * @param k k
		 */
		void traverse(
			int32_t tree, const int32_t p, struct TreeParseInfo info,
			const int32_t depth, int32_t* const x, const int32_t k);

		/** count
		 *
		 * @param w w
		 * @param depth depth
		 * @param info tree parse info
		 * @param p p
		 * @param x x
		 * @param k
		 */
		void count(
			const float64_t w, const int32_t depth,
			const struct TreeParseInfo info, const int32_t p, int32_t* x,
			const int32_t k);

		/** compact nodes
		 *
		 * @param start_node start node
		 * @param depth depth
		 * @param weights weights
		 */
		int32_t compact_nodes(int32_t start_node, int32_t depth, float64_t * weights);

		/** get cumulative score
		 *
		 * @param pos position
		 * @param seq sequence
		 * @param deg degree
		 * @param weights weights
		 * @return cumulative score
		 */
		float64_t get_cumulative_score(
			int32_t pos, uint64_t seq, int32_t deg, float64_t* weights);

		/** fill backtracking table recursion
		 *
		 * @param tree tree
		 * @param depth depth
		 * @param seq sequence
		 * @param value value
		 * @param table table of concensus entries
		 * @param weights weights
		 */
		void fill_backtracking_table_recursion(
			Trie* tree, int32_t depth, uint64_t seq, float64_t value,
			DynArray<ConsensusEntry>* table, float64_t* weights);

		/** fill backtracking table
		 *
		 * @param pos position
		 * @param prev previous concencus entry
		 * @param cur current concensus entry
		 * @param cumulative if is cumulative
		 * @param weights weights
		 */
		void fill_backtracking_table(
			int32_t pos, DynArray<ConsensusEntry>* prev,
			DynArray<ConsensusEntry>* cur, bool cumulative,
			float64_t* weights);

		/** POIMs extract W
		 *
		 * @param W W
		 * @param K K
		 */
		void POIMs_extract_W(float64_t* const* const W, const int32_t K);

		/** POIMs precalc SLR
		 *
		 * @param distrib distribution
		 */
		void POIMs_precalc_SLR(const float64_t* const distrib);

		/** POIMs get SLR
		 *
		 * @param parentIdx parent index
		 * @param sym symbol
		 * @param depth depth
		 * @param S will point to S
		 * @param L will point to L
		 * @param R will point to R
		 */
		void POIMs_get_SLR(
			const int32_t parentIdx, const int32_t sym, const int32_t depth,
			float64_t* S, float64_t* L, float64_t* R);

		/** POIMs add SLR
		 *
		 * @param poims POIMs
		 * @param K K
		 * @param debug debug level
		 */
		void POIMs_add_SLR(
			float64_t* const* const poims, const int32_t K,
			const int32_t debug);

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
		inline void set_use_compact_terminal_nodes(
			bool p_use_compact_terminal_nodes)
		{
			use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
		}

		/** get number of used nodes
		 *
		 * @return number of used nodes
		 */
		inline int32_t get_num_used_nodes()
		{
			return TreeMemPtr;
		}

		/** set position weights
		 *
		 * @param p_position_weights new position weights
		 */
		inline void set_position_weights(float64_t* p_position_weights)
		{
			position_weights=p_position_weights;
		}

		/** get node
		 *
		 * @return node
		 */
		inline int32_t get_node(bool last_node=false)
		{
			int32_t ret = TreeMemPtr++;
			check_treemem() ;

			if (last_node)
			{
				for (int32_t q=0; q<4; q++)
					TreeMem[ret].child_weights[q]=0.0;
			}
			else
			{
				for (int32_t q=0; q<4; q++)
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
			SG_DEBUG("Extending TreeMem from %i to %i elements\n",
					TreeMemPtrMax, (int32_t) ((float64_t)TreeMemPtrMax*1.2));
			int32_t old_sz=TreeMemPtrMax;
			TreeMemPtrMax = (int32_t) ((float64_t)TreeMemPtrMax*1.2);
			TreeMem = SG_REALLOC(Trie, TreeMem, old_sz, TreeMemPtrMax);
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
		void POIMs_extract_W_helper(
			const int32_t nodeIdx, const int32_t depth, const int32_t offset,
			const int32_t y0, float64_t* const* const W, const int32_t K);

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
		void POIMs_calc_SLR_helper1(
			const float64_t* const distrib, const int32_t i,
			const int32_t nodeIdx, int32_t left_tries_idx[4],
			const int32_t depth, int32_t const lastSym, float64_t* S,
			float64_t* L, float64_t* R);


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
		void POIMs_calc_SLR_helper2(
			const float64_t* const distrib, const int32_t i,
			const int32_t nodeIdx, int32_t left_tries_idx[4],
			const int32_t depth, float64_t* S, float64_t* L, float64_t* R);

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
		void POIMs_add_SLR_helper1(
			const int32_t nodeIdx, const int32_t depth,const int32_t i,
			const int32_t y0, float64_t* const* const poims, const int32_t K,
			const int32_t debug);

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
		void POIMs_add_SLR_helper2(
			float64_t* const* const poims, const int32_t K, const int32_t k,
			const int32_t i, const int32_t y, const float64_t valW,
			const float64_t valS, const float64_t valL, const float64_t valR,
			const int32_t debug);

		/** @return object name */
		virtual const char* get_name() const { return "Trie"; }

	public:
		/** number of symbols */
		int32_t NUM_SYMS;

	protected:
		/** length */
		int32_t length;
		/** trees */
		int32_t * trees;

		/** degree */
		int32_t degree;
		/** position weights */
		float64_t*  position_weights;

		/** tree memory */
		Trie* TreeMem;
		/** tree memory pointer */
		int32_t TreeMemPtr;
		/** tree memory pointer maximum */
		int32_t TreeMemPtrMax;
		/** if compact terminal nodes are used */
		bool use_compact_terminal_nodes;

		/** if weights are in tree */
		bool weights_in_tree;

		/** nofsKmers */
		int32_t* nofsKmers;
};
	template <class Trie>
	CTrie<Trie>::CTrie()
	: CSGObject(), degree(0), position_weights(NULL),
		use_compact_terminal_nodes(false),
		weights_in_tree(true)
	{

		TreeMemPtrMax=0;
		TreeMemPtr=0;
		TreeMem=NULL;

		length=0;
		trees=NULL;

		NUM_SYMS=4;
	}

	template <class Trie>
	CTrie<Trie>::CTrie(int32_t d, bool p_use_compact_terminal_nodes)
	: CSGObject(), degree(d), position_weights(NULL),
		use_compact_terminal_nodes(p_use_compact_terminal_nodes),
		weights_in_tree(true)
	{
		TreeMemPtrMax=1024*1024/sizeof(Trie);
		TreeMemPtr=0;
		TreeMem=SG_MALLOC(Trie, TreeMemPtrMax);

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
			/*SG_MALLOC(float64_t, to_copy.length);
			  for (int32_t i=0; i<to_copy.length; i++)
			  position_weights[i]=to_copy.position_weights[i]; */
		}
		else
			position_weights=NULL;

		TreeMemPtrMax=to_copy.TreeMemPtrMax;
		TreeMemPtr=to_copy.TreeMemPtr;
		TreeMem=SG_MALLOC(Trie, TreeMemPtrMax);
		memcpy(TreeMem, to_copy.TreeMem, TreeMemPtrMax*sizeof(Trie));

		length=to_copy.length;
		trees=SG_MALLOC(int32_t, length);
		for (int32_t i=0; i<length; i++)
			trees[i]=to_copy.trees[i];

		NUM_SYMS=4;
	}

	template <class Trie>
const CTrie<Trie> &CTrie<Trie>::operator=(const CTrie<Trie> & to_copy)
{
	degree=to_copy.degree ;
	use_compact_terminal_nodes=to_copy.use_compact_terminal_nodes ;

	SG_FREE(position_weights);
	position_weights=NULL ;
	if (to_copy.position_weights!=NULL)
	{
		position_weights=to_copy.position_weights ;
		/*position_weights = SG_MALLOC(float64_t, to_copy.length);
		  for (int32_t i=0; i<to_copy.length; i++)
		  position_weights[i]=to_copy.position_weights[i] ;*/
	}
	else
		position_weights=NULL ;

	TreeMemPtrMax=to_copy.TreeMemPtrMax ;
	TreeMemPtr=to_copy.TreeMemPtr ;
	SG_FREE(TreeMem) ;
	TreeMem = SG_MALLOC(Trie, TreeMemPtrMax);
	memcpy(TreeMem, to_copy.TreeMem, TreeMemPtrMax*sizeof(Trie)) ;

	length = to_copy.length ;
	if (trees)
		SG_FREE(trees);
	trees=SG_MALLOC(int32_t, length);
	for (int32_t i=0; i<length; i++)
		trees[i]=to_copy.trees[i] ;

	return *this ;
}

template <class Trie>
int32_t CTrie<Trie>::find_deepest_node(
	int32_t start_node, int32_t& deepest_node) const
{
#ifdef TRIE_CHECK_EVERYTHING
	int32_t ret=0 ;
	SG_DEBUG("start_node=%i\n", start_node)

	if (start_node==NO_CHILD)
	{
		for (int32_t i=0; i<length; i++)
		{
			int32_t my_deepest_node ;
			int32_t depth=find_deepest_node(i, my_deepest_node) ;
			SG_DEBUG("start_node %i depth=%i\n", i, depth)
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
		for (int32_t q=0; q<16; q++)
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

	for (int32_t q=0; q<4; q++)
	{
		int32_t my_deepest_node ;
		if (TreeMem[start_node].children[q]==NO_CHILD)
			continue ;
		int32_t depth=find_deepest_node(abs(TreeMem[start_node].children[q]), my_deepest_node) ;
		if (depth>ret)
		{
			deepest_node=my_deepest_node ;
			ret=depth ;
		}
	}
	return ret ;
#else
	SG_ERROR("not implemented\n")
	return 0 ;
#endif
}

	template <class Trie>
int32_t CTrie<Trie>::compact_nodes(
	int32_t start_node, int32_t depth, float64_t * weights)
{
	SG_ERROR("code buggy\n")

	int32_t ret=0 ;

	if (start_node==NO_CHILD)
	{
		for (int32_t i=0; i<length; i++)
			compact_nodes(i,1, weights) ;
		return 0 ;
	}
	if (start_node<0)
		return -1 ;

	if (depth==degree-1)
	{
		TRIE_ASSERT_EVERYTHING(TreeMem[start_node].has_floats)
		int32_t num_used=0 ;
		for (int32_t q=0; q<4; q++)
			if (TreeMem[start_node].child_weights[q]!=0.0)
				num_used++ ;
		if (num_used>1)
			return -1 ;
		return 1 ;
	}
	TRIE_ASSERT_EVERYTHING(!TreeMem[start_node].has_floats)

	int32_t num_used = 0 ;
	int32_t q_used=-1 ;

	for (int32_t q=0; q<4; q++)
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
		for (int32_t q=0; q<4; q++)
		{
			if (TreeMem[start_node].children[q]==NO_CHILD)
				continue ;
			int32_t num=compact_nodes(abs(TreeMem[start_node].children[q]), depth+1, weights) ;
			if (num<=2)
				continue ;
			int32_t node=get_node() ;

			int32_t last_node=TreeMem[start_node].children[q] ;
			if (weights_in_tree)
			{
				ASSERT(weights[depth]!=0.0)
				TreeMem[node].weight=TreeMem[last_node].weight/weights[depth] ;
			}
			else
				TreeMem[node].weight=TreeMem[last_node].weight ;

#ifdef TRIE_CHECK_EVERYTHING
			TreeMem[node].has_seq=true ;
#endif
			memset(TreeMem[node].seq, TRIE_TERMINAL_CHARACTER, 16) ;
			for (int32_t n=0; n<num; n++)
			{
				ASSERT(depth+n+1<=degree-1)
				ASSERT(last_node!=NO_CHILD)
				if (depth+n+1==degree-1)
				{
					TRIE_ASSERT_EVERYTHING(TreeMem[last_node].has_floats)
					int32_t  k ;
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
					TRIE_ASSERT_EVERYTHING(!TreeMem[last_node].has_floats)
					int32_t k ;
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
bool CTrie<Trie>::compare_traverse(
	int32_t node, const CTrie<Trie> & other, int32_t other_node)
{
	SG_DEBUG("checking nodes %i and %i\n", node, other_node)
	if (fabs(TreeMem[node].weight-other.TreeMem[other_node].weight)>=1e-5)
	{
		SG_DEBUG("CTrie::compare: TreeMem[%i].weight=%f!=other.TreeMem[%i].weight=%f\n", node, TreeMem[node].weight, other_node,other.TreeMem[other_node].weight)
		SG_DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
		display_node(node) ;
		SG_DEBUG("============================================================\n")
		other.display_node(other_node) ;
		SG_DEBUG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
		return false ;
	}

#ifdef TRIE_CHECK_EVERYTHING
	if (TreeMem[node].has_seq!=other.TreeMem[other_node].has_seq)
	{
		SG_DEBUG("CTrie::compare: TreeMem[%i].has_seq=%i!=other.TreeMem[%i].has_seq=%i\n", node, TreeMem[node].has_seq, other_node,other.TreeMem[other_node].has_seq)
		SG_DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
		display_node(node) ;
		SG_DEBUG("============================================================\n")
		other.display_node(other_node) ;
		SG_DEBUG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
		return false ;
	}
	if (TreeMem[node].has_floats!=other.TreeMem[other_node].has_floats)
	{
		SG_DEBUG("CTrie::compare: TreeMem[%i].has_floats=%i!=other.TreeMem[%i].has_floats=%i\n", node, TreeMem[node].has_floats, other_node, other.TreeMem[other_node].has_floats)
		return false ;
	}
	if (other.TreeMem[other_node].has_floats)
	{
		for (int32_t q=0; q<4; q++)
			if (fabs(TreeMem[node].child_weights[q]-other.TreeMem[other_node].child_weights[q])>1e-5)
			{
				SG_DEBUG("CTrie::compare: TreeMem[%i].child_weights[%i]=%e!=other.TreeMem[%i].child_weights[%i]=%e\n", node, q,TreeMem[node].child_weights[q], other_node,q,other.TreeMem[other_node].child_weights[q])
				SG_DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
				display_node(node) ;
				SG_DEBUG("============================================================\n")
				other.display_node(other_node) ;
				SG_DEBUG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
				return false ;
			}
	}
	if (other.TreeMem[other_node].has_seq)
	{
		for (int32_t q=0; q<16; q++)
			if ((TreeMem[node].seq[q]!=other.TreeMem[other_node].seq[q]) && ((TreeMem[node].seq[q]<4)||(other.TreeMem[other_node].seq[q]<4)))
			{
				SG_DEBUG("CTrie::compare: TreeMem[%i].seq[%i]=%i!=other.TreeMem[%i].seq[%i]=%i\n", node,q,TreeMem[node].seq[q], other_node,q,other.TreeMem[other_node].seq[q])
				SG_DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
				display_node(node) ;
				SG_DEBUG("============================================================\n")
				other.display_node(other_node) ;
				SG_DEBUG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
				return false ;
			}
	}
	if (!other.TreeMem[other_node].has_seq && !other.TreeMem[other_node].has_floats)
	{
		for (int32_t q=0; q<4; q++)
		{
			if ((TreeMem[node].children[q]==NO_CHILD) && (other.TreeMem[other_node].children[q]==NO_CHILD))
				continue ;
			if ((TreeMem[node].children[q]==NO_CHILD)!=(other.TreeMem[other_node].children[q]==NO_CHILD))
			{
				SG_DEBUG("CTrie::compare: TreeMem[%i].children[%i]=%i!=other.TreeMem[%i].children[%i]=%i\n", node,q,TreeMem[node].children[q], other_node,q,other.TreeMem[other_node].children[q])
				SG_DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
				display_node(node) ;
				SG_DEBUG("============================================================\n")
				other.display_node(other_node) ;
				SG_DEBUG("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
				return false ;
			}
			if (!compare_traverse(abs(TreeMem[node].children[q]), other, abs(other.TreeMem[other_node].children[q])))
				return false ;
		}
	}
#else
	SG_ERROR("not implemented\n")
#endif

	return true ;
}

	template <class Trie>
bool CTrie<Trie>::compare(const CTrie<Trie> & other)
{
	bool ret=true ;
	for (int32_t i=0; i<length; i++)
		if (!compare_traverse(trees[i], other, other.trees[i]))
			return false ;
		else
			SG_DEBUG("two tries at %i identical\n", i)

	return ret ;
}

template <class Trie>
bool CTrie<Trie>::find_node(
	int32_t node, int32_t * trace, int32_t& trace_len) const
{
#ifdef TRIE_CHECK_EVERYTHING
	ASSERT(trace_len-1>=0)
	ASSERT((trace[trace_len-1]>=0) && (trace[trace_len-1]<TreeMemPtrMax))
		if (TreeMem[trace[trace_len-1]].has_seq)
			return false ;
	if (TreeMem[trace[trace_len-1]].has_floats)
		return false ;

	for (int32_t q=0; q<4; q++)
	{
		if (TreeMem[trace[trace_len-1]].children[q]==NO_CHILD)
			continue ;
		int32_t tl=trace_len+1 ;
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
	SG_ERROR("not implemented\n")
	return false ;
#endif
}

template <class Trie>
void CTrie<Trie>::display_node(int32_t node) const
{
#ifdef TRIE_CHECK_EVERYTHING
	int32_t * trace=SG_MALLOC(int32_t, 2*degree);
	int32_t trace_len=-1 ;
	bool found = false ;
	int32_t tree=-1 ;
	for (tree=0; tree<length; tree++)
	{
		trace[0]=trees[tree] ;
		trace_len=1 ;
		found=find_node(node, trace, trace_len) ;
		if (found)
			break ;
	}
	ASSERT(found)
	SG_PRINT("position %i  trace: ", tree)

	for (int32_t i=0; i<trace_len-1; i++)
	{
		int32_t branch=-1 ;
		for (int32_t q=0; q<4; q++)
			if (abs(TreeMem[trace[i]].children[q])==trace[i+1])
			{
				branch=q;
				break ;
			}
		ASSERT(branch!=-1)
		char acgt[5]="ACGT" ;
		SG_PRINT("%c", acgt[branch])
	}
	SG_PRINT("\nnode=%i\nweight=%f\nhas_seq=%i\nhas_floats=%i\n", node, TreeMem[node].weight, TreeMem[node].has_seq, TreeMem[node].has_floats)
	if (TreeMem[node].has_floats)
	{
		for (int32_t q=0; q<4; q++)
			SG_PRINT("child_weighs[%i] = %f\n", q, TreeMem[node].child_weights[q])
	}
	if (TreeMem[node].has_seq)
	{
		for (int32_t q=0; q<16; q++)
			SG_PRINT("seq[%i] = %i\n", q, TreeMem[node].seq[q])
	}
	if (!TreeMem[node].has_seq && !TreeMem[node].has_floats)
	{
		for (int32_t q=0; q<4; q++)
		{
			if (TreeMem[node].children[q]!=NO_CHILD)
			{
				SG_PRINT("children[%i] = %i -> \n", q, TreeMem[node].children[q])
				display_node(abs(TreeMem[node].children[q])) ;
			}
			else
				SG_PRINT("children[%i] = NO_CHILD -| \n", q, TreeMem[node].children[q])
		}

	}

	SG_FREE(trace);
#else
	SG_ERROR("not implemented\n")
#endif
}


template <class Trie> CTrie<Trie>::~CTrie()
{
	destroy() ;

	SG_FREE(TreeMem) ;
}

template <class Trie> void CTrie<Trie>::destroy()
{
	if (trees!=NULL)
	{
		delete_trees();
		for (int32_t i=0; i<length; i++)
			trees[i] = NO_CHILD;
		SG_FREE(trees);

		TreeMemPtr=0;
		length=0;
		trees=NULL;
	}
}

template <class Trie> void CTrie<Trie>::set_degree(int32_t d)
{
	delete_trees(get_use_compact_terminal_nodes());
	degree=d;
}

template <class Trie> void CTrie<Trie>::create(
	int32_t len, bool p_use_compact_terminal_nodes)
{
	destroy();

	trees=SG_MALLOC(int32_t, len);
	TreeMemPtr=0 ;
	for (int32_t i=0; i<len; i++)
		trees[i]=get_node(degree==1);
	length = len ;

	use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
}


template <class Trie> void CTrie<Trie>::delete_trees(
	bool p_use_compact_terminal_nodes)
{
	if (trees==NULL)
		return;

	TreeMemPtr=0 ;
	for (int32_t i=0; i<length; i++)
		trees[i]=get_node(degree==1);

	use_compact_terminal_nodes=p_use_compact_terminal_nodes ;
}

	template <class Trie>
float64_t CTrie<Trie>::compute_abs_weights_tree(int32_t tree, int32_t depth)
{
	float64_t ret=0 ;

	if (tree==NO_CHILD)
		return 0 ;
	TRIE_ASSERT(tree>=0)

	if (depth==degree-2)
	{
		ret+=(TreeMem[tree].weight) ;

		for (int32_t k=0; k<4; k++)
			ret+=(TreeMem[tree].child_weights[k]) ;

		return ret ;
	}

	ret+=(TreeMem[tree].weight) ;

	for (int32_t i=0; i<4; i++)
		if (TreeMem[tree].children[i]!=NO_CHILD)
			ret += compute_abs_weights_tree(TreeMem[tree].children[i], depth+1)  ;

	return ret ;
}


	template <class Trie>
float64_t *CTrie<Trie>::compute_abs_weights(int32_t &len)
{
	float64_t * sum=SG_MALLOC(float64_t, length*4);
	for (int32_t i=0; i<length*4; i++)
		sum[i]=0 ;
	len=length ;

	for (int32_t i=0; i<length; i++)
	{
		TRIE_ASSERT(trees[i]!=NO_CHILD)
		for (int32_t k=0; k<4; k++)
		{
			sum[i*4+k]=compute_abs_weights_tree(TreeMem[trees[i]].children[k], 0) ;
		}
	}

	return sum ;
}

	template <class Trie>
void CTrie<Trie>::add_example_to_tree_mismatch_recursion(
	int32_t tree,  int32_t i, float64_t alpha,
		int32_t *vec, int32_t len_rem,
		int32_t degree_rec, int32_t mismatch_rec,
		int32_t max_mismatch, float64_t * weights)
{
	if (tree==NO_CHILD)
		tree=trees[i] ;
	TRIE_ASSERT(tree!=NO_CHILD)

	if ((len_rem<=0) || (mismatch_rec>max_mismatch) || (degree_rec>degree))
		return ;
	const int32_t other[4][3] = {	{1,2,3},{0,2,3},{0,1,3},{0,1,2} } ;

	int32_t subtree = NO_CHILD ;

	if (degree_rec==degree-1)
	{
		TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats)
		if (weights_in_tree)
			TreeMem[tree].child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec];
		else
			if (weights[degree_rec]!=0.0)
				TreeMem[tree].child_weights[vec[0]] += alpha*weights[degree_rec+degree*mismatch_rec]/weights[degree_rec];
		if (mismatch_rec+1<=max_mismatch)
			for (int32_t o=0; o<3; o++)
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
		TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats)
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
			int32_t tmp = get_node(degree_rec==degree-2);
			ASSERT(tmp>=0)
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
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats)
			for (int32_t o=0; o<3; o++)
			{
				int32_t ot = other[vec[0]][o] ;
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
					int32_t tmp = get_node(degree_rec==degree-2);
					ASSERT(tmp>=0)
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
void CTrie<Trie>::compute_scoring_helper(
	int32_t tree, int32_t i, int32_t j, float64_t weight, int32_t d,
	int32_t max_degree, int32_t num_feat, int32_t num_sym, int32_t sym_offset,
	int32_t offs, float64_t* result)
{
	if (i+j<num_feat)
	{
		float64_t decay=1.0; //no decay by default
		//if (j>d)
		//	decay=pow(0.5,j); //marginalize out lower order matches

		if (j<degree-1)
		{
			for (int32_t k=0; k<num_sym; k++)
			{
				if (TreeMem[tree].children[k]!=NO_CHILD)
				{
					int32_t child=TreeMem[tree].children[k];
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
			for (int32_t k=0; k<num_sym; k++)
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
void CTrie<Trie>::traverse(
	int32_t tree, const int32_t p, struct TreeParseInfo info,
	const int32_t depth, int32_t* const x, const int32_t k)
{
	const int32_t num_sym = info.num_sym;
	const int32_t y0 = info.y0;
	const int32_t y1 = (k==0) ? 0 : y0 - ( (depth<k) ? 0 : info.nofsKmers[k-1] * x[depth-k] );
	//const int32_t temp = info.substrs[depth]*num_sym - ( (depth<=k) ? 0 : info.nofsKmers[k] * x[depth-k-1] );
	//if( !( info.y0 == temp ) ) {
	//  printf( "\n temp=%d y0=%d k=%d depth=%d \n", temp, info.y0, k, depth );
	//}
	//ASSERT( info.y0 == temp )
	int32_t sym;
	ASSERT( depth < degree )
	//ASSERT( 0 <= info.substrs[depth] && info.substrs[depth] < info.nofsKmers[k] )
	if (depth<degree-1)
	{
		for( sym=0; sym<num_sym; ++sym ) {
			const int32_t childNum = TreeMem[tree].children[ sym ];
			if( childNum != NO_CHILD ) {
				int32_t child = childNum ;
				x[depth] = sym;
				info.substrs[depth+1] = y0 + sym;
				info.y0 = (k==0) ? 0 : (y1+sym)*num_sym;
				//ASSERT( info.y0 == ( info.substrs[depth+1]*num_sym - ( (depth<k) ? 0 : info.nofsKmers[k] * x[depth-k] ) ) )
				count( TreeMem[child].weight, depth, info, p, x, k );
				traverse( child, p, info, depth+1, x, k );
				x[depth] = -1;
			}
		}
	}
	else if( depth == degree-1 )
	{
		for( sym=0; sym<num_sym; ++sym ) {
			const float64_t w = TreeMem[tree].child_weights[ sym ];
			if( w != 0.0 ) {
				x[depth] = sym;
				info.substrs[depth+1] = y0 + sym;
				info.y0 = (k==0) ? 0 : (y1+sym)*num_sym;
				//ASSERT( info.y0 == ( info.substrs[depth+1]*num_sym - ( (depth<k) ? 0 : info.nofsKmers[k] * x[depth-k] ) ) )
				count( w, depth, info, p, x, k );
				x[depth] = -1;
			}
		}
	}
	//info.substrs[depth+1] = -1;
	//info.y0 = temp;
}

	template <class Trie>
void CTrie<Trie>::count(
	const float64_t w, const int32_t depth, const struct TreeParseInfo info,
	const int32_t p, int32_t* x, const int32_t k)
{
	ASSERT( fabs(w) < 1e10 )
	ASSERT( x[depth] >= 0 )
	ASSERT( x[depth+1] < 0 )
	if ( depth < k ) {
		return;
	}
	//ASSERT( info.margFactors[ depth-k ] == pow( 0.25, depth-k ) )
	const int32_t nofKmers = info.nofsKmers[k];
	const float64_t margWeight =  w * info.margFactors[ depth-k ];
	const int32_t m_a = depth - k + 1;
	const int32_t m_b = info.num_feat - p;
	const int32_t m = ( m_a < m_b ) ? m_a : m_b;
	// all proper k-substrings
	const int32_t offset0 = nofKmers * p;
	register int32_t i;
	register int32_t offset;
	offset = offset0;
	for( i = 0; i < m; ++i ) {
		const int32_t y = info.substrs[i+k+1];
		info.C_k[ y + offset ] += margWeight;
		offset += nofKmers;
	}
	if( depth > k ) {
		// k-prefix
		const int32_t offsR = info.substrs[k+1] + offset0;
		info.R_k[offsR] += margWeight;
		// k-suffix
		if( p+depth-k < info.num_feat ) {
			const int32_t offsL = info.substrs[depth+1] + nofKmers * (p+depth-k);
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
void CTrie<Trie>::add_to_trie(
	int32_t i, int32_t seq_offset, int32_t * vec, float32_t alpha,
	float64_t *weights, bool degree_times_position_weights)
{
	int32_t tree = trees[i] ;
	//ASSERT(seq_offset==0)

	int32_t max_depth = 0 ;
	float64_t* weights_column ;
	if (degree_times_position_weights)
		weights_column = &weights[(i+seq_offset)*degree] ;
	else
		weights_column = weights ;

	if (weights_in_tree)
	{
		for (int32_t j=0; (j<degree) && (i+j<length); j++)
			if (CMath::abs(weights_column[j]*alpha)>0)
				max_depth = j+1 ;
	}
	else
		// don't use the weights
		max_depth=degree ;

	for (int32_t j=0; (j<max_depth) && (i+j+seq_offset<length); j++)
	{
		TRIE_ASSERT((vec[i+j+seq_offset]>=0) && (vec[i+j+seq_offset]<4))
		if ((j<degree-1) && (TreeMem[tree].children[vec[i+j+seq_offset]]!=NO_CHILD))
		{
			if (TreeMem[tree].children[vec[i+j+seq_offset]]<0)
			{
				// special treatment of the next nodes
				TRIE_ASSERT(j >= degree-16)
				// get the right element
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats)
				int32_t node = - TreeMem[tree].children[vec[i+j+seq_offset]] ;

				TRIE_ASSERT((node>=0) && (node<=TreeMemPtrMax))
				TRIE_ASSERT_EVERYTHING(TreeMem[node].has_seq)
				TRIE_ASSERT_EVERYTHING(!TreeMem[node].has_floats)

				// check whether the same string is stored
				int32_t mismatch_pos = -1 ;
				{
					int32_t k ;
					for (k=0; (j+k<max_depth) && (i+j+seq_offset+k<length); k++)
					{
						TRIE_ASSERT((vec[i+j+seq_offset+k]>=0) && (vec[i+j+seq_offset+k]<4))
						// ###
						if ((TreeMem[node].seq[k]>=4) && (TreeMem[node].seq[k]!=TRIE_TERMINAL_CHARACTER))
							fprintf(stderr, "+++i=%i j=%i seq[%i]=%i\n", i, j, k, TreeMem[node].seq[k]) ;
						TRIE_ASSERT((TreeMem[node].seq[k]<4) || (TreeMem[node].seq[k]==TRIE_TERMINAL_CHARACTER))
						TRIE_ASSERT(k<16)
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
					int32_t last_node=tree ;

					// create new nodes until mismatch
					int32_t k ;
					for (k=0; k<mismatch_pos; k++)
					{
						TRIE_ASSERT((vec[i+j+seq_offset+k]>=0) && (vec[i+j+seq_offset+k]<4))
						TRIE_ASSERT(vec[i+j+seq_offset+k]==TreeMem[node].seq[k])

						int32_t tmp=get_node();
						TreeMem[last_node].children[vec[i+j+seq_offset+k]]=tmp ;
						last_node=tmp ;
						if (weights_in_tree)
							TreeMem[last_node].weight = (TreeMem[node].weight+alpha)*weights_column[j+k] ;
						else
							TreeMem[last_node].weight = (TreeMem[node].weight+alpha) ;
						TRIE_ASSERT(j+k!=degree-1)
					}
					if ((TreeMem[node].seq[mismatch_pos]>=4) && (TreeMem[node].seq[mismatch_pos]!=TRIE_TERMINAL_CHARACTER))
						fprintf(stderr, "**i=%i j=%i seq[%i]=%i\n", i, j, k, TreeMem[node].seq[mismatch_pos]) ;
					ASSERT((TreeMem[node].seq[mismatch_pos]<4) || (TreeMem[node].seq[mismatch_pos]==TRIE_TERMINAL_CHARACTER))
					TRIE_ASSERT(vec[i+j+seq_offset+mismatch_pos]!=TreeMem[node].seq[mismatch_pos])

					if (j+k==degree-1)
					{
						// init child weights with zero if after dropping out
						// of the k<mismatch_pos loop we are one level below degree
						// (keep this even after get_node() change!)
						for (int32_t q=0; q<4; q++)
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
							for (int32_t q=0; q<16; q++)
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
						TRIE_ASSERT((vec[i+j+seq_offset+mismatch_pos]>=0) && (vec[i+j+seq_offset+mismatch_pos]<4))
						int32_t tmp = get_node() ;
						TreeMem[last_node].children[vec[i+j+seq_offset+mismatch_pos]] = -tmp ;
						last_node=tmp ;

						TreeMem[last_node].weight = alpha ;
#ifdef TRIE_CHECK_EVERYTHING
						TreeMem[last_node].has_seq = true ;
#endif
						memset(TreeMem[last_node].seq, TRIE_TERMINAL_CHARACTER, 16) ;
						for (int32_t q=0; (j+q+mismatch_pos<degree) && (i+j+seq_offset+q+mismatch_pos<length); q++)
							TreeMem[last_node].seq[q] = vec[i+j+seq_offset+mismatch_pos+q] ;
					}
				}
				break ;
			}
			else
			{
				tree=TreeMem[tree].children[vec[i+j+seq_offset]] ;
				TRIE_ASSERT((tree>=0) && (tree<TreeMemPtrMax))
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
				if (weights_in_tree)
					TreeMem[tree].weight += alpha*weights_column[j];
				else
					TreeMem[tree].weight += alpha ;
			}
		}
		else if (j==degree-1)
		{
			// special treatment of the last node
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
			TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats)
			if (weights_in_tree)
				TreeMem[tree].child_weights[vec[i+j+seq_offset]] += alpha*weights_column[j] ;
			else
				TreeMem[tree].child_weights[vec[i+j+seq_offset]] += alpha;

			break;
		}
		else
		{
			bool use_seq = use_compact_terminal_nodes && (j>degree-16) ;
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats)

			int32_t tmp = get_node((j==degree-2) && (!use_seq));
			if (use_seq)
				TreeMem[tree].children[vec[i+j+seq_offset]] = -tmp ;
			else
				TreeMem[tree].children[vec[i+j+seq_offset]] = tmp ;
			tree=tmp ;

			TRIE_ASSERT((tree>=0) && (tree<TreeMemPtrMax))
#ifdef TRIE_CHECK_EVERYTHING
			TreeMem[tree].has_seq = use_seq ;
#endif
			if (use_seq)
			{
				TreeMem[tree].weight = alpha ;
				// important to have the terminal characters (see ###)
				memset(TreeMem[tree].seq, TRIE_TERMINAL_CHARACTER, 16) ;
				for (int32_t q=0; (j+q<degree) && (i+j+seq_offset+q<length); q++)
				{
					TRIE_ASSERT(q<16)
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
float64_t CTrie<Trie>::compute_by_tree_helper(
	int32_t* vec, int32_t len, int32_t seq_pos, int32_t tree_pos,
		int32_t weight_pos, float64_t* weights,
		bool degree_times_position_weights)
{
	int32_t tree = trees[tree_pos] ;

	if ((position_weights!=NULL) && (position_weights[weight_pos]==0))
		return 0.0;

	float64_t *weights_column=NULL ;
	if (degree_times_position_weights)
		weights_column=&weights[weight_pos*degree] ;
	else // weights is a vector (1 x degree)
		weights_column=weights ;

	float64_t sum=0 ;
	for (int32_t j=0; seq_pos+j < len; j++)
	{
		TRIE_ASSERT((vec[seq_pos+j]<4) && (vec[seq_pos+j]>=0))

		if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
		{
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats)
			if (TreeMem[tree].children[vec[seq_pos+j]]<0)
			{
				tree = - TreeMem[tree].children[vec[seq_pos+j]];
				TRIE_ASSERT(tree>=0)
				TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq)
				float64_t this_weight=0.0 ;
				for (int32_t k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
				{
					TRIE_ASSERT((vec[seq_pos+j+k]<4) && (vec[seq_pos+j+k]>=0))
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
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
				if (weights_in_tree)
					sum += TreeMem[tree].weight ;
				else
					sum += TreeMem[tree].weight * weights_column[j] ;
			} ;
		}
		else
		{
			TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
			if (j==degree-1)
			{
				TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_floats)
				if (weights_in_tree)
					sum += TreeMem[tree].child_weights[vec[seq_pos+j]] ;
				else
					sum += TreeMem[tree].child_weights[vec[seq_pos+j]] * weights_column[j] ;
			}
			else
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_floats)

			break;
		}
	}

	if (position_weights!=NULL)
		return sum*position_weights[weight_pos] ;
	else
		return sum ;
}

	template <class Trie>
void CTrie<Trie>::compute_by_tree_helper(
	int32_t* vec, int32_t len, int32_t seq_pos, int32_t tree_pos,
	int32_t weight_pos, float64_t* LevelContrib, float64_t factor,
	int32_t mkl_stepsize, float64_t * weights,
	bool degree_times_position_weights)
{
	int32_t tree = trees[tree_pos] ;
	if (factor==0)
		return ;

	if (position_weights!=NULL)
	{
		factor *= position_weights[weight_pos] ;
		if (factor==0)
			return ;
		if (!degree_times_position_weights) // with position_weigths, weights is a vector (1 x degree)
		{
			for (int32_t j=0; seq_pos+j<len; j++)
			{
				if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
				{
					if (TreeMem[tree].children[vec[seq_pos+j]]<0)
					{
						tree = -TreeMem[tree].children[vec[seq_pos+j]];
						TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq)
						for (int32_t k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
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
						TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
						if (weights_in_tree)
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
						else
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j] ;
					}
				}
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
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
			for (int32_t j=0; seq_pos+j<len; j++)
			{
				if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
				{
					if (TreeMem[tree].children[vec[seq_pos+j]]<0)
					{
						tree = -TreeMem[tree].children[vec[seq_pos+j]];
						TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq)
						for (int32_t k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
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
						TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
						if (weights_in_tree)
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight ;
						else
							LevelContrib[weight_pos/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j+weight_pos*degree] ;
					}
				}
				else
				{
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
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
		for (int32_t j=0; seq_pos+j<len; j++)
		{
			if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
			{
				if (TreeMem[tree].children[vec[seq_pos+j]]<0)
				{
					tree = -TreeMem[tree].children[vec[seq_pos+j]];
					TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq)
					for (int32_t k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
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
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
					if (weights_in_tree)
						LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].weight ;
					else
						LevelContrib[j/mkl_stepsize] += factor*TreeMem[tree].weight*weights[j] ;
				}
			}
			else
			{
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
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
		  position_mask = SG_MALLOC(bool, len);
		  for (int32_t i=0; i<len; i++)
		  {
		  position_mask[i]=false ;

		  for (int32_t j=0; j<degree; j++)
		  if (weights[i*degree+j]!=0.0)
		  {
		  position_mask[i]=true ;
		  break ;
		  }
		  }
		  }
		  if (position_mask[weight_pos]==0)
		  return ;*/

		for (int32_t j=0; seq_pos+j<len; j++)
		{
			if ((j<degree-1) && (TreeMem[tree].children[vec[seq_pos+j]]!=NO_CHILD))
			{
				if (TreeMem[tree].children[vec[seq_pos+j]]<0)
				{
					tree = -TreeMem[tree].children[vec[seq_pos+j]];
					TRIE_ASSERT_EVERYTHING(TreeMem[tree].has_seq)
					for (int32_t k=0; (j+k<degree) && (seq_pos+j+k<length); k++)
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
					TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
					if (weights_in_tree)
						LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].weight ;
					else
						LevelContrib[(j+degree*weight_pos)/mkl_stepsize] += factor * TreeMem[tree].weight * weights[j+weight_pos*degree] ;
				}
			}
			else
			{
				TRIE_ASSERT_EVERYTHING(!TreeMem[tree].has_seq)
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
void CTrie<Trie>::fill_backtracking_table_recursion(
	Trie* tree, int32_t depth, uint64_t seq, float64_t value,
	DynArray<ConsensusEntry>* table, float64_t* weights)
{
	float64_t w=1.0;

	if (weights_in_tree || depth==0)
		value+=tree->weight;
	else
	{
		w=weights[degree-1];
		value+=weights[depth-1]*tree->weight;
	}

	if (degree-1==depth)
	{
		for (int32_t sym=0; sym<4; sym++)
		{
			float64_t v=w*tree->child_weights[sym];
			if (v!=0.0)
			{
				ConsensusEntry entry;
				entry.bt=-1;
				entry.score=value+v;
				entry.string=seq | ((uint64_t) sym) << (2*(degree-depth-1));

				table->append_element(entry);
			}
		}
	}
	else
	{
		for (int32_t sym=0; sym<4; sym++)
		{
			uint64_t str=seq | ((uint64_t) sym) << (2*(degree-depth-1));
			if (tree->children[sym] != NO_CHILD)
				fill_backtracking_table_recursion(&TreeMem[tree->children[sym]], depth+1, str, value, table, weights);
		}
	}
}

	template <class Trie>
float64_t CTrie<Trie>::get_cumulative_score(
	int32_t pos, uint64_t seq, int32_t deg, float64_t* weights)
{
	float64_t result=0.0;

	//SG_PRINT("pos:%i length:%i deg:%i seq:0x%0llx...\n", pos, length, deg, seq)

	for (int32_t i=pos; i<pos+deg && i<length; i++)
	{
		//SG_PRINT("loop %d\n", i)
		Trie* tree = &TreeMem[trees[i]];

		for (int32_t d=0; d<deg-i+pos; d++)
		{
			//SG_PRINT("loop degree %d shit: %d\n", d, (2*(deg-1-d-i+pos)))
			ASSERT(d-1<degree)
			int32_t sym = (int32_t) (seq >> (2*(deg-1-d-i+pos)) & 3);

			float64_t w=1.0;
			if (!weights_in_tree)
				w=weights[d];

			ASSERT(tree->children[sym] != NO_CHILD)
			tree=&TreeMem[tree->children[sym]];
			result+=w*tree->weight;
		}
	}
	//SG_PRINT("cum: %f\n", result)
	return result;
}

	template <class Trie>
void CTrie<Trie>::fill_backtracking_table(
	int32_t pos, DynArray<ConsensusEntry>* prev,
	DynArray<ConsensusEntry>* cur, bool cumulative, float64_t* weights)
{
	ASSERT(pos>=0 && pos<length)
	ASSERT(!use_compact_terminal_nodes)

	Trie* t = &TreeMem[trees[pos]];

	fill_backtracking_table_recursion(t, 0, (uint64_t) 0, 0.0, cur, weights);


	if (cumulative)
	{
		int32_t num_cur=cur->get_num_elements();
		for (int32_t i=0; i<num_cur; i++)
		{
			ConsensusEntry entry=cur->get_element(i);
			entry.score+=get_cumulative_score(pos+1, entry.string, degree-1, weights);
			cur->set_element(entry,i);
			//SG_PRINT("cum: str:0%0llx sc:%f bt:%d\n",entry.string,entry.score,entry.bt)
		}
	}

	//if previous tree exists find maximum scoring path
	//for each element in cur and update bt table
	if (prev)
	{
		int32_t num_cur=cur->get_num_elements();
		int32_t num_prev=prev->get_num_elements();

		for (int32_t i=0; i<num_cur; i++)
		{
			//uint64_t str_cur_old= cur->get_element(i).string;
			uint64_t str_cur= cur->get_element(i).string >> 2;
			//SG_PRINT("...cur:0x%0llx cur_noprfx:0x%0llx...\n", str_cur_old, str_cur)

			int32_t bt=-1;
			float64_t max_score=0.0;

			for (int32_t j=0; j<num_prev; j++)
			{
				//uint64_t str_prev_old= prev->get_element(j).string;
				uint64_t mask=
					((((uint64_t)0)-1) ^ (((uint64_t) 3) << (2*(degree-1))));
				uint64_t str_prev=  mask & prev->get_element(j).string;
				//SG_PRINT("...prev:0x%0llx prev_nosfx:0x%0llx mask:%0llx...\n", str_prev_old, str_prev,mask)

				if (str_cur == str_prev)
				{
					float64_t sc=prev->get_element(j).score+cur->get_element(i).score;
					if (bt==-1 || sc>max_score)
					{
						bt=j;
						max_score=sc;

						//SG_PRINT("new_max[%i,%i] = %f\n", j,i, max_score)
					}
				}
			}

			ASSERT(bt!=-1)
			ConsensusEntry entry;
			entry.bt=bt;
			entry.score=max_score;
			entry.string=cur->get_element(i).string;
			cur->set_element(entry, i);
			//SG_PRINT("entry[%d]: str:0%0llx sc:%f bt:%d\n",i, entry.string,entry.score,entry.bt)
		}
	}
}
}
#endif // _TRIE_H___
