/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Thoralf Klein, Yuyu Zhang, Bjoern Esser
 */

#ifndef INDEXBLOCKTREE_H_
#define INDEXBLOCKTREE_H_

#include <shogun/lib/config.h>

#include <shogun/lib/IndexBlockRelation.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>

namespace shogun
{
class IndexBlock;

/** @brief class IndexBlockTree used to represent
 * tree guided feature relation.
 *
 * Can be constructed via CIndexBlock instance having
 * sub blocks, adjacency matrix or precomputed indices.
 */
class IndexBlockTree : public IndexBlockRelation
{
public:

	/** default constructor */
	IndexBlockTree();

	/** constructor from index block
	 * @param root_block root block of the tree
	 */
	IndexBlockTree(std::shared_ptr<IndexBlock> root_block);

	/** constructor from adjacency matrix
	 * @param adjacency_matrix adjacency matrix
	 * @param include_supernode whether to include supernode
	 */
	IndexBlockTree(SGMatrix<float64_t> adjacency_matrix, bool include_supernode);

#ifndef SWIG
	/** constructor from general precomputed indices
	 * each node is represented with indices G[ind_t.min:ind_t.max]
	 * and weight ind_t.weight
	 * @param G custom G containing mapping indices
	 * @param ind_t custom ind_t containing flatten parameters of each node [min,max,weight]
	 */
	IndexBlockTree(const SGVector<float64_t>& G, const SGVector<float64_t>& ind_t);
#endif

	/** constructor from basic precomputed indices
	 * each node is represented with indices ind_t.min:ind_t.max
	 * and weight ind_t.weight
	 * @param ind_t custom ind_t containing flatten parameters of each node [min,max,weight]
	 */
	IndexBlockTree(SGVector<float64_t> ind_t);

	/** destructor */
	virtual ~IndexBlockTree();

	/** get root IndexBlock */
	std::shared_ptr<IndexBlock> get_root_block() const;

	/** set root block */
	void set_root_block(std::shared_ptr<IndexBlock> root_block);

	/** returns information about blocks in
	 * SLEP "ind" format
	 */
	virtual SGVector<index_t> get_SLEP_ind();

	/** returns information about blocks in
	 * SLEP "G" format
	 */
	virtual SGVector<float64_t> get_SLEP_G();

	/** returns information about blocks relations
	 * in SLEP "ind_t" format
	 */
	virtual SGVector<float64_t> get_SLEP_ind_t() const;

	/** returns relation type */
	virtual EIndexBlockRelationType get_relation_type() const { return TREE; }

	/** whether relation is general, i.e. not well ordered */
	bool is_general() const;

	/** get name */
	const char* get_name() const { return "IndexBlockTree"; };

protected:

	/** root block */
	std::shared_ptr<IndexBlock> m_root_block;

	/** general */
	bool m_general;

	/** precomputed ind_t */
	SGVector<float64_t> m_precomputed_ind_t;

	/** precomputed G */
	SGVector<float64_t> m_precomputed_G;
};

}
#endif

