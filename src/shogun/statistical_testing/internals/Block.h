/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <memory>
#include <vector>
#include <shogun/lib/common.h>

#ifndef BLOCK_H__
#define BLOCK_H__

namespace shogun
{

class CFeatures;

namespace internal
{

/**
 * @brief Class that holds a block feature. A block feature is a shallow
 * copy of an underlying (non-owning) feature object. In its constructor,
 * it increases the refcount of the original object (since it has to be
 * alive as long as the block is alive) and it decreases the refcount of
 * the original object in destructor.
 */
class Block
{
private:
	/**
	 * Constructor to create a block object. It makes a shallow copy of
	 * the underlying feature object, and adds subset according to the
	 * block begin index and the blocksize.
	 *
	 * Increases the reference count of the underlying feature object.
	 *
	 * @param feats The underlying feature object.
	 * @param index The index of the block.
	 * @param size The size of the block (number of feature vectors).
	 */
	Block(CFeatures* feats, index_t index, index_t size);
public:
	/**
	 * Copy constructor. Every time a block is copied or assigned, the underlying
	 * feature object is SG_REF'd.
	 */
	Block(const Block& other);

	/**
	 * Assignment operator. Every time a block is copied or assigned, the underlying
	 * feature object is SG_REF'd.
	 */
	Block& operator=(const Block& other);

	/**
	 * Destructor. Decreases the reference count of the underlying feature object.
	 */
	~Block();

	/**
	 * Method that creates a number of block objects. See @Block for details.
	 *
	 * @param feats The underlying feature object.
	 * @param num_blocks The number of blocks to be formed.
	 * @param size The size of the block (number of feature vectors).
	 */
	static std::vector<Block> create_blocks(CFeatures* feats, index_t num_blocks, index_t size);

	/**
	 * Operator overloading for getting the block object as a shared ptr (non-const).
	 */
	inline operator std::shared_ptr<CFeatures>()
	{
		return m_block;
	}

	/**
	 * Operator overloading for getting the block object as a naked ptr (non-const, unsafe).
	 */
	inline operator CFeatures*()
	{
		return m_block.get();
	}

	/**
	 * Operator overloading for getting the block object as a naked ptr (const).
	 */
	inline operator const CFeatures*() const
	{
		return m_block.get();
	}

	/**
	 * @return the block feature object (non-const, unsafe).
	 */
	inline CFeatures* get()
	{
		return static_cast<CFeatures*>(*this);
	}

	/**
	 * @return the block feature object (const).
	 */
	inline const CFeatures* get() const
	{
		return static_cast<const CFeatures*>(*this);
	}
private:
	/** Shallow copy representing the block */
	std::shared_ptr<CFeatures> m_block;

	/** Underlying feature object */
	CFeatures* m_feats;
};

}

}
#endif // BLOCK_H__
