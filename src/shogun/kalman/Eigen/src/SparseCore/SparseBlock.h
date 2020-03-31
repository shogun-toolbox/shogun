// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_BLOCK_H
#define EIGEN_SPARSE_BLOCK_H

namespace Eigen { 

template<typename XprType, int BlockRows, int BlockCols>
class BlockImpl<XprType,BlockRows,BlockCols,true,Sparse>
  : public SparseMatrixBase<Block<XprType,BlockRows,BlockCols,true> >
{
public:
    typedef Block<XprType, BlockRows, BlockCols, true> BlockType;
    enum { IsRowMajor = internal::traits<BlockType>::IsRowMajor };
protected:
  typedef typename internal::remove_all<typename XprType::Nested>::type _MatrixTypeNested;
    enum { OuterSize = IsRowMajor ? BlockRows : BlockCols };
public:
    EIGEN_SPARSE_PUBLIC_INTERFACE(BlockType)
    
    class InnerIterator: public XprType::InnerIterator
    {
        typedef typename BlockImpl::Index Index;
      public:
        inline InnerIterator(const Block<XprType, BlockRows, BlockCols, true>& xpr, Index outer)
          : XprType::InnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline Index row() const { return IsRowMajor ? m_outer : this->index(); }
        inline Index col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
         Index m_outer;
    };
    class ReverseInnerIterator: public XprType::ReverseInnerIterator
    {
        typedef typename BlockImpl::Index Index;
      public:
        inline ReverseInnerIterator(const BlockType& xpr, Index outer)
          : XprType::ReverseInnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline Index row() const { return IsRowMajor ? m_outer : this->index(); }
        inline Index col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
        Index m_outer;
    };

    inline BlockImpl(const XprType& xpr, int i)
      : m_matrix(xpr), m_outerStart(i), m_outerSize(OuterSize)
    {}

    inline BlockImpl(const XprType& xpr, int startRow, int startCol, int blockRows, int blockCols)
      : m_matrix(xpr), m_outerStart(IsRowMajor ? startRow : startCol), m_outerSize(IsRowMajor ? blockRows : blockCols)
    {}
    
    inline const Scalar coeff(int row, int col) const
    {
      return m_matrix.coeff(row + IsRowMajor ? m_outerStart : 0, col +IsRowMajor ? 0 :  m_outerStart);
    }
    
    inline const Scalar coeff(int index) const
    {
      return m_matrix.coeff(IsRowMajor ? m_outerStart : index, IsRowMajor ? index :  m_outerStart);
    }

    EIGEN_STRONG_INLINE Index rows() const { return IsRowMajor ? m_outerSize.value() : m_matrix.rows(); }
    EIGEN_STRONG_INLINE Index cols() const { return IsRowMajor ? m_matrix.cols() : m_outerSize.value(); }

  protected:

    typename XprType::Nested m_matrix;
    Index m_outerStart;
    const internal::variable_if_dynamic<Index, OuterSize> m_outerSize;

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(BlockImpl)
  private:
    Index nonZeros() const;
};


/***************************************************************************
* specialisation for SparseMatrix
***************************************************************************/

template<typename _Scalar, int _Options, typename _Index, int BlockRows, int BlockCols>
class BlockImpl<SparseMatrix<_Scalar, _Options, _Index>,BlockRows,BlockCols,true,Sparse>
  : public SparseMatrixBase<Block<SparseMatrix<_Scalar, _Options, _Index>,BlockRows,BlockCols,true> >
{
    typedef SparseMatrix<_Scalar, _Options, _Index> SparseMatrixType;
    typedef typename internal::remove_all<typename SparseMatrixType::Nested>::type _MatrixTypeNested;
    typedef Block<const SparseMatrixType, BlockRows, BlockCols, true> ConstBlockType;
public:
  typedef Block<SparseMatrixType, BlockRows, BlockCols, true> BlockType;
    enum { IsRowMajor = internal::traits<BlockType>::IsRowMajor };
    EIGEN_SPARSE_PUBLIC_INTERFACE(BlockType)
protected:
    enum { OuterSize = IsRowMajor ? BlockRows : BlockCols };
public:
    
    class InnerIterator: public SparseMatrixType::InnerIterator
    {
      public:
        inline InnerIterator(const BlockType& xpr, Index outer)
          : SparseMatrixType::InnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline Index row() const { return IsRowMajor ? m_outer : this->index(); }
        inline Index col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
        Index m_outer;
    };
    class ReverseInnerIterator: public SparseMatrixType::ReverseInnerIterator
    {
      public:
        inline ReverseInnerIterator(const BlockType& xpr, Index outer)
          : SparseMatrixType::ReverseInnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline Index row() const { return IsRowMajor ? m_outer : this->index(); }
        inline Index col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
        Index m_outer;
    };

    inline BlockImpl(const SparseMatrixType& xpr, int i)
      : m_matrix(xpr), m_outerStart(i), m_outerSize(OuterSize)
    {}

    inline BlockImpl(const SparseMatrixType& xpr, int startRow, int startCol, int blockRows, int blockCols)
      : m_matrix(xpr), m_outerStart(IsRowMajor ? startRow : startCol), m_outerSize(IsRowMajor ? blockRows : blockCols)
    {}

    template<typename OtherDerived>
    inline BlockType& operator=(const SparseMatrixBase<OtherDerived>& other)
    {
      typedef typename internal::remove_all<typename SparseMatrixType::Nested>::type _NestedMatrixType;
      _NestedMatrixType& matrix = const_cast<_NestedMatrixType&>(m_matrix);;
      // This assignement is slow if this vector set is not empty
      // and/or it is not at the end of the nonzeros of the underlying matrix.

      // 1 - eval to a temporary to avoid transposition and/or aliasing issues
      SparseMatrix<Scalar, IsRowMajor ? RowMajor : ColMajor, Index> tmp(other);

      // 2 - let's check whether there is enough allocated memory
      Index nnz           = tmp.nonZeros();
      Index start         = m_outerStart==0 ? 0 : matrix.outerIndexPtr()[m_outerStart]; // starting position of the current block
      Index end           = m_matrix.outerIndexPtr()[m_outerStart+m_outerSize.value()]; // ending posiiton of the current block
      Index block_size    = end - start;                                                // available room in the current block
      Index tail_size     = m_matrix.outerIndexPtr()[m_matrix.outerSize()] - end;
      
      Index free_size     = m_matrix.isCompressed()
                          ? Index(matrix.data().allocatedSize()) + block_size
                          : block_size;

      if(nnz>free_size) 
      {
        // realloc manually to reduce copies
        typename SparseMatrixType::Storage newdata(m_matrix.data().allocatedSize() - block_size + nnz);

        std::memcpy(newdata.valuePtr(), m_matrix.data().valuePtr(), start*sizeof(Scalar));
        std::memcpy(newdata.indexPtr(), m_matrix.data().indexPtr(), start*sizeof(Index));

        std::memcpy(newdata.valuePtr() + start, tmp.data().valuePtr(), nnz*sizeof(Scalar));
        std::memcpy(newdata.indexPtr() + start, tmp.data().indexPtr(), nnz*sizeof(Index));

        std::memcpy(newdata.valuePtr()+start+nnz, matrix.data().valuePtr()+end, tail_size*sizeof(Scalar));
        std::memcpy(newdata.indexPtr()+start+nnz, matrix.data().indexPtr()+end, tail_size*sizeof(Index));
        
        newdata.resize(m_matrix.outerIndexPtr()[m_matrix.outerSize()] - block_size + nnz);

        matrix.data().swap(newdata);
      }
      else
      {
        // no need to realloc, simply copy the tail at its respective position and insert tmp
        matrix.data().resize(start + nnz + tail_size);

        std::memmove(matrix.data().valuePtr()+start+nnz, matrix.data().valuePtr()+end, tail_size*sizeof(Scalar));
        std::memmove(matrix.data().indexPtr()+start+nnz, matrix.data().indexPtr()+end, tail_size*sizeof(Index));

        std::memcpy(matrix.data().valuePtr()+start, tmp.data().valuePtr(), nnz*sizeof(Scalar));
        std::memcpy(matrix.data().indexPtr()+start, tmp.data().indexPtr(), nnz*sizeof(Index));
      }
      
      // update innerNonZeros
      if(!m_matrix.isCompressed())
        for(Index j=0; j<m_outerSize.value(); ++j)
          matrix.innerNonZeroPtr()[m_outerStart+j] = tmp.innerVector(j).nonZeros();

      // update outer index pointers
      Index p = start;
      for(Index k=0; k<m_outerSize.value(); ++k)
      {
        matrix.outerIndexPtr()[m_outerStart+k] = p;
        p += tmp.innerVector(k).nonZeros();
      }
      std::ptrdiff_t offset = nnz - block_size;
      for(Index k = m_outerStart + m_outerSize.value(); k<=matrix.outerSize(); ++k)
      {
        matrix.outerIndexPtr()[k] += offset;
      }

      return derived();
    }

    inline BlockType& operator=(const BlockType& other)
    {
      return operator=<BlockType>(other);
    }

    inline const Scalar* valuePtr() const
    { return m_matrix.valuePtr() + m_matrix.outerIndexPtr()[m_outerStart]; }
    inline Scalar* valuePtr()
    { return m_matrix.const_cast_derived().valuePtr() + m_matrix.outerIndexPtr()[m_outerStart]; }

    inline const Index* innerIndexPtr() const
    { return m_matrix.innerIndexPtr() + m_matrix.outerIndexPtr()[m_outerStart]; }
    inline Index* innerIndexPtr()
    { return m_matrix.const_cast_derived().innerIndexPtr() + m_matrix.outerIndexPtr()[m_outerStart]; }

    inline const Index* outerIndexPtr() const
    { return m_matrix.outerIndexPtr() + m_outerStart; }
    inline Index* outerIndexPtr()
    { return m_matrix.const_cast_derived().outerIndexPtr() + m_outerStart; }

    Index nonZeros() const
    {
      if(m_matrix.isCompressed())
        return  std::size_t(m_matrix.outerIndexPtr()[m_outerStart+m_outerSize.value()])
              - std::size_t(m_matrix.outerIndexPtr()[m_outerStart]);
      else if(m_outerSize.value()==0)
        return 0;
      else
        return Map<const Matrix<Index,OuterSize,1> >(m_matrix.innerNonZeroPtr()+m_outerStart, m_outerSize.value()).sum();
    }
    
    inline Scalar& coeffRef(int row, int col)
    {
      return m_matrix.const_cast_derived().coeffRef(row + (IsRowMajor ? m_outerStart : 0), col + (IsRowMajor ? 0 :  m_outerStart));
    }
    
    inline const Scalar coeff(int row, int col) const
    {
      return m_matrix.coeff(row + (IsRowMajor ? m_outerStart : 0), col + (IsRowMajor ? 0 :  m_outerStart));
    }
    
    inline const Scalar coeff(int index) const
    {
      return m_matrix.coeff(IsRowMajor ? m_outerStart : index, IsRowMajor ? index :  m_outerStart);
    }

    const Scalar& lastCoeff() const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(BlockImpl);
      eigen_assert(nonZeros()>0);
      if(m_matrix.isCompressed())
        return m_matrix.valuePtr()[m_matrix.outerIndexPtr()[m_outerStart+1]-1];
      else
        return m_matrix.valuePtr()[m_matrix.outerIndexPtr()[m_outerStart]+m_matrix.innerNonZeroPtr()[m_outerStart]-1];
    }

    EIGEN_STRONG_INLINE Index rows() const { return IsRowMajor ? m_outerSize.value() : m_matrix.rows(); }
    EIGEN_STRONG_INLINE Index cols() const { return IsRowMajor ? m_matrix.cols() : m_outerSize.value(); }

  protected:

    typename SparseMatrixType::Nested m_matrix;
    Index m_outerStart;
    const internal::variable_if_dynamic<Index, OuterSize> m_outerSize;

};


template<typename _Scalar, int _Options, typename _Index, int BlockRows, int BlockCols>
class BlockImpl<const SparseMatrix<_Scalar, _Options, _Index>,BlockRows,BlockCols,true,Sparse>
  : public SparseMatrixBase<Block<const SparseMatrix<_Scalar, _Options, _Index>,BlockRows,BlockCols,true> >
{
    typedef SparseMatrix<_Scalar, _Options, _Index> SparseMatrixType;
    typedef typename internal::remove_all<typename SparseMatrixType::Nested>::type _MatrixTypeNested;
public:
  typedef Block<const SparseMatrixType, BlockRows, BlockCols, true> BlockType;
    enum { IsRowMajor = internal::traits<BlockType>::IsRowMajor };
    EIGEN_SPARSE_PUBLIC_INTERFACE(BlockType)
protected:
    enum { OuterSize = IsRowMajor ? BlockRows : BlockCols };
public:
    
    class InnerIterator: public SparseMatrixType::InnerIterator
    {
      public:
        inline InnerIterator(const BlockType& xpr, Index outer)
          : SparseMatrixType::InnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline Index row() const { return IsRowMajor ? m_outer : this->index(); }
        inline Index col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
        Index m_outer;
    };
    class ReverseInnerIterator: public SparseMatrixType::ReverseInnerIterator
    {
      public:
        inline ReverseInnerIterator(const BlockType& xpr, Index outer)
          : SparseMatrixType::ReverseInnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline Index row() const { return IsRowMajor ? m_outer : this->index(); }
        inline Index col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
        Index m_outer;
    };

    inline BlockImpl(const SparseMatrixType& xpr, int i)
      : m_matrix(xpr), m_outerStart(i), m_outerSize(OuterSize)
    {}

    inline BlockImpl(const SparseMatrixType& xpr, int startRow, int startCol, int blockRows, int blockCols)
      : m_matrix(xpr), m_outerStart(IsRowMajor ? startRow : startCol), m_outerSize(IsRowMajor ? blockRows : blockCols)
    {}

    inline const Scalar* valuePtr() const
    { return m_matrix.valuePtr() + m_matrix.outerIndexPtr()[m_outerStart]; }

    inline const Index* innerIndexPtr() const
    { return m_matrix.innerIndexPtr() + m_matrix.outerIndexPtr()[m_outerStart]; }

    inline const Index* outerIndexPtr() const
    { return m_matrix.outerIndexPtr() + m_outerStart; }

    Index nonZeros() const
    {
      if(m_matrix.isCompressed())
        return  std::size_t(m_matrix.outerIndexPtr()[m_outerStart+m_outerSize.value()])
              - std::size_t(m_matrix.outerIndexPtr()[m_outerStart]);
      else if(m_outerSize.value()==0)
        return 0;
      else
        return Map<const Matrix<Index,OuterSize,1> >(m_matrix.innerNonZeroPtr()+m_outerStart, m_outerSize.value()).sum();
    }
    
    inline const Scalar coeff(int row, int col) const
    {
      return m_matrix.coeff(row + (IsRowMajor ? m_outerStart : 0), col + (IsRowMajor ? 0 :  m_outerStart));
    }
    
    inline const Scalar coeff(int index) const
    {
      return m_matrix.coeff(IsRowMajor ? m_outerStart : index, IsRowMajor ? index :  m_outerStart);
    }

    const Scalar& lastCoeff() const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(BlockImpl);
      eigen_assert(nonZeros()>0);
      if(m_matrix.isCompressed())
        return m_matrix.valuePtr()[m_matrix.outerIndexPtr()[m_outerStart+1]-1];
      else
        return m_matrix.valuePtr()[m_matrix.outerIndexPtr()[m_outerStart]+m_matrix.innerNonZeroPtr()[m_outerStart]-1];
    }

    EIGEN_STRONG_INLINE Index rows() const { return IsRowMajor ? m_outerSize.value() : m_matrix.rows(); }
    EIGEN_STRONG_INLINE Index cols() const { return IsRowMajor ? m_matrix.cols() : m_outerSize.value(); }

  protected:

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(BlockImpl)

    typename SparseMatrixType::Nested m_matrix;
    Index m_outerStart;
    const internal::variable_if_dynamic<Index, OuterSize> m_outerSize;
};

//----------

/** \returns the \a outer -th column (resp. row) of the matrix \c *this if \c *this
  * is col-major (resp. row-major).
  */
template<typename Derived>
typename SparseMatrixBase<Derived>::InnerVectorReturnType SparseMatrixBase<Derived>::innerVector(Index outer)
{ return InnerVectorReturnType(derived(), outer); }

/** \returns the \a outer -th column (resp. row) of the matrix \c *this if \c *this
  * is col-major (resp. row-major). Read-only.
  */
template<typename Derived>
const typename SparseMatrixBase<Derived>::ConstInnerVectorReturnType SparseMatrixBase<Derived>::innerVector(Index outer) const
{ return ConstInnerVectorReturnType(derived(), outer); }

/** \returns the \a outer -th column (resp. row) of the matrix \c *this if \c *this
  * is col-major (resp. row-major).
  */
template<typename Derived>
typename SparseMatrixBase<Derived>::InnerVectorsReturnType
SparseMatrixBase<Derived>::innerVectors(Index outerStart, Index outerSize)
{
  return Block<Derived,Dynamic,Dynamic,true>(derived(),
                                             IsRowMajor ? outerStart : 0, IsRowMajor ? 0 : outerStart,
                                             IsRowMajor ? outerSize : rows(), IsRowMajor ? cols() : outerSize);
  
}

/** \returns the \a outer -th column (resp. row) of the matrix \c *this if \c *this
  * is col-major (resp. row-major). Read-only.
  */
template<typename Derived>
const typename SparseMatrixBase<Derived>::ConstInnerVectorsReturnType
SparseMatrixBase<Derived>::innerVectors(Index outerStart, Index outerSize) const
{
  return Block<const Derived,Dynamic,Dynamic,true>(derived(),
                                                  IsRowMajor ? outerStart : 0, IsRowMajor ? 0 : outerStart,
                                                  IsRowMajor ? outerSize : rows(), IsRowMajor ? cols() : outerSize);
  
}

namespace internal {
  
template< typename XprType, int BlockRows, int BlockCols, bool InnerPanel,
          bool OuterVector =  (BlockCols==1 && XprType::IsRowMajor) || (BlockRows==1 && !XprType::IsRowMajor)>
class GenericSparseBlockInnerIteratorImpl;

}

/** Generic implementation of sparse Block expression.
  * Real-only. 
  */
template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
class BlockImpl<XprType,BlockRows,BlockCols,InnerPanel,Sparse>
  : public SparseMatrixBase<Block<XprType,BlockRows,BlockCols,InnerPanel> >, internal::no_assignment_operator
{
    typedef typename internal::remove_all<typename XprType::Nested>::type _MatrixTypeNested;
  public:
    typedef Block<XprType, BlockRows, BlockCols, InnerPanel> BlockType;
    enum { IsRowMajor = internal::traits<BlockType>::IsRowMajor };
    EIGEN_SPARSE_PUBLIC_INTERFACE(BlockType)

    /** Column or Row constructor
      */
    inline BlockImpl(const XprType& xpr, int i)
      : m_matrix(xpr),
        m_startRow( (BlockRows==1) && (BlockCols==XprType::ColsAtCompileTime) ? i : 0),
        m_startCol( (BlockRows==XprType::RowsAtCompileTime) && (BlockCols==1) ? i : 0),
        m_blockRows(BlockRows==1 ? 1 : xpr.rows()),
        m_blockCols(BlockCols==1 ? 1 : xpr.cols())
    {}

    /** Dynamic-size constructor
      */
    inline BlockImpl(const XprType& xpr, int startRow, int startCol, int blockRows, int blockCols)
      : m_matrix(xpr), m_startRow(startRow), m_startCol(startCol), m_blockRows(blockRows), m_blockCols(blockCols)
    {}

    inline int rows() const { return m_blockRows.value(); }
    inline int cols() const { return m_blockCols.value(); }

    inline Scalar& coeffRef(int row, int col)
    {
      return m_matrix.const_cast_derived()
               .coeffRef(row + m_startRow.value(), col + m_startCol.value());
    }

    inline const Scalar coeff(int row, int col) const
    {
      return m_matrix.coeff(row + m_startRow.value(), col + m_startCol.value());
    }

    inline Scalar& coeffRef(int index)
    {
      return m_matrix.const_cast_derived()
             .coeffRef(m_startRow.value() + (RowsAtCompileTime == 1 ? 0 : index),
                       m_startCol.value() + (RowsAtCompileTime == 1 ? index : 0));
    }

    inline const Scalar coeff(int index) const
    {
      return m_matrix
             .coeff(m_startRow.value() + (RowsAtCompileTime == 1 ? 0 : index),
                    m_startCol.value() + (RowsAtCompileTime == 1 ? index : 0));
    }
    
    inline const _MatrixTypeNested& nestedExpression() const { return m_matrix; }
    
    typedef internal::GenericSparseBlockInnerIteratorImpl<XprType,BlockRows,BlockCols,InnerPanel> InnerIterator;
    
    class ReverseInnerIterator : public _MatrixTypeNested::ReverseInnerIterator
    {
      typedef typename _MatrixTypeNested::ReverseInnerIterator Base;
      const BlockType& m_block;
      Index m_begin;
    public:

      EIGEN_STRONG_INLINE ReverseInnerIterator(const BlockType& block, Index outer)
        : Base(block.derived().nestedExpression(), outer + (IsRowMajor ? block.m_startRow.value() : block.m_startCol.value())),
          m_block(block),
          m_begin(IsRowMajor ? block.m_startCol.value() : block.m_startRow.value())
      {
        while( (Base::operator bool()) && (Base::index() >= (IsRowMajor ? m_block.m_startCol.value()+block.m_blockCols.value() : m_block.m_startRow.value()+block.m_blockRows.value())) )
          Base::operator--();
      }

      inline Index index()  const { return Base::index() - (IsRowMajor ? m_block.m_startCol.value() : m_block.m_startRow.value()); }
      inline Index outer()  const { return Base::outer() - (IsRowMajor ? m_block.m_startRow.value() : m_block.m_startCol.value()); }
      inline Index row()    const { return Base::row()   - m_block.m_startRow.value(); }
      inline Index col()    const { return Base::col()   - m_block.m_startCol.value(); }
      
      inline operator bool() const { return Base::operator bool() && Base::index() >= m_begin; }
    };
  protected:
    friend class internal::GenericSparseBlockInnerIteratorImpl<XprType,BlockRows,BlockCols,InnerPanel>;
    friend class ReverseInnerIterator;
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(BlockImpl)

    typename XprType::Nested m_matrix;
    const internal::variable_if_dynamic<Index, XprType::RowsAtCompileTime == 1 ? 0 : Dynamic> m_startRow;
    const internal::variable_if_dynamic<Index, XprType::ColsAtCompileTime == 1 ? 0 : Dynamic> m_startCol;
    const internal::variable_if_dynamic<Index, RowsAtCompileTime> m_blockRows;
    const internal::variable_if_dynamic<Index, ColsAtCompileTime> m_blockCols;
  private:
    Index nonZeros() const;
};

namespace internal {
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  class GenericSparseBlockInnerIteratorImpl<XprType,BlockRows,BlockCols,InnerPanel,false> : public internal::remove_all<typename XprType::Nested>::type::InnerIterator
  {
  public:
    typedef Block<XprType, BlockRows, BlockCols, InnerPanel> BlockType;
    enum {
      IsRowMajor = BlockType::IsRowMajor
    };
    typedef typename BlockType::Index Index;
  protected:
    typedef typename internal::remove_all<typename XprType::Nested>::type _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator Base;
    const BlockType& m_block;
    Index m_end;
  public:

    EIGEN_STRONG_INLINE GenericSparseBlockInnerIteratorImpl(const BlockType& block, Index outer)
      : Base(block.derived().nestedExpression(), outer + (IsRowMajor ? block.m_startRow.value() : block.m_startCol.value())),
        m_block(block),
        m_end(IsRowMajor ? block.m_startCol.value()+block.m_blockCols.value() : block.m_startRow.value()+block.m_blockRows.value())
    {
      while( (Base::operator bool()) && (Base::index() < (IsRowMajor ? m_block.m_startCol.value() : m_block.m_startRow.value())) )
        Base::operator++();
    }
    
    inline Index index()  const { return Base::index() - (IsRowMajor ? m_block.m_startCol.value() : m_block.m_startRow.value()); }
    inline Index outer()  const { return Base::outer() - (IsRowMajor ? m_block.m_startRow.value() : m_block.m_startCol.value()); }
    inline Index row()    const { return Base::row()   - m_block.m_startRow.value(); }
    inline Index col()    const { return Base::col()   - m_block.m_startCol.value(); }
    
    inline operator bool() const { return Base::operator bool() && Base::index() < m_end; }
  };
  
  // Row vector of a column-major sparse matrix or column of a row-major one.
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  class GenericSparseBlockInnerIteratorImpl<XprType,BlockRows,BlockCols,InnerPanel,true>
  {
  public:
    typedef Block<XprType, BlockRows, BlockCols, InnerPanel> BlockType;
    enum {
      IsRowMajor = BlockType::IsRowMajor
    };
    typedef typename BlockType::Index Index;
    typedef typename BlockType::Scalar Scalar;
  protected:
    typedef typename internal::remove_all<typename XprType::Nested>::type _MatrixTypeNested;
    const BlockType& m_block;
    Index m_outerPos;
    Index m_innerIndex;
    Scalar m_value;
    Index m_end;
  public:

    EIGEN_STRONG_INLINE GenericSparseBlockInnerIteratorImpl(const BlockType& block, Index outer = 0)
      : 
        m_block(block),
        m_outerPos( (IsRowMajor ? block.m_startCol.value() : block.m_startRow.value()) - 1), // -1 so that operator++ finds the first non-zero entry
        m_innerIndex(IsRowMajor ? block.m_startRow.value() : block.m_startCol.value()),
        m_end(IsRowMajor ? block.m_startCol.value()+block.m_blockCols.value() : block.m_startRow.value()+block.m_blockRows.value())
    {
      EIGEN_UNUSED_VARIABLE(outer);
      eigen_assert(outer==0);
      
      ++(*this);
    }
    
    inline Index index()  const { return m_outerPos - (IsRowMajor ? m_block.m_startCol.value() : m_block.m_startRow.value()); }
    inline Index outer()  const { return 0; }
    inline Index row()    const { return IsRowMajor ? 0 : index(); }
    inline Index col()    const { return IsRowMajor ? index() : 0; }
    
    inline Scalar value() const { return m_value; }
    
    inline GenericSparseBlockInnerIteratorImpl& operator++()
    {
      // search next non-zero entry
      while(m_outerPos<m_end)
      {
        m_outerPos++;
        typename XprType::InnerIterator it(m_block.m_matrix, m_outerPos);
        // search for the key m_innerIndex in the current outer-vector
        while(it && it.index() < m_innerIndex) ++it;
        if(it && it.index()==m_innerIndex)
        {
          m_value = it.value();
          break;
        }
      }
      return *this;
    }
    
    inline operator bool() const { return m_outerPos < m_end; }
  };
  
} // end namespace internal


} // end namespace Eigen

#endif // EIGEN_SPARSE_BLOCK_H
