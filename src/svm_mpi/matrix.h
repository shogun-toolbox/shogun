/* -*-C++-*- */

/*
 * matrix.h : definitions for matrix class
 * Copyright (C) 2000-2001 The Australian National University
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef __MATRIX_H_
#define __MATRIX_H_

#include "config.h"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <cmath>
#include <cstdarg>
#ifdef HAVE_MATLAB
#include "mat.h"
#endif /* ! NO_MATLAB */
#ifdef DISABLE_MPI
#undef HAVE_MPI
#endif
#ifdef HAVE_MPI
#include "mpi.h"
#endif /* ! NO_MPI */
#include "bcache.h"

#ifndef MIN
#define MIN(a, b) (((a) > (b))?(b):(a))
#endif

#ifndef MAX
#define MAX(a, b) (((a) < (b))?(b):(a))
#endif

using namespace std;

template <class T> class CMatrix;

//! Indexing class for CMatrix
/*!
  This class contains two unsigned integer values are supposed to represent
  indices into a matrix.  This is used by operator() in CMatrix to refer to
  submatrices in a way that is largely compatible with the colon notation in
  Matlab.  The usual way to use this class is via the functions colon() 
  (see colon() documentation for examples).  Currently this can only be used
  to refer to contiguous regions within a matrix; Matlab constructs like
  \c foo(3:2:20,4:3:9) are not supported.
 */
class CMatrixIndex
{
public:
  //! Creates a set of indices that covers all elements (equivalent to (:) in Matlab)
  CMatrixIndex(void) : m_StartIdx(0), m_EndIdx(UINT_MAX) { }
  /**
   *  Creates a set of indices that contains one element only (equivalent to 
   * (idx) in Matlab).
   * @param idx the index of the desired element.
   */
  CMatrixIndex(unsigned idx) : m_StartIdx(idx), m_EndIdx(idx) { }
  /**
   * Creates a set of indices in the specified range (equivalent to 
   * (start:end) in Matlab).
   * @param start is the first index in the desired range
   * @param end is the last index in the desired range
   */
  CMatrixIndex(unsigned start, unsigned end) : 
    m_StartIdx((end < start)?end:start), m_EndIdx((end < start)?start:end) 
  { if (end < start) abort(); }
  //! Destructor; does nothing but free memory
  ~CMatrixIndex(void) { /* nothing to do */ }

  friend class CMatrix<double>;
  friend class CMatrix<float>;
  friend class CMatrix<unsigned char>;

private:
  unsigned m_StartIdx;
  unsigned m_EndIdx;
};

// Docs for these are in the .cpp file
CMatrixIndex colon(void);
CMatrixIndex colon(const unsigned idx);
CMatrixIndex colon(const unsigned start, const unsigned end);

//! Matrix implementation class (private to CMatrix, not designed for isolated use)
/*!
  This class contains the actual values for a particular matrix or submatrix.
  Each CMatrix object contains a (private) pointer to an object of this type.
  Memory management is done automatically via the reference count contained
  within this object; when it drops to zero (meaning there are no CMatrix
  objects left referring to this implementation) the implementation is freed.
 */
template <class T>
class CMatrixImpl
{
public:
  /**
   * Creates a new matrix implementation of the specified size.
   * @param nrows is the number of rows in the new implementation.
   * @param ncols is the number of columns in the new implementation.
   */
  CMatrixImpl(const unsigned nrows, const unsigned ncols,
	      const unsigned long sizehint = 0);
  /**
   */
  CMatrixImpl(const unsigned nrows, const unsigned ncols, T *data,
	      void (*freefunc)(void *));
#ifdef HAVE_MATLAB
  /**
   * Creates a new matrix implementation whose values are copied from the
   * specified Matlab matrix.
   * @param m is a pointer to a Matlab matrix.
   */
  CMatrixImpl(mxArray *m);
#endif /* HAVE_MATLAB */
  /**
   * Matrix data is destroyed and this object is freed.
   */
  ~CMatrixImpl(void);

  //! @return The number of rows in this implementation.
  inline unsigned GetNumRows(void) const { return (m_nRows); }
  //! @return The number of columns in this implementation.
  inline unsigned GetNumColumns(void) const { return (m_nCols); }
  //! @return The number of elements (rows * columns) in this implementation.
  inline unsigned GetNumElements(void) const { return (m_nRows * m_nCols); }

  //! Increment this implementation's reference count.
  void AddRef(void);
  //! Decrement this implementation's reference count (a possible side-effect of this is that the object may be destroyed, be careful!)
  void DeleteRef(void);
  /**
   * If this implementation has more than one reference then a new 
   * implementation is returned with a new set of copied values.  If there
   * is only one reference to this matrix this the method returns @c this.
   * This method needs to \b always be called if the values of the matrix
   * are to be changed (otherwise any changes made will propagate elsewhere
   * where they are not wanted).
   * @param ptr (optional, can be @c NULL) is a pointer to somewhere within the current value array.  The value of the pointer will be adjusted so that it points to the equivalent location inside the new implementation.
   * @param transpose whether the matrix values should be transposed @b if a copy is made.
   * @return A new implementation (with a reference count of one, that can be modified at will by the caller without disturbing the other matrices that have a handle on the same implementation) or @c this if there are no other references.
   */
  CMatrixImpl *Dirty(T **ptr, const bool transpose);

  /**
   * Resizes the implementation values array to the specified new size.  Values
   * from the old matrix whose indices exist in the new matrix are copied,
   * any values that did not exist in the first matrix (because the new matrix
   * is large in one or both dimensions) are zeroed.
   * @param nrows is the number of rows in the implementation after resizing.
   * @param ncols is the number of columns in the implementation after resizing.
   * @param ptr (optional, can be @c NULL) is a pointer to somewhere within the current value array.  The value of the pointer will be adjusted so that it points to the equivalent location inside the new implementation.
   */
  CMatrixImpl<T> *Resize(const unsigned nrows, const unsigned ncols,
			 T **ptr);

  /**
   * Test if a pointer is inside or outside of the
   * data area.  This is primarily for debugging purposes, to see if a 
   * particular pointer has been scrambled.
   * @param ptr is a pointer to test.
   * @return @c true if the pointer is inside the data area, otherwise @false.
   */
  inline bool IsPtrInside(const T *ptr) const 
  { return ((ptr >= m_Data) && (ptr < (m_Data + m_nRows * m_nCols))); }

  /**
   * Gives this implementation a "hint" that the matrix is expected to grow
   * to @nelements elements in size.  This is useful for matrices that are
   * built incrementally because this avoids reallocation on each iteration.
   * It is especially useful in the case where columns only are added since 
   * the matrix can grow without having to rearrange its values.  If the
   * matrix is re-sized to an extent that the hint is too small, it is reset
   * to zero (no hint); similarily if @nelements is less than or equal to
   * the number of elements currently allocated this hint is ignored.
   */
  void ProvideSizeHint(const unsigned long nelements) { if (nelements > m_nAllocedElements) m_SizeHint = nelements; }

  friend class CMatrix<T>;
private:
  inline T *GetDataPtr(const unsigned i, const unsigned j) { 
    assert((i < m_nRows) && (j < m_nCols));
    return (m_Data + j * m_nRows + i);
  }
  inline const T *GetDataPtr(const unsigned i, const unsigned j) const { 
    assert((i < m_nRows) && (j < m_nCols));
    return (m_Data + j * m_nRows + i); 
  }

  inline T *GetElementPtr(const unsigned i, const unsigned j, 
			  const bool transposed) { 
    assert(((transposed?j:i) < m_nRows) && ((transposed?i:j) < m_nCols));
    return (m_Data + (transposed?i:j) * m_nRows + (transposed?j:i));
  }
  inline const T *GetElementPtr(const unsigned i, const unsigned j,
				const bool transposed) const {
    assert(((transposed?j:i) < m_nRows) && ((transposed?i:j) < m_nCols));
    return (m_Data + (transposed?i:j) * m_nRows + (transposed?j:i));
  }
  void GetElementIdxFromPtr(const T *ptr, unsigned *i, unsigned *j,
			    const bool transposed) const;

  //! Number of rows
  unsigned m_nRows;
  //! Number of columns
  unsigned m_nCols;
  //! Pointer to values (in column-major [Fortran] order)
  T *m_Data;
  //! Number of references currently active on this implementation (object will be freed when this drops to zero)
  unsigned m_nRefs;
  //! Number of elements that this matrix is likely to grow to (supplied externally as a "hint")
  unsigned m_SizeHint;
  //! Number of elements that are currently allocated to m_Data
  unsigned m_nAllocedElements;
  //!
  void (*m_FreeFunc)(void *);
};

template <class T> struct __matrix_iterator;
template <class T> struct __matrix_const_iterator;

/* Need to pre-declare friend functions here */
//! @return The sum of the two specified matrices.
template <class T> CMatrix<T> operator+(const CMatrix<T> &arg1,
					const CMatrix<T> &arg2);
//!
template <class T> CMatrix<T> operator+(const CMatrix<T> &arg1,
					const T arg2);
//! @return The difference between the two specified matrices.
template <class T> CMatrix<T> operator-(const CMatrix<T> &arg1,
					const CMatrix<T> &arg2);
//! @return The element-by-element difference between the matrix and the specified value.
template <class T> CMatrix<T> operator-(const CMatrix<T> &arg1,
					const T arg2);
//! @return The product of the two specified matrices.
template <class T> CMatrix<T> operator*(const CMatrix<T> &arg1,
					const CMatrix<T> &arg2);
//! @return The product of the specified matrix and a scalar.
template <class T> CMatrix<T> operator*(const T val,
					const CMatrix<T> &m);
//! @return The product of the specified matrix and a scalar.
template <class T> CMatrix<T> operator*(const CMatrix<T> &m,
					const T val);
template <class T> CMatrix<T> operator/(const CMatrix<T> &arg1,
					const CMatrix<T> &arg2);
//! @return The element-by-element division of the specified matrix by the specified value.
template <class T> CMatrix<T> operator/(const CMatrix<T> &m, const double val);
//! @return @c true if the two specified matrices are equal.
template <class T> bool operator==(const CMatrix<T> &arg1, 
				   const CMatrix<T> &arg2);
//! @ return The element-by-element comparison between the matrix and the value.
template <class T> CMatrix<T> operator==(const CMatrix<T> &arg1, const T arg2);
//! @return @c true if the two specified matrices are not equal.
template <class T> bool operator!=(const CMatrix<T> &arg1, 
				   const CMatrix<T> &arg2);
//! Prints a pretty version of the matrix's contents on the specified stream.
template <class T> ostream &operator<<(ostream &os, const CMatrix<T> &m);
//! @return The negative of the specified matrix.
template <class T> CMatrix<T> operator-(const CMatrix<T> &m);
//! @return
template <class T> CMatrix<T> pow(const CMatrix<T> &m, const T power);
/**
 * Creates a matrix whose values are all unity.  Note that since the template
 * type is not used in any of the parameters you need to qualify this when
 * using it in applications, e.g. @c ones<double>(3,3) rather than
 * @c ones(3,3) (which is ambiguous).
 * @param nrows is the number of rows for the new matrix.
 * @param nrows is the number of columns for the new matrix.
 */
template <class T> CMatrix<T> ones(const unsigned nrows, const unsigned ncols);
/**
 */
template <class T> CMatrix<T> ones(const unsigned ndims);
/**
 * Creates a matrix whose values are all zero.  Note that since the template
 * type is not used in any of the parameters you need to qualify this when
 * using it in applications, e.g. @c zeros<double>(3,3) rather than
 * @c zeros(3,3) (which is ambiguous).
 * @param nrows is the number of rows for the new matrix.
 * @param nrows is the number of columns for the new matrix.
 */
template <class T> CMatrix<T> zeros(const unsigned nrows, 
				    const unsigned ncols);
/**
 */
template <class T> CMatrix<T> zeros(const unsigned ndims);
/**
 * Creates an identity matrix of the specified size.  Note that since the template
 * type is not used in any of the parameters you need to qualify this when
 * using it in applications, e.g. @c eye<double>(3) rather than
 * @c eye(3) (which is ambiguous).
 * @param size is the number of rows and columns for the new matrix
 */
template <class T> CMatrix<T> eye(const unsigned size);
//! @return The Cholesky decomposition of the specified matrix.
template <class T> CMatrix<T> chol(const CMatrix<T> &m, const bool upper = true, const bool eraseother = true);
//! @return A vector containing the diagonal terms of the specified matrix.
template <class T> CMatrix<T> diag(const CMatrix<T> &m);
//! @return A new matrix containing the absolute values of those in the specified matrix.
template <class T> CMatrix<T> abs(const CMatrix<T> &m);
//! @return A row vector containing the minimum values of the columns of the specified matrix.
template <class T> CMatrix<T> min(const CMatrix<T> &m);
//! @return A row vector containing the maximum values of the columns of the specified matrix.
template <class T> CMatrix<T> max(const CMatrix<T> &m);
//! @return The index of the element with the maximum value in the specified row or column vector.
template <class T> unsigned max_idx(const CMatrix<T> &m);
//! @return The transpose of the specified matrix.
template <class T> CMatrix<T> ctranspose(const CMatrix<T> &m);
//! @return The maximum singular value of the specified matrix.
template <class T> double norm(const CMatrix<T> &m);
//! @return A matrix containing the element-by-element minimum of the specified matrix and the specified value.
template <class T> CMatrix<T> min(const CMatrix<T> &m, const T bound);
//!
template <class T> CMatrix<T> min(const CMatrix<T> &m1, const CMatrix<T> &m2);
//! @return A matrix containing the element-by-element maximum of the specified matrix and the specified value.
template <class T> CMatrix<T> max(const CMatrix<T> &m, const T bound);
//!
template <class T> CMatrix<T> max(const CMatrix<T> &m1, const CMatrix<T> &m2);
//! @return For dim==0: a vector containing matrix summed over columns (@m is a matrix), dim==1 sums over rows.  Return value is a vector for matrix inputs otherwise a scalar.
template <class T> CMatrix<T> sum(const CMatrix<T> &m, const int dim = 0);
//! @return
template <class T> CMatrix<T> sumsquare(const CMatrix<T> &m, const int dim = 0);
//! @return
template <class T, class R> void sumsquare(const CMatrix<T> &m, CMatrix<R> &rv, const int dim = 0);
//! @return The matrix division of @c arg1 into @c arg2, e.g. @c inv(arg1)*arg2 (same as @c arg1\arg2 in Matlab) @sa solve()
template <class T> CMatrix<T> mldivide(const CMatrix<T> &arg1,
				       const CMatrix<T> &arg2);
//! @return The matrix division of @c arg2 into @c arg1, e.g. @c arg1*inv(arg2) (same as @c arg1/arg2 in Matlab) @sa solve()
template <class T> CMatrix<T> mrdivide(const CMatrix<T> &arg1,
				       const CMatrix<T> &arg2);
//! @return The vertical concatenation of the two specified matrices (same as @c [arg1;arg2] in Matlab).
template <class T> CMatrix<T> vertcat(const CMatrix<T> &arg1,
				      const CMatrix<T> &arg2);
//! Does the same as @c vertcat except the result goes back into @c arg1
template <class T> void vertcat2(CMatrix<T> &arg1, const CMatrix<T> &arg2);
//! Private implementation class.  Do not use.  Only declared because it needs friendship with CMatrix and you can't do that without a declaration....
template <class T> void vertcat_impl(const CMatrix<T> &, const CMatrix<T> &,
				     CMatrix<T> &, const unsigned);
//! @return The horizontal concatenation of the two specified matrices (same as @c [arg1 @c arg2] in Matlab).
template <class T> CMatrix<T> horzcat(const CMatrix<T> &arg1,
				      const CMatrix<T> &arg2);
//! Does the same as @c horzcat except the result goes back into @c arg1
template <class T> void horzcat2(CMatrix<T> &arg1, const CMatrix<T> &arg2);
//! Private implementation class.  Do not use.  Only declared because it needs friendship with CMatrix and you can't do that without a declaration....
template <class T> void horzcat_impl(const CMatrix<T> &, const CMatrix<T> &,
				     CMatrix<T> &, const unsigned);
//! @return The element-by-element multiplication of the two specified matrices (same as @c arg1.*arg2 in Matlab).
template <class T> CMatrix<T> times(const CMatrix<T> &arg1,
				    const CMatrix<T> &arg2);
//! @return The element-by-element multiplication of the contents of the specified matrix and the specified scalar.
template <class T> CMatrix<T> times(const T arg1, const CMatrix<T> &arg2);
//! @return The element-by-element multiplication of the contents of the specified matrix and the specified scalar.
template <class T> CMatrix<T> times(const CMatrix<T> &arg1, const T arg2);
//! @return The element-by-element division of the contents of the two specified matrices (same as @c arg1./arg2 in Matlab).
template <class T> CMatrix<T> rdivide(const CMatrix<T> &arg1,
				      const CMatrix<T> &arg2);
//! @return The division of the specified scalar by each element in the specified matrix (same as @c arg1./arg2 in Matlab).
template <class T> CMatrix<T> rdivide(const double arg1, 
				      const CMatrix<T> &arg2);
//! @return The element-by-element division of the contents of the specified matrix by the specified scalar (same as @c arg1./arg2 in Matlab).
template <class T> CMatrix<T> rdivide(const CMatrix<T> &arg1,
				      const double arg2);
/**
 * Returns solution to @c AX=B .  This is either done by inversion or least
 * squares depending on the relative sizes of the two matrices.
 * @param A is the matrix A in the above.
 * @param B is the matrix B in the above.
 * @param transA specifies whether the transpose of A should be used instead of A.  Note that this is on top of any transpose flag that is already inside @c A.
 * @param transB specifies whether the transpose of A should be used instead of B.  Note that this is on top of any transpose flag that is already inside @c B.
 */
template <class T> CMatrix<T> solve(const CMatrix<T> &A,
				    const CMatrix<T> &B, const bool transA,
				    const bool transB);
/**
 */
template <class T> CMatrix<T> solve_triangular(const CMatrix<T> &A,
					       const CMatrix<T> &B,
					       const bool uppertriangleA = true);
//! @return A vector of the singular values of the specified matrix.
template <class T> CMatrix<T> svd_vals(const CMatrix<T> &m);
//! @return The inverse of the specified matrix.
template <class T> CMatrix<T> inv(const CMatrix<T> &m);
//! @return A vector with the input vector in sorted order.  Note that unlike the Matlab function of the same name this function does not at this time support full matrices, only vectors.
template <class T> CMatrix<T> sort(const CMatrix<T> &vec, unsigned **index = NULL);
//! @return
template <class T> CMatrix<T> scale_columns(const CMatrix<T> &m, const CMatrix<T> &vec, const bool invert = false);
//! @return
template <class T> CMatrix<T> scale_rows(const CMatrix<T> &m, const CMatrix<T> &vec, const bool invert = false);
//!
template <class T> void whos(ostream &os, const char *name, const CMatrix<T> *m, ...);
//!
template <class T> CMatrix<T> sign(const CMatrix<T> &m);
//!
template <class T> CMatrix<T> rand(const unsigned nrows, const unsigned ncols);
//!
template <class T> CMatrix<T> sqrt(const CMatrix<T> &m);
//!
template <class T> bool has_infinity(const CMatrix<T> &m);

#ifdef __GNUG__
#define FRIEND_TEMPLATE <T>
#else
#define FRIEND_TEMPLATE /* nothing */
#endif

//! Matrix handle class
/**
 * This is the matrix class.  Application writers should use this class
 * exclusively; the CMatrixImpl class is private to this class and should
 * only be used if you are extending CMatrix.
 *
 * This class has been designed with the following things in mind :
 * -  Matrix data should be reference counted with copy-on-write semantics.
 * Thus we have a handle class (this class) and an implementation class
 * (CMatrixImpl) which actually contains the data.  Copying merely means having
 * two instances of this class with pointers to the same implementation
 * data.  When one object wants to change (i.e. write to) the data it then
 * makes its own copy of the implementation object to play with.
 * -  This class (and the implementation class) are templates.  As such we
 * can use the same code without source duplication for methods that work the
 * same way for single- or double-precision and we provide template
 * specializations for those which need different implementations for each
 * (e.g. ATLAS and LAPACK routines).
 * -  Transposes are handled without copying.  This handle class contains
 * a flag to say whether the implementation data should be interpreted as if
 * it is actually transposed.  Transposed implementations should be transparent
 * to the user.
 * -  Matlab-style submatrices are supported without unnecesary copying.
 * Matlab supports references to portions of a matrix via. "colon" notion,
 * e.g. @c foo(3:5,4:7) .  How this new matrix is treated in terms of
 * assignments is rather complicated.
 * Consider the Matlab expression @c foo(3:5,4:7)=bar .  The semantics of the
 * assignment statment here are different to a "normal" assignment in the 
 * sense that the values of @c bar are directly inserted into @c foo rather
 * than @c foo becoming in some way a "copy" of bar (which is what happens
 * in normal assignment).  Sub-matrix expressions thus need to be treated
 * differently depending on whether they are used on the left-hand-side or
 * right-hand-size of an '='.  This class deals with these issues in a manner
 * that is consistent with Matlab.
 *
 * It is recommended the copy constructor is not used, i.e.
 *
 * CMatrix<T> foo(bar);
 *
 * often doesn't do what you expect whereas
 *
 * CMatrix<T> foo;
 * foo = bar;
 *
 * does.
 *
 * Note that most functionality to be used by applications programmers is in
 * friend functions, not this class.
 */
template <class T>
class CMatrix
{
public:
  /**
   * Creates a new matrix of zero size.
   */
  CMatrix(void);
  /**
   * Creates a new matrix of the specified size (and with an optional
   * initializer).
   * @param nrows is the number of rows for the matrix.
   * @param ncols is the number of columns for the matrix.
   * @param init is an optional pointer to an array of values that should be
   * copied into the matrix (the values of @c init will appear in the matrix
   * in column order).
   * @param copydata specifies if the data should be copied into newly
   * allocated space or if it should just be installed "as-is".  Note that
   * if you use false here you must allocate your block with the same
   * allocator (malloc or new) that the matrix class was built with.
   */
  CMatrix(const unsigned nrows, const unsigned ncols, 
	  T *init = NULL, const bool copydata = true,
	  void (*freefunc)(void *) = NULL);
  /**
   */
  CMatrix(const unsigned nrows, const unsigned ncols, const T val);
  /**
   * Copy constructor (note that this doesn't copy the value of the matrix,
   * but rather increments the reference count in the corresponding 
   * implementation).  FIXME: I think this doco. is wrong about ref. count.
   * @param other is the matrix to copy.
   */
  CMatrix(const CMatrix<T> &other);
  /**
   * Create a submatrix from a region inside a scalar or vector.
   * @param start is the index of the first element in the region.
   * @param end is the index of the last element in the region.
   */
  CMatrix(const CMatrix<T> &other, unsigned start, unsigned end);
  /**
   * Create a submatrix from a region inside another matrix.
   * @param istart is the index of the first row in the region.
   * @param iend is the index of the last row in the region.
   * @param jstart is the index of the first column in the region.
   * @param jend is the index of the last column in the region.
   */
  CMatrix(const CMatrix<T> &other, unsigned istart, unsigned iend,
	  unsigned jstart, unsigned jend);
#ifdef HAVE_MATLAB
  /**
   * Creates a new matrix class whose implementation is a copy of the
   * supplied Matlab matrix.
   * @param is a Matlab matrix whose values are to be used in the new matrix.
   */
  CMatrix(mxArray *m);
#endif /* ! NO_MATLAB */
#ifdef HAVE_MPI
  /**
   * Create a matrix of the specified size and intialize with data retrieved
   * from MPI message.
   * @param nrows is the number of rows for the new matrix.
   * @param ncols is the number of columns for the new matrix.
   * @param source is passed to MPI_Recv.
   * @param tag is passed to MPI_Recv.
   * @param comm is passed to MPI_Recv.
   * @param status is passed to MPI_Recv.
   * @param ishomogeneous controls whether data is transmitted in machine-independent
   * form (@c false) or whether we assume we are using a homogeneous
   * network in which case format conversion does not need to be done.
   */
  CMatrix(const unsigned nrows, const unsigned ncols,
	  int source, int tag, MPI_Comm comm, MPI_Status *status,
	  const bool ishomogeneous = true);
#endif /* HAVE_MPI */
  /**
   * Create a matrix by reading raw data from a file.
   * @param nrows is the number of rows for the new matrix.
   * @param ncols is the number of columns for the new matrix.
   * @param filename is the path to the file to read the data from.
   * @param offset is the byte offset from where reading should start.
   * @param swapbytes specifies if the byte order of the data read should be changed.
   */
  CMatrix(const unsigned nrows, const unsigned ncols, const char *filename,
	  const long int offset, const bool swapbytes = false);
  /**
   * Destructor.  The matrix values (implementation) are destroyed only if
   * there are no other CMatrix objects with references to the same
   * implementation.
   */
  ~CMatrix(void);

  friend ostream &operator<< FRIEND_TEMPLATE (ostream &os, const CMatrix<T> &m);

  friend CMatrix<T> operator+ FRIEND_TEMPLATE (const CMatrix<T> &arg1, 
					       const CMatrix<T> &arg2);
  friend CMatrix<T> operator+ FRIEND_TEMPLATE (const CMatrix<T> &arg1,
					       const T arg2);
  friend CMatrix<T> operator- FRIEND_TEMPLATE (const CMatrix<T> &arg1, 
				 const CMatrix<T> &arg2);
  friend CMatrix<T> operator- FRIEND_TEMPLATE (const CMatrix<T> &arg1,
					       const T arg2);
  friend CMatrix<T> operator* FRIEND_TEMPLATE (const CMatrix<T> &arg1, 
					       const CMatrix<T> &arg2);
  friend CMatrix<T> operator* FRIEND_TEMPLATE (const T val,
					       const CMatrix<T> &m);
  friend CMatrix<T> operator* FRIEND_TEMPLATE (const CMatrix<T> &m,
					       const T val);
  friend CMatrix<T> operator/ FRIEND_TEMPLATE (const CMatrix<T> &arg1, 
				 const CMatrix<T> &arg2);
  friend CMatrix<T> operator/ FRIEND_TEMPLATE (const CMatrix<T> &m,
					       const double val);
  friend CMatrix<T> operator- FRIEND_TEMPLATE (const CMatrix<T> &m);

  /**
   * Matrix assignment.
   * @return A reference to the current matrix.
   */
  CMatrix<T> &operator=(const CMatrix<T> &other);
  /**
   * Matrix assignment to a scalar.
   * @return A reference to the current matrix.
   */
  CMatrix<T> &operator=(const T val);

  /**
   * Create a submatrix from a region inside this matrix (this form is designed for use on vectors only, hence only on CMatrixIndex object is needed).
   * @param idx specifies the range of values used to make up the submatrix.
   * @return A new matrix (submatrix) object.
   */
  CMatrix<T> operator()(const CMatrixIndex &idx) 
  { return (CMatrix(*this, idx.m_StartIdx, idx.m_EndIdx)); }
  CMatrix<T> operator()(const CMatrixIndex &idx) const 
  { return (CMatrix(*this, idx.m_StartIdx, idx.m_EndIdx)); }

  /**
   */
  inline T *GetDataPointer(void) {   
    Dirty(); 
    return (m_TopLeftPtr); 
  }

  /**
   * Create a submatrix from a region inside this matrix.
   * @param iidx specifies the range of rows used to make up the submatrix.   
   * @param jidx specifies the range of columns used to make up the submatrix.
   * @return A new matrix (submatrix) object.
   */
  CMatrix<T> operator()(const CMatrixIndex &iidx, const CMatrixIndex &jidx)
  { return (CMatrix<T>(*this, iidx.m_StartIdx, iidx.m_EndIdx, jidx.m_StartIdx,
		       jidx.m_EndIdx)); }
  CMatrix<T> operator()(const CMatrixIndex &iidx, 
			const CMatrixIndex &jidx) const
  { return (CMatrix(*this, iidx.m_StartIdx, iidx.m_EndIdx, jidx.m_StartIdx,
		    jidx.m_EndIdx)); }

  /**
   */
  inline T GetValue(const unsigned i) const {
    if (IsRowVector())
      return (*GetImplDataPtr(0, i));
    if (IsColumnVector())
      return (*GetImplDataPtr(i, 0));
    abort();
    return ((T)0); /* Compaq C++ doesn't realize abort() is the end */
  }
  /**
   * Returns the (scalar) value of a particular matrix element (equivalent to
   * @c operator()(colon(i),colon(j)) or @c foo(i,j) in Matlab).
   * @param i is the row index for the desired value.
   * @param j is the column index for the desired value.
   * @return The matrix value at the specified location.
   */
  inline T GetValue(const unsigned i, const unsigned j) const 
  { return (*GetImplDataPtr(i, j)); }
  /**
   */
  inline void SetValue(const unsigned i, const T val) {
    if (IsRowVector()) {
      Resize(1, MAX(i, GetNumColumns()));
      Dirty();
      *GetImplDataPtr(0, i) = val;
    } else if (IsColumnVector()) {
      Resize(MAX(i, GetNumRows()), 1);
      Dirty();
      *GetImplDataPtr(i, 0) = val;
    } else
      abort();
  }
  /**
   * Sets the value of a single element of the matrix (equivalent to
   * @c foo(i,j)=val in Matlab).
   * @param i is the row index for the value to be set.
   * @param j is the column index for the value to be set.
   * @param val is the new value for the specified matrix element.
   */
  inline void SetValue(const unsigned i, const unsigned j, const T val) { 
    Resize(MAX(i, GetNumRows()), MAX(j, GetNumColumns()));
    Dirty();
    *GetImplDataPtr(i, j) = val;
  }

  /**
   * Fill the matrix with random integer values in the specified range.
   * @param intmin is the minimum value to use.
   * @param intmax is the maximum value to use.
   */
  void Randomize(const int intmin = 0, const int intmax = 0);
  /**
   */
  void Randomize(const T minval, const T maxval);
  /**
   */
  void RandomizePM1(void);

  /**
   * Resize the matrix to the specified size and optionally initialize with
   * the specified array of values.  (This is usually used to set a matrix's
   * value to something useful after it was originally created with the
   * default constructor).
   * @param nrows is the number of rows the matrix should have.
   * @param ncols is the number of columns the matrix should have.
   * @param initvalues is an (optional) array of values that are copied into
   * the matrix column by column to initialize it.
   */
  void SetVals(const unsigned nrows, const unsigned ncols,
	       const T *initvals) {
    if (m_Impl)
      m_Impl->DeleteRef();
    m_Impl = new CMatrixImpl<T>(m_nRows = nrows, m_nCols = ncols);
    m_TopLeftPtr = m_Impl->m_Data;
    m_Transposed = false;
    memcpy(m_TopLeftPtr, initvals, nrows * ncols * sizeof(T));
  }

  /**
   */
  void SetAllValues(const T val) {
    Dirty();
    for (unsigned i = 0; i < GetNumElements(); ++i)
      m_TopLeftPtr[i] = val;
  }

  /**
   */
  void Resize(const unsigned nrows, const unsigned ncols);

  /**
   */
  template <class R>
  void ConvertFrom(const CMatrix<R> &m) {
    Resize(0, 0); /* nuke current data */
    Resize(m.GetNumRows(), m.GetNumColumns());
    Dirty();
    __matrix_const_iterator<R> src;
    __matrix_iterator<T> dest;
    unsigned i;
    for (i = 0, src = m.begin(), dest = begin(); i < GetNumElements();
	 ++i, ++src, ++dest)
      *dest = (T)*src;
  }

  void ReadFromFile(const char *path, const long offset,
		    const bool swapbytes = false);

#ifdef HAVE_MATLAB
  mxArray *ConvertToMatlabArray(const char *varname);
#endif /* HAVE_MATLAB */

  /**
   */
  void DeleteRows(const unsigned ndelrows, const unsigned *idxarr,
		  const bool invertset = false,
		  const bool forceimplresize = false);
  /**
   */
  void DeleteRowRange(const unsigned startrow, const unsigned endrow,
		      const bool forceimplresize = false);

  /**
   */
  void DeleteColumns(const unsigned ndelcols, const unsigned *idxarr,
		     const bool invertset = false,
		     const bool forceimplresize = false);

  /**
   */
  void DeleteColumnRange(const unsigned startcol, const unsigned endcol,
			 const bool forceimplresize = false);

  /**
   */
  CMatrix<T> ExtractRows(const unsigned ndelrows, const unsigned *idxarr,
			 const bool invertset = false,
			 const bool forceimplresize = false);

  /**
   */
  CMatrix<T> ExtractColumns(const unsigned ndelcols, const unsigned *idxarr,
			    const bool invertset = false,
			    const bool forceimplresize = false);

  /**
   */
  CMatrix<T> CopyRows(const unsigned nrows, const unsigned *idxarr) const;

  /**
   */
  CMatrix<T> CopyColumns(const unsigned ncols, const unsigned *idxarr) const;

  /**
   * WARNING: this routine has some sort of subtle bug, check any results you
   * get from it.
   */
  void gemm(const CMatrix<T> &a, const T alpha, const CMatrix<T> &b, 
	    const T beta);

#ifdef HAVE_MPI
  /**
   *
   */
  void MPI_Recv(int source, int tag, MPI_Comm comm, MPI_Status *status,
		const bool ishomogeneous = true);
  /**
   */
  void MPI_Send(int dest, int tag, MPI_Comm comm, 
		const bool ishomogeneous = true);
  /**
   */
  void MPI_Bcast(int root, MPI_Comm comm, const bool ishomogeneous = true);
  /**
   */
  void MPI_Reduce(int root, MPI_Comm comm, MPI_Op op);
  /**
   */
  void MPI_GatherRoot(int rank, MPI_Comm comm, const unsigned nexpectedvals,
		      const bool splitbyrows,
		      const bool ishomogeneous = true);
  /**
   */
  void MPI_GatherNode(int root, MPI_Comm comm, const bool splitbyrows, 
		      const bool ishomogeneous = true);

  /**
   */
  MPI_Datatype GetMPIDatatype(void);
#endif /* ! HAVE_MPI */

  //! @return The number of rows the matrix has.
  inline unsigned GetNumRows(void) const { return (m_Transposed?m_nCols:m_nRows); }
  //! @return The number of columns the matrix has.
  inline unsigned GetNumColumns(void) const { return (m_Transposed?m_nRows:m_nCols); }
  //! @return The number of elements the matrix has (rows * columns).
  inline unsigned GetNumElements(void) const { return (m_nCols * m_nRows); }
  //! @return @c true if this matrix and the matrix specified are the same size.
  inline bool IsSameSize(const CMatrix &other) const { return (((m_Transposed?m_nRows:m_nCols) == (other.m_Transposed?other.m_nRows:other.m_nCols)) && ((m_Transposed?m_nCols:m_nRows) == (other.m_Transposed?other.m_nCols:other.m_nRows))); }
  //! @return @c true if this matrix is square.
  inline bool IsSquareMatrix(void) const { return (m_nCols == m_nRows); }
  //! @return @c true if this matrix is actually a scalar (one row and one column).
  inline bool IsScalar(void) const { return ((m_nCols == 1) && (m_nRows == 1)); }
  //! @return @c true if this matrix is either a row or column vector.
  inline bool IsVector(void) const { return (IsRowVector() || IsColumnVector()); }
  //! @return @c true if this matrix is a row vector (one row only).
  inline bool IsRowVector(void) const { return ((m_Transposed?m_nCols:m_nRows) == 1); }
  //! @return @c true if this matrix is a column vector (one column only).
  inline bool IsColumnVector(void) const { return ((m_Transposed?m_nRows:m_nCols) == 1); }
  //! @return @c true if either the matrix is a scalar or it has more than one row @b and more than one column (i.e. is not a row or column vector).
  inline bool IsMatrix(void) const { return (IsScalar() || ((m_nRows > 1) && (m_nCols > 1))); }

  //! 
  void ProvideSizeHint(const unsigned long nelems) { if (m_Impl) m_Impl->ProvideSizeHint(nelems); else m_PendingSizeHint = nelems; }

  /**
   * This conversion operator only works if the matrix is a scalar.  It
   * can be used to get at the single value without having to use @c GetVal. 
   * @sa GetVal().
   */
  operator double(void) const { assert(IsScalar()); return (*m_TopLeftPtr); }
  /**
   * This conversion operator only works if the matrix is a scalar.  It
   * can be used to get at the single value without having to use @c GetVal. 
   * @sa GetVal().
   */
  operator float(void) const { assert(IsScalar()); return (*m_TopLeftPtr); }
  /**
   */
  operator unsigned char(void) const { assert(IsScalar()); return ((unsigned char)*m_TopLeftPtr); }

  typedef __matrix_iterator<T> iterator;
  typedef __matrix_const_iterator<T> const_iterator;

  /**
   * @return An iterator that traverses the matrix column by column.
   */
  iterator begin(const bool dirty = false) 
  { if (dirty) Dirty(); return (__matrix_iterator<T>(this, m_TopLeftPtr, true)); }
  /**
   * @return An iterator that traverses the matrix column by column (values cannot be changed through this iterator).
   */
  const_iterator begin() const
  { return (__matrix_const_iterator<T>(this, m_TopLeftPtr, true)); }
  /**
   * @return An iterator that traverses the matrix column by column (values cannot be changed through this iterator).
   */
  const_iterator const_begin() const
  { return (__matrix_const_iterator<T>(this, m_TopLeftPtr, true)); }
  /**
   * @return An iterator that traverses the matrix row by row.
   */
  iterator rbegin(const bool dirty = false) 
  { if (dirty) Dirty(); return (__matrix_iterator<T>(this, m_TopLeftPtr, false)); }
  /**
   * @return An iterator that traverses the matrix row by row (values cannot be changed through this iterator).
   */
  const_iterator rbegin() const 
  { return (__matrix_const_iterator<T>(this, m_TopLeftPtr, false)); }
  /**
   * @return An iterator that traverses the matrix row by row (values cannot be changed through this iterator).
   */
  const_iterator const_rbegin() const 
  { return (__matrix_const_iterator<T>(this, m_TopLeftPtr, false)); }

  friend CMatrix<T> ones FRIEND_TEMPLATE (const unsigned ndims);
  friend CMatrix<T> ones FRIEND_TEMPLATE (const unsigned nrows, const unsigned ncols);
  friend CMatrix<T> zeros FRIEND_TEMPLATE (const unsigned ndims);
  friend CMatrix<T> zeros FRIEND_TEMPLATE (const unsigned nrows, const unsigned ncols);
  friend CMatrix<T> eye FRIEND_TEMPLATE (const unsigned size);
  friend CMatrix<T> ctranspose FRIEND_TEMPLATE (const CMatrix<T> &m);
  /* mldivide: arg1 / arg2 (i.e. arg1 * inv(arg2)) */
  friend CMatrix<T> mldivide FRIEND_TEMPLATE (const CMatrix<T> &arg1, const CMatrix<T> &arg2);
  /* mrdivide: arg2 \ arg2 (i.e. inv(arg1) * arg2) */
  friend CMatrix<T> mrdivide FRIEND_TEMPLATE (const CMatrix<T> &arg1, const CMatrix<T> &arg2);
  /* rdivide : arg1 ./ arg2 */
  friend CMatrix<T> rdivide FRIEND_TEMPLATE (const CMatrix<T> &arg1, const CMatrix<T> &arg2);
  friend CMatrix<T> rdivide FRIEND_TEMPLATE (const double arg1, const CMatrix<T> &arg2);
  friend CMatrix<T> rdivide FRIEND_TEMPLATE (const CMatrix<T> &arg1, const double arg2);
  friend CMatrix<T> pow FRIEND_TEMPLATE (const CMatrix<T> &m, const T power);
  friend CMatrix<T> sort FRIEND_TEMPLATE (const CMatrix<T> &vec, unsigned **);
  friend CMatrix<T> sign FRIEND_TEMPLATE (const CMatrix<T> &);
  friend CMatrix<T> rand FRIEND_TEMPLATE (const unsigned, const unsigned);
  friend CMatrix<T> sqrt FRIEND_TEMPLATE (const CMatrix<T> &);

  /* times : arg1 .* arg2 */
  friend CMatrix<T> times FRIEND_TEMPLATE (const CMatrix<T> &arg1, const CMatrix<T> &arg2);
  friend CMatrix<T> times FRIEND_TEMPLATE (const T arg1, const CMatrix<T> &arg2);
  friend CMatrix<T> times FRIEND_TEMPLATE (const CMatrix<T> &arg1, const T arg2);
  /* vertcat : [arg1; arg2] */
  friend CMatrix<T> vertcat FRIEND_TEMPLATE (const CMatrix<T> &arg1, const CMatrix<T> &arg2);
  friend void vertcat2 FRIEND_TEMPLATE (CMatrix<T> &arg1, const CMatrix<T> &arg2);
  friend void vertcat_impl FRIEND_TEMPLATE (const CMatrix<T> &, const CMatrix<T> &, CMatrix<T> &, const unsigned);
  /* horzcat : [arg1 arg2] */
  friend CMatrix<T> horzcat FRIEND_TEMPLATE (const CMatrix<T> &arg1, const CMatrix<T> &arg2);
  friend void horzcat2 FRIEND_TEMPLATE (CMatrix<T> &arg1, const CMatrix<T> &arg2);
  friend void horzcat_impl FRIEND_TEMPLATE (const CMatrix<T> &, const CMatrix<T> &, CMatrix<T> &, const unsigned);
  friend CMatrix<T> diag FRIEND_TEMPLATE (const CMatrix<T> &m);
  friend CMatrix<T> chol FRIEND_TEMPLATE (const CMatrix<T> &m, const bool, const bool);
  friend CMatrix<T> abs FRIEND_TEMPLATE (const CMatrix<T> &m);
  friend double norm FRIEND_TEMPLATE (const CMatrix<T> &m);
  friend CMatrix<T> min FRIEND_TEMPLATE (const CMatrix<T> &m);
  friend CMatrix<T> min FRIEND_TEMPLATE (const CMatrix<T> &m, const T bound);
  friend CMatrix<T> min FRIEND_TEMPLATE (const CMatrix<T> &, const CMatrix<T> &);
  friend CMatrix<T> max FRIEND_TEMPLATE (const CMatrix<T> &m);
  friend CMatrix<T> max FRIEND_TEMPLATE (const CMatrix<T> &m, const T bound);
  friend CMatrix<T> max FRIEND_TEMPLATE (const CMatrix<T> &, const CMatrix<T> &);
  friend unsigned max_idx FRIEND_TEMPLATE (const CMatrix<T> &m);
  friend CMatrix<T> svd_vals FRIEND_TEMPLATE (const CMatrix<T> &m);
  friend CMatrix<T> inv FRIEND_TEMPLATE (const CMatrix<T> &m);
  friend CMatrix<T> scale_columns FRIEND_TEMPLATE (const CMatrix<T> &m, const CMatrix<T> &vec, const bool);
  friend CMatrix<T> scale_rows FRIEND_TEMPLATE (const CMatrix<T> &m, const CMatrix<T> &vec, const bool);

  friend bool operator== FRIEND_TEMPLATE (const CMatrix<T> &arg1, const CMatrix<T> &arg2);
  friend CMatrix<T> operator== FRIEND_TEMPLATE (const CMatrix<T> &arg1, const T arg2);
  friend bool operator!= FRIEND_TEMPLATE (const CMatrix<T> &arg1, const CMatrix<T> &arg2);

  friend bool has_infinity FRIEND_TEMPLATE (const CMatrix<T> &m);

  friend struct __matrix_iterator<T>;
  friend struct __matrix_const_iterator<T>;
private:
  friend CMatrix<T> solve FRIEND_TEMPLATE (const CMatrix<T> &A, 
					   const CMatrix<T> &B,
					   const bool transA, 
					   const bool transB);
  friend CMatrix<T> solve_triangular FRIEND_TEMPLATE (const CMatrix<T> &A,
						      const CMatrix<T> &B,
						      const bool uppertriangleA);

  inline unsigned GetStride(void) const { return (m_Impl->m_nRows); }
  void Dirty(void);

  inline T *GetImplDataPtr(const unsigned i, const unsigned j) {
    assert((i < GetNumRows()) && (j < GetNumColumns()));
    return (m_TopLeftPtr + (m_Transposed?i:j) * GetStride() + (m_Transposed?j:i)); 
  }
  inline const T *GetImplDataPtr(const unsigned i, const unsigned j) const { 
    return (m_TopLeftPtr + (m_Transposed?i:j) * GetStride() + (m_Transposed?j:i)); 
  }

  void blas_gemm(const int order, const int transA, const int transB, 
		 const int M, const int N, const int K,
		 const T alpha, const T *A, const int lda, const T *B, 
		 const int ldb, const T beta,
		 T *C, const int ldc) const;
  void blas_axpy(const int N, const T alpha, const T *X, const int incX,
		 T *Y, const int incY) const;
  void blas_scal(const int N, const T alpha, T *X, const int incX) const;
  void blas_copy(const int N, const T *X, const int incX, T *Y,
		 const int incY) const;
  void blas_trsm(const int order, const int side, const int uplo,
		 const int transA, const int diag, const int M, const int N,
		 const T alpha, const T *A, const int lda, T *B,
		 const int ldb) const;

  void lapack_potrf(const int upper, const int N, T *A, const int lda) const;
  /* IMPORTANT NOTE: the {d,s}getrf from ATLAS is *not* compatible with 
     {d,s}getri from LAPACK for computing inverses ! */
  void lapack_getrf(const int order, const int M, const int N, T *A, 
		    const int lda, int *piv,
		    const bool reallapack = false) const;
  void lapack_getrs(const int order, int trans, int N, int nrhs, const T *A,
		    const int lda, const int *piv, T *B, 
		    const int ldb) const;
  void lapack_gels(const int M, const int N, const int nrhs, T *A,
		   const int lda, T *B, const int ldb, T *work, 
		   int lwork) const;
  void lapack_gesvd(const char *jobu, const char *jobv, const int M,
		    const int N, T *A, const int lda, T *S, T *U,
		    const int ldu, T *vt, const int ldvt, T *work,
		    const int lwork) const;
  void lapack_getri(const int N, T *A, const int lda, const int *piv, 
		    T *work, const int lwork) const;

  void SetToZero(void) {
    Dirty();
    assert((m_Impl->m_nRefs == 1) && (m_TopLeftPtr == m_Impl->m_Data));
    memset(m_TopLeftPtr, 0, m_nRows * m_nCols * sizeof(T));
  }

  void SetToNull(void) {
    m_Impl->DeleteRef();
    m_Impl = NULL;
    m_TopLeftPtr = NULL;
    m_nRows = m_nCols = 0;
  }

  CMatrixImpl<T> *m_Impl;
  T *m_TopLeftPtr;
  unsigned m_nRows;
  unsigned m_nCols;
  bool m_Transposed;
  CMatrix<T> *m_SubMatrixParent;
  unsigned long m_PendingSizeHint;
};

/**
 * This is an iterator that can be used to traverse all elements in a matrix
 * in either row or column order.  Application writers should not need to
 * create instances of this directly, but rather they should use the begin(),
 * rbegin(), end() and rend() methods in CMatrix.
 *
 * Examples of use can be found in matrix.cpp.
 */
template <class T>
struct __matrix_iterator {
  typedef __matrix_iterator<T> iterator;
  typedef __matrix_const_iterator<T> const_iterator;

  unsigned i;
  unsigned j;
  CMatrix<T> *m;
  bool itcols; /* this is the natural direction for untransposed matrices */
  T *ptr;

  //! Default constructor; iterator is in unititialized state.
  __matrix_iterator() { }
  /**
   * Builds an iterator bound to the specified matrix
   * @param matrix is the matrix we want to iterate over.
   * @param ptrinit is a pointer to the first element of the matrix values
   * to be used.
   * @param itbycols if @c true denotes that the matrix should be iterated
   * over in column order instead of row order (note that the default is
   * row order which happens to be the opposite order that the elements are
   * actually stored in inside the implementation class).
   */
  __matrix_iterator(CMatrix<T> *matrix, T *ptrinit, bool itbycols) : 
    i(0), j(0), m(matrix), itcols(itbycols), ptr(ptrinit) { }
  /**
   * Deferencing oprerator
   * @return A reference to the value at the current iterator position.
   */
  T &operator*() const {
    assert((i < m->GetNumRows()) && (j < m->GetNumColumns()));
    assert(m->m_Impl->IsPtrInside(ptr)); 
    return (*ptr);
  }
  /**
   * Deferencing oprerator
   * @return A pointer to the value at the current iterator position.
   */
  T *operator->() const { return (ptr); }
  /**
   * Moves the iterator along to point to the next matrix element.
   * @return A reference to the iterator.
   */
  iterator &operator++() {
    if ((itcols && !m->m_Transposed) || (!itcols && m->m_Transposed))
      ++ptr;
    else
      ptr += m->GetStride();
    if (itcols) {
      if (++i >= m->GetNumRows()) {
	i = 0;
	++j;
	if (m->m_Transposed)
	  ptr = m->m_TopLeftPtr + j;
	else
	  ptr += (m->GetStride() - m->m_nRows);
      }
    } else {
      if (++j >= m->GetNumColumns()) {
	j = 0;
	++i;
	if (!m->m_Transposed)
	  ptr = m->m_TopLeftPtr + i;
	else
	  ptr += (m->GetStride() - m->m_nRows);
      }
    }
    assert(m->m_Impl->IsPtrInside(ptr) || (i >= m->GetNumRows()) ||
	   (j >= m->GetNumColumns()));
    return (*this);
  }

  //! @return @c true if the current @c iterator points to the same place as the specified iterator.
  inline bool operator==(const iterator &it) const { return (ptr == it.ptr); }
  //! @return @c true if the current @c const_iterator points to the same place as the specified iterator.
  inline bool operator==(const const_iterator &it) const { return (ptr == it.ptr); }
  //! @return @c true if the current @c iterator points to a different place as the specified iterator.
  inline bool operator!=(const iterator &it) const { return (ptr != it.ptr); }
  //! @return @c true if the current @c const_iterator points to a different place as the specified iterator.
  inline bool operator!=(const const_iterator &it) const { return (ptr != it.ptr); }

  //! @return Row index of current iterator position.
  inline unsigned I(void) const { return (i); }
  //! @return Column index of current iterator position.
  inline unsigned J(void) const { return (j); }
};

/**
 * This class is the same as __matrix_const_iterator except that no changes
 * to the matrix values can be made through it (it's @c const).
 */
template <class T>
struct __matrix_const_iterator {
  typedef __matrix_iterator<T> iterator;
  typedef __matrix_const_iterator<T> const_iterator;

  unsigned i;
  unsigned j;
  const CMatrix<T> *m;
  bool itcols; /* this is the natural direction for untransposed matrices */
  const T *ptr;

  //! Default constructor; iterator is in unititialized state.
  __matrix_const_iterator() { }
  /**
   * Builds an iterator bound to the specified matrix
   * @param matrix is the matrix we want to iterate over.
   * @param ptrinit is a pointer to the first element of the matrix values
   * to be used.
   * @param itbycols if @c true denotes that the matrix should be iterated
   * over in column order instead of row order (note that the default is
   * row order which happens to be the opposite order that the elements are
   * actually stored in inside the implementation class).
   */
  __matrix_const_iterator(const CMatrix<T> *matrix, const T *ptrinit,
			  const bool itbycols) : 
    i(0), j(0), m(matrix), itcols(itbycols), ptr(ptrinit) { }
  /**
   * Deferencing oprerator
   * @return A reference to the value at the current iterator position.
   */
  const T &operator*() const { 
    assert((i < m->GetNumRows()) && (j < m->GetNumColumns()));
    assert(m->m_Impl->IsPtrInside(ptr)); 
    return (*ptr);
  }
  /**
   * Deferencing oprerator
   * @return A pointer to the value at the current iterator position.
   */
  const T *operator->() const { return (ptr); }
  /**
   * Moves the iterator along to point to the next matrix element.
   * @return A reference to the iterator.
   */
  const_iterator &operator++() {
    if ((itcols && !m->m_Transposed) || (!itcols && m->m_Transposed))
      ++ptr;
    else
      ptr += m->GetStride();
    if (itcols) {
      if (++i >= m->GetNumRows()) {
	i = 0;
	++j;
	if (m->m_Transposed)
	  ptr = m->m_TopLeftPtr + j;
	else
	  ptr += (m->GetStride() - m->m_nRows);
      }
    } else {
      if (++j >= m->GetNumColumns()) {
	j = 0;
	++i;
	if (!m->m_Transposed)
	  ptr = m->m_TopLeftPtr + i;
	else {
	  ptr += (m->GetStride() - m->m_nRows);
	  assert(ptr == (m->m_TopLeftPtr + i * m->GetStride()));
	}
      }
    }
    assert(m->m_Impl->IsPtrInside(ptr) || (i >= m->GetNumRows()) ||
	   (j >= m->GetNumColumns()));
    return (*this);
  }

  //! @return @c true if the current @c iterator points to the same place as the specified iterator.
  inline bool operator==(const iterator &it) const { return (ptr == it.ptr); }
  //! @return @c true if the current @c const_iterator points to the same place as the specified iterator.
  inline bool operator==(const const_iterator &it) const { return (ptr == it.ptr); }
  //! @return @c true if the current @c iterator points to a different place as the specified iterator.
  inline bool operator!=(const iterator &it) const { return ptr != it.ptr; }
  //! @return @c true if the current @c const_iterator points to a different place as the specified iterator.
  inline bool operator!=(const const_iterator &it) const { return (ptr != it.ptr); }

  //! @return Row index of current iterator position.
  inline unsigned I(void) const { return (i); }
  //! @return Column index of current iterator position.
  inline unsigned J(void) const { return (j); }
};

//! Aborts with an error message if there are any matrix implementations outstanding (designed for debugging purposes when all handles have been freed).
void matrix_check_all_impl_freed(void);
//! @return the maximum number of bytes of data that has been simultaneously allocated for matrix data since the program was started.
//!
unsigned long matrix_get_max_data_bytes(void);
//!
unsigned long matrix_get_current_data_bytes(void);
//!
template <class T>
void matrix_set_cache_mgr(CBlockCache<T> *c);
//!
template <class T>
void matrix_add_cached_size(const unsigned numelems);
//!
template <class T>
unsigned matrix_get_kb_outstanding(void);
//!
template <class T>
unsigned matrix_get_max_kb_outstanding(void);
//!
template <class T>
void matrix_summarize_cached(ostream &);
//!
template <class T>
void matrix_flush_cache(void);

#ifdef HAVE_MATLAB
/**
 */
template <class T>
int matrix_read_from_mat_file(const char *matpath, ...);
/**
 */
template <class T>
int matrix_write_to_mat_file(const char *matpath, ...);

/**
 */
template <class T>
void matrix_compare_to_mat_file_var(const CMatrix<T> &m, const char *matpath,
				    const char *varname);
#endif /* HAVE_MATLAB */

/**
 */
template <class T>
void matrix_list_mem_usage(ostream &os, const char *varname,
			   const CMatrix<T> *m, ...);

#endif /* ! __MATRIX_H_ */
