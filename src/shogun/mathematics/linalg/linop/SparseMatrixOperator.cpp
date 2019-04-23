/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/eigen3.h>

namespace shogun
{

template<class T>
SparseMatrixOperator<T>::SparseMatrixOperator()
	: MatrixOperator<T>()
	{
		init();
	}

template<class T>
SparseMatrixOperator<T>::SparseMatrixOperator(SGSparseMatrix<T> op)
	: MatrixOperator<T>(op.num_features),
	  m_operator(op)
	{
		init();
	}

template<class T>
SparseMatrixOperator<T>::SparseMatrixOperator
	(const SparseMatrixOperator<T>& orig)
	: MatrixOperator<T>(orig.get_dimension())
	{
		init();

		typedef SGSparseVector<T> vector;
		typedef SGSparseVectorEntry<T> entry;

		m_operator=SGSparseMatrix<T>(orig.m_operator.num_vectors, orig.m_operator.num_features);

		vector* rows=SG_MALLOC(vector, m_operator.num_features);
		for (index_t i=0; i<m_operator.num_vectors; ++i)
		{
			entry* features=SG_MALLOC(entry, orig.m_operator[i].num_feat_entries);
			for (index_t j=0; j<orig.m_operator[i].num_feat_entries; ++j)
			{
				features[j].feat_index=orig.m_operator[i].features[j].feat_index;
				features[j].entry=orig.m_operator[i].features[j].entry;
			}
			rows[i].features=features;
			rows[i].num_feat_entries=m_operator[i].num_feat_entries;
		}
		m_operator.sparse_matrix=rows;

		SG_TRACE("{} deep copy created ({})", this->get_name(), fmt::ptr(this));
	}

template<class T>
void SparseMatrixOperator<T>::init()
	{
		SGObject::set_generic<T>();
	}

template<class T>
SparseMatrixOperator<T>::~SparseMatrixOperator()
	{
	}

template<class T>
SGSparseMatrix<T> SparseMatrixOperator<T>::get_matrix_operator() const
	{
		return m_operator;
	}

template<class T>
SparsityStructure* SparseMatrixOperator<T>::get_sparsity_structure(
	int64_t power) const
	{
		require(power>0, "matrix-power is non-positive!");

		// create casted operator in bool for capturing the sparsity
		SparseMatrixOperator<bool>* sp_str
			=static_cast<SparseMatrixOperator<bool>*>(*this);

		// eigen3 map for this sparse matrix in which we compute current power
		Eigen::SparseMatrix<bool> current_power
			=EigenSparseUtil<bool>::toEigenSparse(sp_str->get_matrix_operator());

		// final power of the matrix goes into this one
		Eigen::SparseMatrix<bool> matrix_power;

		// compute matrix power with O(log(n)) matrix-multiplication!
		// traverse the bits of the power and compute the powers of 2 which
		// makes up this number. in the process multiply these to get the result
		bool lsb=true;
		while (power)
		{
			// if the current bit is on, it should contribute to the final result
			if (1 & power)
			{
				if (lsb)
				{
					// if seeing a 1 for the first time, then this should be the first
					// power we should consider
					matrix_power=current_power;
					lsb=false;
				}
				else
					matrix_power=matrix_power*current_power;
			}
			power=power>>1;

			// save unnecessary matrix-multiplication
			if (power)
				current_power=current_power*current_power;
		}

		// create the sparsity structure using the final power
		int32_t* outerIndexPtr=const_cast<int32_t*>(matrix_power.outerIndexPtr());
		int32_t* innerIndexPtr=const_cast<int32_t*>(matrix_power.innerIndexPtr());

		return new SparsityStructure(outerIndexPtr, innerIndexPtr,
			matrix_power.rows());
	}

template<> \
SparsityStructure* SparseMatrixOperator<complex128_t>
	::get_sparsity_structure(int64_t power) const
  {
    error("Not supported for complex128_t");
    return new SparsityStructure();
  }

template<class T>
SGVector<T> SparseMatrixOperator<T>::get_diagonal() const
	{
		require(m_operator.sparse_matrix, "Operator not initialized!");

		const int32_t diag_size=m_operator.num_vectors>m_operator.num_features ?
			m_operator.num_features : m_operator.num_vectors;

		SGVector<T> diag(diag_size);
		diag.set_const(static_cast<T>(0));
		for (index_t i=0; i<diag_size; ++i)
		{
			SGSparseVectorEntry<T>* current_row=m_operator[i].features;
			for (index_t j=0; j<m_operator[i].num_feat_entries; ++j)
			{
				if (i==current_row[j].feat_index)
				{
					diag[i]=current_row[j].entry;
					break;
				}
			}
		}

		return diag;
	}

template<class T>
void SparseMatrixOperator<T>::set_diagonal(SGVector<T> diag)
	{
		require(m_operator.sparse_matrix, "Operator not initialized!");
		require(diag.vector, "Diagonal not initialized!");

		const int32_t diag_size=m_operator.num_vectors>m_operator.num_features ?
			m_operator.num_features : m_operator.num_vectors;

		require(diag_size==diag.vlen, "Dimension mismatch!");

		bool need_sorting=false;
		for (index_t i=0; i<diag_size; ++i)
		{
			SGSparseVectorEntry<T>* current_row=m_operator[i].features;
			bool inserted=false;
			// we just change the entry if the diagonal element for this row exists
			for (index_t j=0; j<m_operator[i].num_feat_entries; ++j)
			{
				if (i==current_row[j].feat_index)
				{
					current_row[j].entry=diag[i];
					inserted=true;
					break;
				}
			}

			// we create a new entry if the diagonal element for this row doesn't exist
			if (!inserted)
			{
				index_t j=m_operator[i].num_feat_entries;
				m_operator[i].num_feat_entries=j+1;
				m_operator[i].features=SG_REALLOC(SGSparseVectorEntry<T>,
					m_operator[i].features, j, j+1);
				m_operator[i].features[j].feat_index=i;
				m_operator[i].features[j].entry=diag[i];
				need_sorting=true;
			}
		}

		if (need_sorting)
			m_operator.sort_features();
	}

template<class T>
SGVector<T> SparseMatrixOperator<T>::apply(SGVector<T> b) const
	{
		require(m_operator.sparse_matrix, "Operator not initialized!");
		require(this->get_dimension()==b.vlen,
			"Number of rows of vector must be equal to the "
			"number of cols of the operator!");

		SGVector<T> result(m_operator.num_vectors);
		result=m_operator*b;

		return result;
	}

#define UNDEFINED(type) \
template<> \
SGVector<type> SparseMatrixOperator<type>::apply(SGVector<type> b) const \
	{	\
		error("Not supported for {}", #type);\
		return b; \
	}

UNDEFINED(bool)
UNDEFINED(char)
UNDEFINED(int8_t)
UNDEFINED(uint8_t)
UNDEFINED(int16_t)
UNDEFINED(uint16_t)
UNDEFINED(int32_t)
UNDEFINED(uint32_t)
UNDEFINED(int64_t)
UNDEFINED(uint64_t)
UNDEFINED(float32_t)
UNDEFINED(floatmax_t)
#undef UNDEFINED

template class SparseMatrixOperator<bool>;
template class SparseMatrixOperator<char>;
template class SparseMatrixOperator<int8_t>;
template class SparseMatrixOperator<uint8_t>;
template class SparseMatrixOperator<int16_t>;
template class SparseMatrixOperator<uint16_t>;
template class SparseMatrixOperator<int32_t>;
template class SparseMatrixOperator<uint32_t>;
template class SparseMatrixOperator<int64_t>;
template class SparseMatrixOperator<uint64_t>;
template class SparseMatrixOperator<float32_t>;
template class SparseMatrixOperator<float64_t>;
template class SparseMatrixOperator<floatmax_t>;
template class SparseMatrixOperator<complex128_t>;
}
