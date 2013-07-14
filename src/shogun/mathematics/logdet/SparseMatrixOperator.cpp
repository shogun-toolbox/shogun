/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>

namespace shogun
{

template<class T, class ST>
CSparseMatrixOperator<T, ST>::CSparseMatrixOperator()
	: CMatrixOperator<T, ST>()
	{
		init();

		SG_SGCDEBUG("%s created (%p)\n", this->get_name(), this);
	}

template<class T, class ST>
CSparseMatrixOperator<T, ST>::CSparseMatrixOperator(SGSparseMatrix<T> op)
	: CMatrixOperator<T, ST>(op.num_features),
	  m_operator(op)
	{
		init();

		SG_SGCDEBUG("%s created (%p)\n", this->get_name(), this);
	}

template<class T, class ST>
void CSparseMatrixOperator<T, ST>::init()
	{
		CSGObject::set_generic<T>();

		CSGObject::m_parameters->add(&m_operator,
			"sparse_matrix",
			"The sparse matrix of the linear operator");
	}

template<class T, class ST>
CSparseMatrixOperator<T, ST>::~CSparseMatrixOperator()
	{
		SG_SGCDEBUG("%s destroyed (%p)\n", this->get_name(), this);
	}

template<class T, class ST>
SGSparseMatrix<T> CSparseMatrixOperator<T, ST>::get_matrix_operator() const
	{
		return m_operator;
	}

template<class T, class ST>
SGVector<T> CSparseMatrixOperator<T, ST>::get_diagonal() const
	{
		REQUIRE(m_operator.sparse_matrix, "Operator not initialized!\n");

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

template<class T, class ST>
void CSparseMatrixOperator<T, ST>::set_diagonal(SGVector<T> diag)
	{
		REQUIRE(m_operator.sparse_matrix, "Operator not initialized!\n");
		REQUIRE(diag.vector, "Diagonal not initialized!\n");

		const int32_t diag_size=m_operator.num_vectors>m_operator.num_features ?
			m_operator.num_features : m_operator.num_vectors;

		REQUIRE(diag_size==diag.vlen, "Dimension mismatch!\n");

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

template<class T, class ST>
SGVector<T> CSparseMatrixOperator<T, ST>::apply(SGVector<ST> b) const
	{
		REQUIRE(m_operator.sparse_matrix, "Operator not initialized!\n");
		REQUIRE(this->get_dimension()==b.vlen,
			"Number of rows of vector must be equal to the "
			"number of cols of the operator!\n");

		SGVector<T> result(m_operator.num_vectors);
		result=m_operator*b;

		return result;
	}

#define UNDEFINED(type) \
template<> \
SGVector<type> CSparseMatrixOperator<type, type>::apply(SGVector<type> b) const \
	{	\
		SG_SERROR("Not supported for %s\n", #type);\
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

template class CSparseMatrixOperator<bool>;
template class CSparseMatrixOperator<char>;
template class CSparseMatrixOperator<int8_t>;
template class CSparseMatrixOperator<uint8_t>;
template class CSparseMatrixOperator<int16_t>;
template class CSparseMatrixOperator<uint16_t>;
template class CSparseMatrixOperator<int32_t>;
template class CSparseMatrixOperator<uint32_t>;
template class CSparseMatrixOperator<int64_t>;
template class CSparseMatrixOperator<uint64_t>;
template class CSparseMatrixOperator<float32_t>;
template class CSparseMatrixOperator<float64_t>;
template class CSparseMatrixOperator<floatmax_t>;
template class CSparseMatrixOperator<complex64_t>;
template class CSparseMatrixOperator<complex64_t, int32_t>;
template class CSparseMatrixOperator<complex64_t, float64_t>;
}
