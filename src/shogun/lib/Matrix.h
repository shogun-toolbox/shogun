#ifndef MATRIX_H__
#define MATRIX_H__

#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

class Matrix {

	void *m_data;
	EPrimitiveType m_ptype;
	index_t m_rows, m_cols;

	template <typename T>
	static EPrimitiveType ptype();

public:
	template <typename T>
	Matrix(T *data, index_t rows, index_t cols)  :
		m_data(data), m_rows(rows), m_cols(cols)
	{
		m_ptype = ptype<T>();
	}

	template <typename T>
	operator SGMatrix<T>() const
	{
		return SGMatrix<T>(static_cast<T*>(m_data), m_rows, m_cols, false);
	}

	EPrimitiveType ptype() const
	{
		return m_ptype;
	}

	template <typename T>
	T* raw_data()
	{
		return static_cast<T*>(m_data);
	}
};

template <>
inline EPrimitiveType Matrix::ptype<float32_t>() { return PT_FLOAT32; }

template <>
inline EPrimitiveType Matrix::ptype<float64_t>() { return PT_FLOAT64; }

template <>
inline EPrimitiveType Matrix::ptype<floatmax_t>() { return PT_FLOATMAX; }

}

#endif // MATRIX_H_
