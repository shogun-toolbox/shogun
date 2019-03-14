/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Jacob Walker,
 *          Thoralf Klein, Soumyajit De, Yuyu Zhang, Evan Shelhamer
 */
#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <shogun/lib/config.h>

#include <shogun/base/DynArray.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/common.h>
#include <shogun/lib/type_case.h>
#include <type_traits>

namespace shogun
{

	class CSGObject;
	class CSerializableFile;
	template <class ST>
	class SGString;
	template <class T>
	class SGMatrix;
	template <class T>
	class SGSparseMatrix;
	template <class T>
	class SGVector;
	template <class T>
	class SGSparseVector;

	/** @brief parameter struct */
	struct TParameter
	{
		/** explicit constructor
		 * @param datatype datatype
		 * @param parameter pointer to parameter
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		explicit TParameter(
		    const TSGDataType* datatype, void* parameter, const char* name,
		    const char* description);

		/** destructor */
		~TParameter();

		/** print with prefix
		 * @param prefix prefix to print
		 */
		void print(const char* prefix);

		/** save to serializable file
		 * @param file destination file
		 * @param prefix prefix
		 */
		bool save(CSerializableFile* file, const char* prefix = "");

		/** load from serializable file
		 * @param file source file
		 * @param prefix prefix
		 */
		bool load(CSerializableFile* file, const char* prefix = "");

		/** operator for comparison, (by string m_name) */
		bool operator==(const TParameter& other) const;

		/** operator for comparison (by string m_name) */
		bool operator<(const TParameter& other) const;

		/** operator for comparison (by string m_name) */
		bool operator>(const TParameter& other) const;

		/** type of parameter */
		TSGDataType m_datatype;
		/** pointer to parameter */
		void* m_parameter;
		/** name of parameter */
		char* m_name;
		/** description of parameter */
		char* m_description;

		/** Incrementally get a hash from parameter value
		 *
		 * @param hash current hash value
		 * @param carry value for incremental murmur hashing
		 * @param total_length byte length of parameters. Function will
		 * add byte length to received value
		 *
		 */
		void get_incremental_hash(
		    uint32_t& hash, uint32_t& carry, uint32_t& total_length);

		/** test if parameter can be validly accessed, e.g., in case of a
		 * list/vector/matrix of objects the list/vector/matrix has non-zero
		 * length
		 */
		bool is_valid();

	private:
		char* new_prefix(const char* s1, const char* s2);
		void delete_cont();
		void new_cont(SGVector<index_t> dims);
		bool new_sgserial(
		    CSGObject** param, EPrimitiveType generic,
		    const char* sgserializable_name, const char* prefix);
		bool save_ptype(
		    CSerializableFile* file, const void* param, const char* prefix);
		bool
		load_ptype(CSerializableFile* file, void* param, const char* prefix);
		bool save_stype(
		    CSerializableFile* file, const void* param, const char* prefix);
		bool
		load_stype(CSerializableFile* file, void* param, const char* prefix);
	};

	/** @brief Parameter class
	 *
	 * Must not be an CSGObject to prevent a recursive call of
	 * constructors.
	 */
	class Parameter
	{
	public:
		/** explicit constructor */
		explicit Parameter();
		/** destructor */
		virtual ~Parameter();

		/** print
		 * @param prefix prefix
		 */
		virtual void print(const char* prefix = "");

		/** save to serializable file
		 * @param file destination file
		 * @param prefix prefix
		 */
		virtual bool save(CSerializableFile* file, const char* prefix = "");

		/** load from serializable file
		 * @param file source file
		 * @param prefix prefix
		 * */
		virtual bool load(CSerializableFile* file, const char* prefix = "");

		/** getter for number of parameters
		 * @return number of parameters
		 */
		virtual int32_t get_num_parameters()
		{
			return m_params.get_num_elements();
		}

		/** Takes another Parameter instance and sets all parameters of this
		 * instance (with equal name) to the values of the provided one.
		 * (Note that if CSGObjects are replaced, the old ones are SG_UNREFed
		 * and the new ones are SG_REFed)
		 * Currently only works for any float64_t and CSGObject type.
		 *
		 * @param params another Parameter instance
		 */
		void set_from_parameters(Parameter* params);

		/** Adds all parameters from another instance to this one
		 *
		 * @param params another Parameter instance
		 *
		 */
		void add_parameters(Parameter* params);

		/** Checks if a parameter with the spcified name is included
		 *
		 * @return true if parameter with name is included
		 */
		bool contains_parameter(const char* name);

		/** Getter for TParameter elements (Does not to bound checking)
		 *
		 * @param idx desired index
		 * @return pointer to the TParameter with the specified index
		 */
		inline TParameter* get_parameter(int32_t idx)
		{
			return m_params.get_element(idx);
		}

		/** Getter for Tparameter elements by name
		 *
		 * @param name name of desired parameter
		 * @return parameter with desired name, NULL if non such found
		 */
		inline TParameter* get_parameter(const char* name)
		{
			TParameter* result = NULL;

			for (index_t i = 0; i < m_params.get_num_elements(); ++i)
			{
				result = m_params.get_element(i);
				if (!strcmp(name, result->m_name))
					break;
				else
					result = NULL;
			}

			return result;
		}

		/* ************************************************************ */
		/* Scalar wrappers  */

		template <template <class> class CT, template <class> class ST, class T>
		std::enable_if_t<
		    type_internal::is_sg_container<CT, ST<T>>::value &&
		    type_internal::is_sg_struct<ST, T>::value>
		add(CT<ST<T>>* param, const char* name, const char* description = "")
		{
			EPrimitiveType ptype = _get_ptype<T>();
			EStructType stype = type_internal::sg_struct_type<ST, T>::stype;
			EContainerType ctype =
			    type_internal::sg_container_type<CT, T>::ctype;
			TSGDataType type = _create_tsg_type(ctype, stype, ptype, param);
			add_type(&type, param, name, description);
		}

		/** add param
		 * @param param parameter itself
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		template <template <class> class CT, class T>
		std::enable_if_t<type_internal::is_sg_container<CT, T>::value, void>
		add(CT<T>* param, const char* name, const char* description = "")
		{
			EPrimitiveType ptype = _get_ptype<T>();
			EContainerType ctype =
			    type_internal::sg_container_type<CT, T>::ctype;
			TSGDataType type = _create_tsg_type(ctype, ST_NONE, ptype, param);
			add_type(&type, param, name, description);
		}

		/** add param
		 * @param param parameter itself
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		template <template <class> class ST, class T>
		std::enable_if_t<type_internal::is_sg_struct<ST, T>::value, void>
		add(ST<T>* param, const char* name, const char* description = "")
		{
			EPrimitiveType ptype = _get_ptype<T>();
			EStructType stype = type_internal::sg_struct_type<ST, T>::stype;
			TSGDataType type(CT_SCALAR, stype, ptype);
			add_type(&type, param, name, description);
		}

		/** add param
		 * @param param parameter itself
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		template <typename T>
		void add(T* param, const char* name, const char* description = "")
		{
			EPrimitiveType ptype = _get_ptype<T>();
			TSGDataType type(CT_SCALAR, ST_NONE, ptype);
			add_type(&type, param, name, description);
		}

		/* ************************************************************ */
		/* Vector wrappers  */

		/** add param
		 * @param param parameter itself
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		template <template <class> class ST, class T>
		std::enable_if_t<type_internal::is_sg_struct<ST, T>::value, void>
		add_vector(
		    ST<T>** param, index_t* length, const char* name,
		    const char* description = "")
		{
			EPrimitiveType ptype = _get_ptype<T>();
			EStructType stype = type_internal::sg_struct_type<ST, T>::stype;
			TSGDataType type(CT_VECTOR, stype, ptype, length);
			add_type(&type, param, name, description);
		}

		/** add param
		 * @param param parameter itself
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		template <typename T>
		void add_vector(
		    T** param, index_t* length, const char* name,
		    const char* description = "")
		{
			EPrimitiveType ptype = _get_ptype<T>();
			TSGDataType type(CT_VECTOR, ST_NONE, ptype, length);
			add_type(&type, param, name, description);
		}

		/* ************************************************************ */
		/* Matrix wrappers  */

		/** add matrix param
		 * @param param parameter matrix itself
		 * @param length_y y size of matrix
		 * @param length_x x size of matrix
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		template <template <class> class ST, class T>
		std::enable_if_t<type_internal::is_sg_struct<ST, T>::value, void>
		add_matrix(
		    ST<T>** param, index_t* length_y, index_t* length_x,
		    const char* name, const char* description = "")
		{
			EPrimitiveType ptype = _get_ptype<T>();
			EStructType stype = type_internal::sg_struct_type<ST, T>::stype;
			TSGDataType type(CT_MATRIX, stype, ptype, length_y, length_x);
			add_type(&type, param, name, description);
		}

		/** add matrix param
		 * @param param parameter matrix itself
		 * @param length_y y size of matrix
		 * @param length_x x size of matrix
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		template <typename T>
		void add_matrix(
		    T** param, index_t* length_y, index_t* length_x, const char* name,
		    const char* description = "")
		{
			EPrimitiveType ptype = _get_ptype<T>();
			TSGDataType type(CT_MATRIX, ST_NONE, ptype, length_y, length_x);
			add_type(&type, param, name, description);
		}

	protected:
		/** array of parameters */
		DynArray<TParameter*> m_params;

		/** add new type
		 * @param type type to be added
		 * @param param pointer to parameter
		 * @param name name of parameter
		 * @param description description of parameter
		 */
		virtual void add_type(
		    const TSGDataType* type, void* param, const char* name,
		    const char* description);

	private:
		template <typename T>
		inline TSGDataType _create_tsg_type(
		    EContainerType ctype, EStructType stype, EPrimitiveType ptype,
		    SGVector<T>* container = nullptr)
		{
			return TSGDataType(ctype, stype, ptype, &container->vlen);
		}
		template <typename T>
		inline TSGDataType _create_tsg_type(
		    EContainerType ctype, EStructType stype, EPrimitiveType ptype,
		    SGMatrix<T>* container = nullptr)
		{
			return TSGDataType(
			    ctype, stype, ptype, &container->num_rows,
			    &container->num_cols);
		}

		template <typename T>
		std::enable_if_t<
		    type_internal::is_sg_primitive<T>::value, EPrimitiveType>
		_get_ptype()
		{
			return (EPrimitiveType)type_internal::sg_type<T>::ptype;
		}
		template <typename T>
		std::enable_if_t<
		    !type_internal::is_sg_primitive<T>::value, EPrimitiveType>
		_get_ptype()
		{
			return EPrimitiveType::PT_SGOBJECT;
		}
	};
} // namespace shogun
#endif //__PARAMETER_H__
