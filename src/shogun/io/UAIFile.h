/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Abinash Panda
 */

#ifndef __UAIFILE_H__
#define __UAIFILE_H__

#include <shogun/io/File.h>

#include <shogun/io/LineReader.h>
#include <shogun/io/Parser.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

/** @brief Class UAIFILE used to read data from UAI files.
 * See http://graphmod.ics.uci.edu/uai08/FileFormat for more
 * details.
 */
class CUAIFile : public CFile
{
public:
    /** default constructor */
    CUAIFile();

    /** constructor
     *
     * @param f already opened file
     * @param name variable name (e.g. "x" or "/path/to/x")
     */
    CUAIFile(FILE* f, const char* name=NULL);

    /** constructor
     *
     * @param fd already opened file descriptor
     * @param mode mode, 'r' or 'w'
     * @param name variable name (e.g. "x" or "/path/to/x")
     */
    CUAIFile(int fd, const char* mode, const char* name=NULL);

    /** constructor
     *
     * @param fname filename to open
     * @param rw mode, 'r' or 'w'
     * @param name variable name (e.g. "x" or "/path/to/x")
     */
    CUAIFile(const char* fname, char rw='r', const char* name=NULL);

    /** destructor */
    virtual ~CUAIFile();

    /** @name Parse Function
     *
     * Function to parse a given UAI file
     */
    virtual void parse();

    /** @name Network Type Access Function
     *
     * Function to set the network type
     * @param net_type type of network model (either MARKOV or BAYES)
     */
    virtual void set_net_type(const char* net_type);

    /** Function to set number of variables
     *
     * @param num_vars number of variables
     */
    virtual void set_num_vars(int32_t num_vars);

    /** Function to set number of factors
     *
     * @param num_vars number of factors
     */
    virtual void set_num_factors(int32_t num_vars);

    /** Function to set the cardinality of varibles
     *
     * @param vars_card vector of cardinality of variables
     */
    virtual void set_vars_card(SGVector<int32_t> vars_card);

    /** Function to set the scope of factors
     *
     * @param num_factors number of factors
     * @param factors_scope list of factors scope (SGVector<int32_t>)
     */
    virtual void set_factors_scope(int32_t num_factors,
                                   const SGVector<int32_t>* factors_scope);

    /** Function to set the table of factors
     *
     * @param num_factors number of factors
     * @param factors_table list of factors table (SGVector<float64_t>)
     */
    virtual void set_factors_table(int32_t num_factors,
                                   const SGVector<float64_t>* factors_table);

    /** Function to access preamble
     *
     * @param net_type type of network
     * @param num_vars number of variables
     * @param vars_card cardinality of variables
     * @param num_factors number of factors
     * @param factors_scope scope of all the factors
     */
    virtual void get_preamble(SGVector<char>& net_type,
                              int32_t& num_vars,
                              SGVector<int32_t>& vars_card,
                              int32_t& num_factors,
                              SGVector<int32_t>*& factors_scope);

    /** Function to access factor table
     * 
     * @param factors_table table of all the factors
     */
    virtual void get_factors_table(SGVector<float64_t>*& factors_table);

#ifndef SWIG // SWIG should skip this
    /** @name Vector Access Functions
     *
     * Functions to access vectors of one of the several base data types.
     * These functions are used when loading vectors from e.g. file
     * and return the vector and its length len by reference
     */
    //@{
    virtual void get_vector(int8_t*& vector, int32_t& len);
    virtual void get_vector(uint8_t*& vector, int32_t& len);
    virtual void get_vector(char*& vector, int32_t& len);
    virtual void get_vector(int32_t*& vector, int32_t& len);
    virtual void get_vector(uint32_t*& vector, int32_t& len);
    virtual void get_vector(float64_t*& vector, int32_t& len);
    virtual void get_vector(float32_t*& vector, int32_t& len);
    virtual void get_vector(floatmax_t*& vector, int32_t& len);
    virtual void get_vector(int16_t*& vector, int32_t& len);
    virtual void get_vector(uint16_t*& vector, int32_t& len);
    virtual void get_vector(int64_t*& vector, int32_t& len);
    virtual void get_vector(uint64_t*& vector, int32_t& len);
    //@}

    /** @name Matrix Access Functions
     *
     * Functions to access matrices of one of the several base data types.
     * These functions are used when loading matrices from e.g. file
     * and return the matrices and its dimensions num_feat and num_vec
     * by reference
     */
    //@{
    virtual void get_matrix(
            uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            int8_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            char*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            int32_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            uint32_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            int64_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            uint64_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            float32_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            float64_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            floatmax_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            int16_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_matrix(
            uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    //@}

    /** @name N-Dimensional Array Access Functions
     *
     * Functions to access n-dimensional arrays of one of the several base
     * data types. These functions are used when loading n-dimensional arrays
     * from e.g. file and return the them and its dimensions dims and num_dims
     * by reference
     */
    //@{
    virtual void get_ndarray(
            uint8_t*& array, int32_t*& dims, int32_t& num_dims) { };
    virtual void get_ndarray(
            char*& array, int32_t*& dims, int32_t& num_dims) { };
    virtual void get_ndarray(
            int32_t*& array, int32_t*& dims, int32_t& num_dims) { };
    virtual void get_ndarray(
            float32_t*& array, int32_t*& dims, int32_t& num_dims) { };
    virtual void get_ndarray(
            float64_t*& array, int32_t*& dims, int32_t& num_dims) { };
    virtual void get_ndarray(
            int16_t*& array, int32_t*& dims, int32_t& num_dims) { };
    virtual void get_ndarray(
            uint16_t*& array, int32_t*& dims, int32_t& num_dims) { };
    //@}

    /** @name Sparse Matrix Access Functions
     *
     * Functions to access sparse matrices of one of the several base data types.
     * These functions are used when loading sparse matrices from e.g. file
     * and return the sparse matrices and its dimensions num_feat and num_vec
     * by reference
     */
    //@{
    virtual void get_sparse_matrix(
            SGSparseVector<bool>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
        SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<uint32_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<int64_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<uint64_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec) { };
    virtual void get_sparse_matrix(
            SGSparseVector<floatmax_t>*& matrix, int32_t& num_feat, int32_t& num_vec)  { };
    //@}

    /** @name String Access Functions
     *
     * Functions to access strings of one of the several base data types.
     * These functions are used when loading variable length datatypes
     * from e.g. file and return the strings and their number
     * by reference
     */
    //@{
    virtual void get_string_list(
            SGString<uint8_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<int8_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<char>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<int32_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<uint32_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<int16_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<uint16_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<int64_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<uint64_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<float32_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<float64_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    virtual void get_string_list(
            SGString<floatmax_t>*& strings, int32_t& num_str,
            int32_t& max_string_len) { };
    //@}

    /** vector access functions */
    /*virtual void get_vector(void*& vector, int32_t& len, DataType& dtype);*/

    /** @name Vector Access Functions
     *
     * Functions to access vectors of one of the several base data types.
     * These functions are used when writing vectors of length len
     * to e.g. a file
     */
    //@{
    virtual void set_vector(const int8_t* vector, int32_t len);
    virtual void set_vector(const uint8_t* vector, int32_t len);
    virtual void set_vector(const char* vector, int32_t len);
    virtual void set_vector(const int32_t* vector, int32_t len);
    virtual void set_vector(const uint32_t* vector, int32_t len);
    virtual void set_vector(const float32_t* vector, int32_t len);
    virtual void set_vector(const float64_t* vector, int32_t len);
    virtual void set_vector(const floatmax_t* vector, int32_t len);
    virtual void set_vector(const int16_t* vector, int32_t len);
    virtual void set_vector(const uint16_t* vector, int32_t len);
    virtual void set_vector(const int64_t* vector, int32_t len);
    virtual void set_vector(const uint64_t* vector, int32_t len);
    //@}

    /** @name Matrix Access Functions
     *
     * Functions to access matrices of one of the several base data types.
     * These functions are used when writing matrices of num_feat rows and
     * num_vec columns to e.g. a file
     */
    //@{
    virtual void set_matrix(
            const uint8_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const int8_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const char* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const int32_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const uint32_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const int64_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const uint64_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const float32_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const float64_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const floatmax_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const int16_t* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_matrix(
            const uint16_t* matrix, int32_t num_feat, int32_t num_vec)  { };
    //@}

    /** @name Sparse Matrix Access Functions
     *
     * Functions to access sparse matrices of one of the several base data types.
     * These functions are used when writing sparse matrices of num_feat rows and
     * num_vec columns to e.g. a file
     */
    //@{
    virtual void set_sparse_matrix(
            const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    virtual void set_sparse_matrix(
            const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec) { };
    //@}

    /** @name String Access Functions
     *
     * Functions to access strings of one of the several base data types.
     * These functions are used when writing variable length datatypes
     * like strings to a file. Here num_str denotes the number of strings
     * and strings is a pointer to a string structure.
     */
    //@{
    virtual void set_string_list(
            const SGString<uint8_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<int8_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<char>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<int32_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<uint32_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<int16_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<uint16_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<int64_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<uint64_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<float32_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<float64_t>* strings, int32_t num_str) { };
    virtual void set_string_list(
            const SGString<floatmax_t>* strings, int32_t num_str) { };
    //@}
#endif // #ifndef SWIG // SWIG should skip this

    virtual const char* get_name() const { return "UAIFile"; }

private:
    /** class initialization */
    void init();

    /** class initialization */
    void init_with_defaults();

protected:
    /** object for reading lines from file */
    CLineReader* m_line_reader;

    /** parser of lines */
    CParser* m_parser;

    /** tokenizer for line_reader */
    CDelimiterTokenizer* m_line_tokenizer;

    /** tokenizer for parser */
    CDelimiterTokenizer* m_tokenizer;

    /** delimiter */
    char m_delimiter;

    /** number of variables */
    int32_t m_num_vars;

    /** number of factors */
    int32_t m_num_factors;

    /** type of network (either "MARKOV" or "BAYES") */
    SGVector<char> m_net_type;

    /** variable cardinality */
    SGVector<int32_t> m_vars_card;

    /** scope of all the factors */
    SGVector<int32_t>* m_factors_scope;

    /** data of all the factors */
    SGVector<float64_t>* m_factors_table;
};

}

#endif /** __UAIFILE_H__ */

