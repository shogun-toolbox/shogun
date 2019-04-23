/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Abinash Panda, Abhinav Agarwalla, Bjoern Esser, 
 *          Sergey Lisitsyn
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
class UAIFile : public File
{
public:
    /** default constructor */
    UAIFile();

    /** constructor
     *
     * @param f already opened file
     * @param name variable name (e.g. "x" or "/path/to/x")
     */
    UAIFile(FILE* f, const char* name=NULL);

#ifdef HAVE_FDOPEN
    /** constructor
     *
     * @param fd already opened file descriptor
     * @param mode mode, 'r' or 'w'
     * @param name variable name (e.g. "x" or "/path/to/x")
     */
    UAIFile(int fd, const char* mode, const char* name=NULL);
#endif

    /** constructor
     *
     * @param fname filename to open
     * @param rw mode, 'r' or 'w'
     * @param name variable name (e.g. "x" or "/path/to/x")
     */
    UAIFile(const char* fname, char rw='r', const char* name=NULL);

    /** destructor */
    virtual ~UAIFile();

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

#endif // #ifndef SWIG // SWIG should skip this

    virtual const char* get_name() const { return "UAIFile"; }

private:
    /** class initialization */
    void init();

    /** class initialization */
    void init_with_defaults();

protected:
    /** object for reading lines from file */
    std::shared_ptr<LineReader> m_line_reader;

    /** parser of lines */
    std::shared_ptr<Parser> m_parser;

    /** tokenizer for line_reader */
    std::shared_ptr<DelimiterTokenizer> m_line_tokenizer;

    /** tokenizer for parser */
    std::shared_ptr<DelimiterTokenizer> m_tokenizer;

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

