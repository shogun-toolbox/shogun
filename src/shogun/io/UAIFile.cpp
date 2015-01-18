/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Abinash Panda
 */

#include <shogun/io/UAIFile.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;

CUAIFile::CUAIFile()
{
    init();
}

CUAIFile::CUAIFile(FILE* f, const char* name) :
    CFile(f, name)
{
    init();
    init_with_defaults();
}

CUAIFile::CUAIFile(int fd, const char* mode, const char* name) :
    CFile(fd, mode, name)
{
    init();
    init_with_defaults();
}

CUAIFile::CUAIFile(const char* fname, char rw, const char* name) :
    CFile(fname, rw, name)
{
    init();
    init_with_defaults();
}

CUAIFile::~CUAIFile()
{
    SG_UNREF(m_tokenizer);
    SG_UNREF(m_line_tokenizer);
    SG_UNREF(m_parser);
    SG_UNREF(m_line_reader);

    SG_FREE(m_factors_table);
    SG_FREE(m_factors_scope);
}

void CUAIFile::init()
{ 
    SG_ADD((CSGObject**)&m_line_reader, "line_reader", "line reader used to read lines from file", MS_NOT_AVAILABLE);
    SG_ADD((CSGObject**)&m_parser, "parser", "parser used to parse file", MS_NOT_AVAILABLE);
    SG_ADD((CSGObject**)&m_line_tokenizer, "line_tokenizer", "line tokenizer used to parse file", MS_NOT_AVAILABLE);
    SG_ADD((CSGObject**)&m_tokenizer, "tokenizer", "tokenizer used to parse file", MS_NOT_AVAILABLE);
    SG_ADD(&m_delimiter, "delimiter", "delimiter used in get_vector function", MS_NOT_AVAILABLE);
    
    SG_ADD(&m_num_vars, "num_vars", "number of variables", MS_NOT_AVAILABLE);
    SG_ADD(&m_num_factors, "num_factors", "number of factors", MS_NOT_AVAILABLE);
    SG_ADD(&m_net_type, "net_type", "network type (either BAYES or MARKOV)", MS_NOT_AVAILABLE);
    SG_ADD(&m_vars_card, "vars_card", "cardinality of all the variables", MS_NOT_AVAILABLE);

    /** Can only be enable after this issue is https://github.com/shogun-toolbox/shogun/issues/1972
     * resolved
     * SG_ADD(m_factors_table, "m_factors_table", "table of factors", MS_NOT_AVAILABLE);
     * SG_ADD(m_factors_scope, "m_factors_scope", "scope of factors", MS_NOT_AVAILABLE);
     */

    m_delimiter = ' ';
    m_tokenizer = NULL;
    m_line_tokenizer = NULL;
    m_parser = NULL;
    m_line_reader = NULL;

    m_num_vars = 0;
    m_num_factors = 0;
    m_factors_table = NULL;
    m_factors_scope = NULL;
}

void CUAIFile::init_with_defaults()
{
    m_delimiter=' ';
    
    m_tokenizer=new CDelimiterTokenizer(true);
    m_tokenizer->delimiters[m_delimiter]=1;
    SG_REF(m_tokenizer);

    m_line_tokenizer=new CDelimiterTokenizer(true);
    m_line_tokenizer->delimiters['\n']=1;
    SG_REF(m_line_tokenizer);

    m_parser=new CParser();
    m_parser->set_tokenizer(m_tokenizer);
    SG_REF(m_parser);

    m_line_reader=new CLineReader(file, m_line_tokenizer);
    SG_REF(m_line_reader);
}

#define GET_VECTOR(read_func, sg_type) \
void CUAIFile::get_vector(sg_type*& vector, int32_t& len) \
{ \
    if (!m_line_reader->has_next()) \
        return; \
    \
    SGVector<char> line; \
    int32_t num_elements = 0; \
    \
    line = m_line_reader->read_line(); \
    m_tokenizer->set_text(line); \
    while (m_tokenizer->has_next()) \
    { \
        int32_t temp_start; \
        m_tokenizer->next_token_idx(temp_start); \
        num_elements++; \
    } \
    \
    vector = SG_MALLOC(sg_type, num_elements); \
    m_parser->set_text(line); \
    for (int32_t i=0; i<num_elements; i++) \
        vector[i] = m_parser->read_func(); \
    len = num_elements; \
}

GET_VECTOR(read_char, int8_t)
GET_VECTOR(read_byte, uint8_t)
GET_VECTOR(read_char, char)
GET_VECTOR(read_int, int32_t)
GET_VECTOR(read_uint, uint32_t)
GET_VECTOR(read_short_real, float32_t)
GET_VECTOR(read_real, float64_t)
GET_VECTOR(read_long_real, floatmax_t)
GET_VECTOR(read_short, int16_t)
GET_VECTOR(read_word, uint16_t)
GET_VECTOR(read_long, int64_t)
GET_VECTOR(read_ulong, uint64_t)
#undef GET_VECTOR

#define SET_VECTOR(format, sg_type) \
void CUAIFile::set_vector(const sg_type* vector, int32_t len) \
{ \
    SG_SET_LOCALE_C; \
    \
    int32_t i; \
    for (i=0; i<len-1; i++) \
        fprintf(file, "%" format "%c", vector[i], m_delimiter); \
    fprintf(file, "%" format "\n", vector[i]); \
    \
    SG_RESET_LOCALE; \
}

SET_VECTOR(SCNi8, int8_t)
SET_VECTOR(SCNu8, uint8_t)
SET_VECTOR(SCNu8, char)
SET_VECTOR(SCNi32, int32_t)
SET_VECTOR(SCNu32, uint32_t)
SET_VECTOR(SCNi64, int64_t)
SET_VECTOR(SCNu64, uint64_t)
SET_VECTOR(".16g", float32_t)
SET_VECTOR(".16g", float64_t)
SET_VECTOR(".16Lg", floatmax_t)
SET_VECTOR(SCNi16, int16_t)
SET_VECTOR(SCNu16, uint16_t)
#undef SET_VECTOR

void CUAIFile::parse()
{
    if (!file)
        SG_SERROR("No file specified");

    SGVector<char> line, n_type;

    line = m_line_reader->read_line();
    m_parser->set_text(line);
    m_net_type = m_parser->read_string();
    
    line = m_line_reader->read_line();
    m_parser->set_text(line);
    m_num_vars = m_parser->read_int();

    get_vector(m_vars_card.vector, m_vars_card.vlen);
    
    line = m_line_reader->read_line();
    m_parser->set_text(line);
    m_num_factors = m_parser->read_int();

    m_factors_scope = new SGVector<int32_t> [m_num_factors];
    for (int32_t i=0; i<m_num_factors; i++)
    {
        int32_t num_elems;
        line = m_line_reader->read_line();
        m_parser->set_text(line);
        num_elems = m_parser->read_int();
        SGVector<int32_t> vars_index(num_elems);
        for (int32_t j=0; j<num_elems; j++)
            vars_index[j] = m_parser->read_int(); 
        m_factors_scope[i] = vars_index;
    }

    m_factors_table = new SGVector<float64_t> [m_num_factors];
    for (int32_t i=0; i<m_num_factors; i++)
    {
        int32_t data_size;
        line=m_line_reader->read_line();
        m_parser->set_text(line);
        data_size = m_parser->read_int();
        SGVector<float64_t> data;
        get_vector(data.vector, data.vlen);
        if (data_size != data.vlen)
            SG_SERROR("Data size mismatch. Expected %d size data; \
                got %d size data\n", data_size, data.vlen);
        m_factors_table[i] = data;
    }
}

void CUAIFile::set_net_type(const char* net_type)
{
    REQUIRE ((strncmp(net_type, "BAYES", 5) == 0 || strncmp(net_type, "MARKOV", 6) == 0),
        "Network type should be either MARKOV or BAYES");

    m_net_type = SGVector<char>(strlen(net_type));
    for (uint32_t i=0; i<strlen(net_type); i++)
        m_net_type[i] = net_type[i];

    fprintf(file, "%s\n", net_type);
}

void CUAIFile::set_num_vars(int32_t num_vars)
{
    m_num_vars = num_vars;
    fprintf(file, "%d\n", num_vars);
}

void CUAIFile::set_vars_card(SGVector<int32_t> vars_card)
{
    REQUIRE (m_num_vars == vars_card.vlen, 
        "Variables mismatch. Expected %d variables, got %d variables",
         m_num_vars, vars_card.vlen);

    m_vars_card = vars_card;
    set_vector(vars_card.vector, vars_card.vlen);
}

void CUAIFile::set_num_factors(int32_t num_factors)
{
    m_num_factors = num_factors;
    fprintf(file, "%d\n", num_factors);
}

void CUAIFile::set_factors_scope(int num_factors,
                                 const SGVector<int32_t>* factors_scope)
{
    REQUIRE(num_factors == m_num_factors, "Factors mismatch. Expected %d factors; \
        got %d factors", m_num_factors, num_factors)
    
    m_factors_scope = new SGVector<int32_t> [m_num_factors];
    for (int32_t i=0; i<m_num_factors; i++)
    {
        SGVector<int32_t> scope = factors_scope[i];
        m_factors_scope[i] = scope;
        fprintf(file, "%d ", scope.vlen);
        for (int32_t j=0; j<scope.vlen; j++)
            fprintf(file, "%d ", scope[j]);
        fprintf(file, "\n");
    }
}

void CUAIFile::set_factors_table(int32_t num_factors,
                                 const SGVector<float64_t>* factors_table)
{
    REQUIRE(num_factors == m_num_factors, "Factors mismatch. Expected %d factors; \
        got %d factors", m_num_factors, num_factors);

    m_factors_table = new SGVector<float64_t> [m_num_factors];
    for (int32_t i=0; i<m_num_factors; i++)
    {
        fprintf(file, "\n");
        SGVector<float64_t> data = factors_table[i];
        m_factors_table[i] = data;
        fprintf(file, "%d\n", data.size());
        set_vector(data.vector, data.vlen);
    }
}

void CUAIFile::get_preamble(SGVector<char>& net_type,
                            int32_t& num_vars,
                            SGVector<int32_t>& vars_card,
                            int32_t& num_factors,
                            SGVector<int32_t>*& factors_scope)
{
    net_type = m_net_type;
    num_vars = m_num_vars;
    vars_card = m_vars_card;
    num_factors = m_num_factors;
    
    factors_scope = new SGVector<int32_t> [m_num_factors];
    for (int32_t i=0; i<m_num_factors; i++)
        factors_scope[i] = m_factors_scope[i];
}

void CUAIFile::get_factors_table(SGVector<float64_t>*& factors_table)
{
    factors_table = new SGVector<float64_t> [m_num_factors];
    for (int32_t i=0; i<m_num_factors; i++)
        factors_table[i] = m_factors_table[i];
}

