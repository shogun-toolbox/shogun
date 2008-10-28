#include "lib/config.h"

#if defined(HAVE_CMDLINE)

#include "interface/CmdLineInterface.h"
#include "interface/SGInterface.h"

#include "lib/ShogunException.h"
#include "lib/io.h"
#include "lib/SimpleFile.h"

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

#ifndef WIN32
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

const int32_t READLINE_BUFFER_SIZE = 10000;
extern CSGInterface* interface;
extern CSGInterfaceMethod sg_methods[];

CCmdLineInterface::CCmdLineInterface()
: CSGInterface(), m_lhs(NULL), m_rhs(NULL)
{
	reset();
}

CCmdLineInterface::~CCmdLineInterface()
{
	delete m_rhs;
}

void CCmdLineInterface::reset(const char* line)
{
	CSGInterface::reset();

	if (!line)
		return;

	char* element=NULL;
	char delim_equal[]="=";
	char delim_lhs[]="=, \t\n";
	char delim_rhs[]=" \t\n";

	delete m_lhs;
	m_lhs=NULL;
	delete m_rhs;
	m_rhs=NULL;

	// split lhs from rhs
	char* equal_sign=strstr(line, delim_equal);
	if (equal_sign)
	//if (strstr(line, delim_equal))
	{
#ifdef DEBUG_CMDLINEIF
		SG_PRINT("has lhs!\n");
#endif
		element=strtok((char*) line, delim_lhs);
		if (element)
		{
			m_lhs=new CDynamicArray<char*>();
			m_lhs->append_element(element);
			m_nlhs++;
			while ((element=strtok(NULL, delim_lhs)))
			{
				if (element>equal_sign) // on rhs now
					break;

				m_lhs->append_element(element);
				m_nlhs++;
			}
		}
	}
	else
		element=strtok((char*) line, delim_rhs);

	if (element)
	{
		m_rhs=new CDynamicArray<char*>();
		m_rhs->append_element(element);
		m_nrhs++;
		while ((element=strtok(NULL, delim_rhs)))
		{
			m_rhs->append_element(element);
			m_nrhs++;
		}
	}
	else
		m_rhs=NULL;

#ifdef DEBUG_CMDLINEIF
	SG_PRINT("nlhs=%d nrhs=%d\n", m_nlhs, m_nrhs);
	if (m_lhs)
	{
		for (int32_t i=0; i<m_lhs->get_num_elements(); i++)
			SG_PRINT("element lhs %i %s\n", i, m_lhs->get_element(i));
	}

	if (m_rhs)
	{
		for (int32_t i=0; i<m_rhs->get_num_elements(); i++)
			SG_PRINT("element rhs %i %s\n", i, m_rhs->get_element(i));
	}
#endif
}


/** get functions - to pass data from the target interface to shogun */

/** determine argument type
 *
 * a signature is read from a data file. this signature contains the argument
 * type.
 *
 * currently, the signature must be in the beginning of the file,
 * consists of a line starting with 3 hash signs, 1 space and then
 * the argument type. e.g.:
 *
 * ### SHOGUN V0 STRING_CHAR
 * ACGTGCAAAAGC
 * AGTCDTCD
 */
IFType CCmdLineInterface::get_argument_type()
{
	const int32_t len=1024;
	IFType argtype=UNDEFINED;
	const char* filename=m_rhs->get_element(m_rhs_counter);

	// read the first 1024 of the file and heuristically decide about its
	// content
	FILE* fh=fopen((char*) filename, "r");
	if (!fh)
		SG_ERROR("Could not find file %s.\n", filename);

	char* chunk=new char[len+1];
	memset(chunk, 0, sizeof(char)*(len+1));
	size_t nread=fread(chunk, sizeof(char), len, fh);
	fclose(fh);
	if (nread<=0)
		SG_ERROR("Could not read data from %s.\n");

	char* signature=new char[len+1];
	int32_t num=sscanf(chunk, "### SHOGUN V0 %s\n", signature);

	// if file has valid shogun signature use it to determine file type
	if (num==1)
	{
		SG_DEBUG("read signature: %s\n", signature);

		if (strncmp(signature, "STRING_CHAR", 11)==0)
			argtype=STRING_CHAR;
		else if (strncmp(signature, "STRING_BYTE", 11)==0)
			argtype=STRING_BYTE;
		else if (strncmp(signature, "DENSE_REAL", 10)==0)
			argtype=DENSE_REAL;
		else if (strncmp(signature, "SPARSE_REAL", 11)==0)
			argtype=SPARSE_REAL;
	}
	else
	{
		SG_DEBUG("could not find signature in file %s guessing file type.\n", filename);

		// special for cubes
		if (strspn(chunk, "123456\n")==nread)
		{
			argtype=STRING_CHAR;
			SG_DEBUG("guessing STRING_CHAR\n");
		}
		else if (strspn(chunk, "0123456789.e+- \t\n")==nread)
		{
			argtype=DENSE_REAL;
			SG_DEBUG("guessing DENSE_REAL\n");
		}
		else if (strspn(chunk, "0123456789:.e+- \t\n")==nread)
		{
			argtype=SPARSE_REAL;
			SG_DEBUG("guessing SPARSE_REAL\n");
		}
		else
		{
			argtype=STRING_CHAR;
			SG_DEBUG("guessing STRING_CHAR\n");
		}
	}

	delete[] signature;
	delete[] chunk;
	return argtype;
}


int32_t CCmdLineInterface::get_int()
{
	const char* i=get_arg_increment();
	if (!i)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	int32_t value=-1;
	int32_t num=sscanf(i, "%d", &value);
	if (num!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	return value;
}

float64_t CCmdLineInterface::get_real()
{
	const char* r=get_arg_increment();
	if (!r)
		SG_ERROR("Expected Scalar Real as argument %d\n", m_rhs_counter);

	float64_t value=-1;
	int32_t num=sscanf(r, "%lf", &value);
	if (num!=1)
		SG_ERROR("Expected Scalar Real as argument %d\n", m_rhs_counter);

	return value;
}

bool CCmdLineInterface::get_bool()
{
	const char* b=get_arg_increment();
	if (!b)
		SG_ERROR("Expected Scalar Bool as argument %d\n", m_rhs_counter);

	int32_t value=-1;
	int32_t num=sscanf(b, "%i", &value);
	if (num!=1)
		SG_ERROR("Expected Scalar Bool as argument %d\n", m_rhs_counter);

	return (value!=0);
}


char* CCmdLineInterface::get_string(int32_t& len)
{
	const char* s=get_arg_increment();
	if (!s)
		SG_ERROR("Expected 1 String as argument %d.\n", m_rhs_counter);

	len=strlen(s);
	ASSERT(len>0);

	char* result=new char[len+1];
	memcpy(result, s, len*sizeof(char));
	result[len]='\0';

	return result;
}

void CCmdLineInterface::get_byte_vector(uint8_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CCmdLineInterface::get_char_vector(char*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CCmdLineInterface::get_int_vector(int32_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
/*
	vec=NULL;
	len=0;

	void* rvec=CAR(get_arg_increment());
	if( TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Integer Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=new int32_t[len];
	ASSERT(vec);

	for (int32_t i=0; i<len; i++)
		vec[i]= (int32_t) INTEGER(rvec)[i];
		*/
}

void CCmdLineInterface::get_shortreal_vector(float32_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CCmdLineInterface::get_real_vector(float64_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;

	const char* filename=get_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to read REAL matrix.\n");

	CFile f((char*) filename, 'r', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to read REAL matrix.\n", filename);

	int32_t num_feat=0;
	int32_t num_vec=0;

	if (!f.read_real_valued_dense(vec, num_feat, num_vec))
		SG_ERROR("Could not read REAL data from %s.\n", filename);

	if ((num_feat==1) || (num_vec==1))
	{
		if (num_feat==1)
			len=num_vec;
		else
			len=num_feat;
	}
	else
	{
		delete[] vec;
		vec=NULL;
		len=0;
		SG_ERROR("Could not read REAL vector from file %s (shape %dx%d found but vector expected).\n", filename, num_vec, num_feat);
	}

}

void CCmdLineInterface::get_short_vector(int16_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CCmdLineInterface::get_word_vector(uint16_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}


void CCmdLineInterface::get_byte_matrix(uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_char_matrix(char*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_int_matrix(int32_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_shortreal_matrix(float32_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_real_matrix(float64_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	const char* filename=get_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to read REAL matrix.\n");

	CFile f((char*) filename, 'r', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to read REAL matrix.\n", filename);

	if (!f.read_real_valued_dense(matrix, num_feat, num_vec))
		SG_ERROR("Could not read REAL data from %s.\n", filename);

	CMath::transpose_matrix(matrix, num_feat, num_vec);
}

void CCmdLineInterface::get_short_matrix(int16_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_word_matrix(uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_byte_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_char_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_int_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_shortreal_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_real_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_short_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_word_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_real_sparsematrix(TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	const char* filename=get_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to read SPARSE REAL matrix.\n");

	CFile f((char*) filename, 'r', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to read SPARSE REAL matrix.\n", filename);

	if (!f.read_real_valued_sparse(matrix, num_feat, num_vec))
		SG_ERROR("Could not read SPARSE REAL data from %s.\n", filename);
}


void CCmdLineInterface::get_byte_string_list(T_STRING<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_char_string_list(T_STRING<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	const char* filename=get_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to read CHAR string list.\n");

	CFile f((char*) filename, 'r', F_CHAR);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to read CHAR string list.\n", filename);

	if (!f.read_char_valued_strings(strings, num_str, max_string_len))
		SG_ERROR("Could not read CHAR data from %s.\n", filename);

/*
	for (int32_t i=0; i<num_str; i++)
		SG_PRINT("%s\n", strings[i].string);
*/
}

void CCmdLineInterface::get_int_string_list(T_STRING<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_short_string_list(T_STRING<int16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_word_string_list(T_STRING<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

/** set functions - to pass data from shogun to the target interface */
bool CCmdLineInterface::create_return_values(int32_t num)
{
	if (num==m_nlhs)
		return true;

	return false;
}

void* CCmdLineInterface::get_return_values()
{
	return NULL;
}


/** set functions - to pass data from shogun to the target interface */

void CCmdLineInterface::set_int(int32_t scalar)
{
	//set_arg_increment(ScalarInteger(scalar));
}

void CCmdLineInterface::set_real(float64_t scalar)
{
	//set_arg_increment(ScalarReal(scalar));
}

void CCmdLineInterface::set_bool(bool scalar)
{
	//set_arg_increment(ScalarLogical(scalar));
}


void CCmdLineInterface::set_char_vector(const char* vec, int32_t len)
{
}

void CCmdLineInterface::set_short_vector(const int16_t* vec, int32_t len)
{
}

void CCmdLineInterface::set_byte_vector(const uint8_t* vec, int32_t len)
{
}

void CCmdLineInterface::set_int_vector(const int32_t* vec, int32_t len)
{
}

void CCmdLineInterface::set_shortreal_vector(const float32_t* vec, int32_t len)
{
}

void CCmdLineInterface::set_real_vector(const float64_t* vec, int32_t len)
{
	const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write REAL vector.\n");

	CFile f((char*) filename, 'w', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to write REAL vector.\n", filename);

	if (!f.write_real_valued_dense(vec, len, 1))
		SG_ERROR("Could not write REAL data to %s.\n", filename);
}

void CCmdLineInterface::set_word_vector(const uint16_t* vec, int32_t len)
{
}

/*
#undef SET_VECTOR
#define SET_VECTOR(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CCmdLineInterface::function_name(const sg_type* vec, int32_t len)	\
{																\
	void* feat=NULL;												\
	PROTECT( feat = allocVector(r_type, len) );					\
																\
	for (int32_t i=0; i<len; i++)									\
		r_cast(feat)[i]=(if_type) vec[i];						\
																\
	UNPROTECT(1);												\
	set_arg_increment(feat);									\
}

SET_VECTOR(set_byte_vector, INTSXP, INTEGER, uint8_t, int, "Byte")
SET_VECTOR(set_int_vector, INTSXP, INTEGER, int32_t, int, "Integer")
SET_VECTOR(set_short_vector, INTSXP, INTEGER, int16_t, int, "Short")
SET_VECTOR(set_shortreal_vector, XP, REAL, float32_t, float, "Single Precision")
SET_VECTOR(set_real_vector, XP, REAL, float64_t, double, "Double Precision")
SET_VECTOR(set_word_vector, INTSXP, INTEGER, uint16_t, int, "Word")
#undef SET_VECTOR
*/


void CCmdLineInterface::set_char_matrix(const char* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CCmdLineInterface::set_byte_matrix(const uint8_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CCmdLineInterface::set_int_matrix(const int32_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CCmdLineInterface::set_short_matrix(const int16_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CCmdLineInterface::set_shortreal_matrix(const float32_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CCmdLineInterface::set_real_matrix(const float64_t* matrix, int32_t num_feat, int32_t num_vec)
{
	const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write REAL matrix.\n");

	CFile f((char*) filename, 'w', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to write REAL matrix.\n", filename);

	if (!f.write_real_valued_dense(matrix, num_feat, num_vec))
		SG_ERROR("Could not write REAL data to %s.\n", filename);
}
void CCmdLineInterface::set_word_matrix(const uint16_t* matrix, int32_t num_feat, int32_t num_vec)
{
}

/*
#define SET_MATRIX(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CCmdLineInterface::function_name(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{																			\
	void* feat=NULL;															\
	PROTECT( feat = allocMatrix(r_type, num_feat, num_vec) );				\
																			\
	for (int32_t i=0; i<num_vec; i++)											\
	{																		\
		for (int32_t j=0; j<num_feat; j++)										\
			r_cast(feat)[i*num_feat+j]=(if_type) matrix[i*num_feat+j];		\
	}																		\
																			\
	UNPROTECT(1);															\
	set_arg_increment(feat);												\
}
SET_MATRIX(set_byte_matrix, INTSXP, INTEGER, uint8_t, int, "Byte")
SET_MATRIX(set_int_matrix, INTSXP, INTEGER, int32_t, int, "Integer")
SET_MATRIX(set_short_matrix, INTSXP, INTEGER, int16_t, int, "Short")
SET_MATRIX(set_shortreal_matrix, XP, REAL, float32_t, float, "Single Precision")
SET_MATRIX(set_real_matrix, XP, REAL, float64_t, double, "Double Precision")
SET_MATRIX(set_word_matrix, INTSXP, INTEGER, uint16_t, int, "Word")
#undef SET_MATRIX
*/


void CCmdLineInterface::set_real_sparsematrix(const TSparse<float64_t>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz)
{
	const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write SPARSE REAL matrix.\n");

	CFile f((char*) filename, 'w', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to write SPARSE REAL matrix.\n", filename);

	if (!f.write_real_valued_sparse(matrix, num_feat, num_vec))
		SG_ERROR("Could not write SPARSE REAL data to %s.\n", filename);
}

void CCmdLineInterface::set_byte_string_list(const T_STRING<uint8_t>* strings, int32_t num_str)
{
}

void CCmdLineInterface::set_char_string_list(const T_STRING<char>* strings, int32_t num_str)
{
	const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write CHAR string list.\n");

	CFile f((char*) filename, 'w', F_CHAR);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to write CHAR string list.\n", filename);

	if (!f.write_char_valued_strings(strings, num_str))
		SG_ERROR("Could not write CHAR data to %s.\n", filename);
}

void CCmdLineInterface::set_int_string_list(const T_STRING<int32_t>* strings, int32_t num_str)
{
}

void CCmdLineInterface::set_short_string_list(const T_STRING<int16_t>* strings, int32_t num_str)
{
}

void CCmdLineInterface::set_word_string_list(const T_STRING<uint16_t>* strings, int32_t num_str)
{
}


bool CCmdLineInterface::skip_line(const char* line)
{
	if (!line)
		return true;

	if (line[0]=='\n' ||
		(line[0]=='\r' && line[1]=='\n') ||
		int(line[0])==0)
	{
		return true;
	}

	//SG_PRINT("ascii(0) %d, %c\n", int(line[0]), line[0]);

	char* skipped=CIO::skip_blanks((char*) line);
	if (skipped[0]==CMDLINE_COMMENT0 || skipped[0]==CMDLINE_COMMENT1)
		return true;

	return false;
}

void CCmdLineInterface::print_prompt()
{
	SG_PRINT( "\033[1;34mshogun\033[0m >> ");
	//SG_PRINT("shogun >> ");
}


char* CCmdLineInterface::get_line(FILE* infile, bool interactive_mode)
{
	char* in=NULL;
	memset(input, 0, sizeof(input));

	if (feof(infile))
		return NULL;

#ifdef HAVE_READLINE
	if (interactive_mode)
	{
		in=readline("\033[1;34mshogun\033[0m >> ");
		if (in)
		{
			strncpy(input, in, sizeof(input));
			add_history(in);
			free(in);
		}
	}
	else
	{
		if (fgets(input, sizeof(input), infile)==NULL)
			return NULL;
		in=input;
	}
#else
	if (interactive_mode)
		print_prompt();
	if (fgets(input, sizeof(input), infile)==NULL)
		return NULL;
	in=input;
#endif

	if (in==NULL)
		return NULL;
	else
		return input;
}

bool CCmdLineInterface::parse_line(char* line)
{
	if (!line)
		return false;
	
	if (skip_line(line))
		return true;
	else
	{
		((CCmdLineInterface*) interface)->reset(line);
		return interface->handle();
	}
}

#ifdef HAVE_READLINE
char* command_generator(const char *text, int state)
{
	static int list_index, len;
	const char *name;

	/* If this is a new word to complete, initialize now.  This
	 *      includes saving the length of TEXT for efficiency, and
	 *           initializing the index variable to 0. */
	if (!state)
	{
		list_index = 0;
		len = strlen (text);
	}

	/* Return the next name which partially matches from the
	 *      command list. */
	while ((name = sg_methods[list_index].command))
	{
		list_index++;

		if (strncmp (name, text, len) == 0)
			return (strdup(name));
	}

	/* If no names matched, then return NULL. */
	return NULL;
}

/* Attempt to complete on the contents of TEXT.  START and END
 * bound the region of rl_line_buffer that contains the word to
 * complete.  TEXT is the word to complete.  We can use the entire
 * contents of rl_line_buffer in case we want to do some simple
 * parsing.  Returnthe array of matches, or NULL if there aren't
 * any. */
char** shogun_completion (const char *text, int start, int end)
{
	char **matches;

	matches = (char **)NULL;

	/* If this word is at the start of the line, then it is a command
	 *      to complete.  Otherwise it is the name of a file in the
	 *      current
	 *           directory. */
	if (start == 0)
		matches = rl_completion_matches (text, command_generator);

	return (matches);
}
#endif //HAVE_READLINE

int main(int argc, char* argv[])
{	
#ifdef HAVE_READLINE
	rl_readline_name = "shogun";
	rl_attempted_completion_function = shogun_completion;
#endif //HAVE_READLINE

	try
	{
		interface=new CCmdLineInterface();
		CCmdLineInterface* intf=(CCmdLineInterface*) interface;

		// interactive
		if (argc<=1)
		{
			while (true)
			{
				char* line=intf->get_line();

				if (!line)
					break;

				try
				{
					intf->parse_line(line);
				}
				catch (ShogunException e) { }

			}

			delete interface;
			return 0;
		}

		// help
		if ( argc>2 || ((argc==2) && 
					( !strcmp(argv[1], "-h") || !strcmp(argv[1], "/?") || !strcmp(argv[1], "--help")) )
		   )
		{
			SG_SPRINT("\n\n");
			SG_SPRINT("usage: shogun [ -h | --help | /? | -i | filename ]\n\n");
			SG_SPRINT("if no options are given shogun enters interactive mode\n");
			SG_SPRINT("if filename is specified the commands will be read and executed from file\n");
			SG_SPRINT("if -i is specified shogun will listen on port 7367 from file\n");
			SG_SPRINT("==hex(sg), *dangerous* as commands from any source are accepted\n\n");

			delete interface;
			return 1;
		}

#ifndef CYGWIN
		// from tcp
		if ( argc==2 && !strcmp(argv[1], "-i"))
		{
			int s=socket(AF_INET, SOCK_STREAM, 0);
			struct sockaddr_in sa;
			sa.sin_family=AF_INET;
			sa.sin_port=htons(7367);
			sa.sin_addr.s_addr=INADDR_ANY;
			bzero(&(sa.sin_zero), 8);

			bind(s, (sockaddr*) (&sa), sizeof(sockaddr_in));
			listen(s, 1);
			int s2=accept(s, NULL, NULL);
			SG_SINFO( "accepting connection\n");

			char input[READLINE_BUFFER_SIZE];
			do
			{
				bzero(input, sizeof(input));
				int length=read(s2, input, sizeof(input));
				if (length>0 && length<(int) sizeof(input))
					input[length]='\0';
				else
				{
					SG_SERROR( "error reading cmdline\n");
					return 1;
				}
			}
			while (intf->parse_line(input));
			delete interface;
			return 0;
		}
#endif

		// from file
		if (argc==2)
		{
			FILE* file=fopen(argv[1], "r");

			if (!file)
			{
				SG_SERROR( "error opening/reading file: \"%s\"",argv[1]);
				delete interface;
				return 1;
			}
			else
			{
				try
				{
					while(!feof(file) && intf->parse_line(intf->get_line(file, false)));
				}
				catch (ShogunException e)
				{
					fclose(file);
					delete interface;
					return 1;
				}

				fclose(file);
				delete interface;
				return 0;
			}
		}

	}
	catch (std::bad_alloc)
	{
		SG_PRINT("Out of memory error.\n");
	}
	catch (ShogunException e)
	{
		SG_PRINT("%s", e.get_exception_string());
	}

}

#endif // HAVE_CMDLINE
