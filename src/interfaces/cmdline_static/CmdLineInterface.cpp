#include "CmdLineInterface.h"

#include <shogun/lib/config.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/CSVFile.h>
#include <shogun/ui/SGInterface.h>

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

#ifndef WIN32
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

#include <stdio.h>
#include <strings.h>

void cmdline_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void cmdline_print_warning(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void cmdline_print_error(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void cmdline_cancel_computations(bool &delayed, bool &immediately)
{
}

using namespace shogun;

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
	delete m_lhs;
	delete m_rhs;
}

void CCmdLineInterface::reset(const char* line)
{
	CSGInterface::reset();

	if (!line)
		return;

	char* element=NULL;
	const char delim_equal[]="=";
	const char delim_lhs[]="=, \t\n";
	const char delim_rhs[]=" \t\n";

	delete m_lhs;
	m_lhs=NULL;
	delete m_rhs;
	m_rhs=NULL;

	/* split lhs from rhs
	 * for some reason strstr on sunos and newer libc's
	 * requires a char* haystack
	 */
	char* equal_sign=strstr((char*) line, delim_equal);
	if (equal_sign)
	{
#ifdef DEBUG_CMDLINEIF
		SG_PRINT("has lhs!\n");
#endif
		element=strtok((char*) line, delim_lhs);
		if (element)
		{
			m_lhs=new DynArray<char*>();
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
		m_rhs=new DynArray<char*>();
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

	char* chunk=SG_MALLOC(char, len+1);
	memset(chunk, 0, sizeof(char)*(len+1));
	size_t nread=fread(chunk, sizeof(char), len, fh);
	fclose(fh);
	if (nread<=0)
		SG_ERROR("Could not read data from %s.\n");

	char* signature=SG_MALLOC(char, len+1);
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

	SG_FREE(signature);
	SG_FREE(chunk);
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

	char* result=SG_MALLOC(char, len+1);
	memcpy(result, s, len*sizeof(char));
	result[len]='\0';

	return result;
}

#define GET_VECTOR(fname, sg_type) \
void CCmdLineInterface::fname(sg_type*& vec, int32_t& len)	\
{																		\
	const char* filename=get_arg_increment();							\
	if (!filename)														\
		SG_ERROR("No filename given to read vector.\n");				\
																		\
	CCSVFile f((char*) filename, 'r');								\
																		\
	try																	\
	{																	\
		f.fname(vec, len);												\
	}																	\
	catch (...)															\
	{																	\
		SG_ERROR("Could not read data from %s.\n", filename);			\
	}																	\
}
GET_VECTOR(get_vector, uint8_t)
GET_VECTOR(get_vector, char)
GET_VECTOR(get_vector, int32_t)
GET_VECTOR(get_vector, float32_t)
GET_VECTOR(get_vector, float64_t)
GET_VECTOR(get_vector, int16_t)
GET_VECTOR(get_vector, uint16_t)
#undef GET_VECTOR

#define GET_MATRIX(fname, sg_type) \
void CCmdLineInterface::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec) \
{																		\
	const char* filename=get_arg_increment();							\
	if (!filename)														\
		SG_ERROR("No filename given to read matrix.\n");				\
																		\
	CCSVFile f((char*) filename, 'r');								\
																		\
	try																	\
	{																	\
		f.fname(matrix, num_feat, num_vec);								\
	}																	\
	catch (...)															\
	{																	\
		SG_ERROR("Could not read data from %s.\n", filename);			\
	}																	\
}
GET_MATRIX(get_matrix, uint8_t)
GET_MATRIX(get_matrix, char)
GET_MATRIX(get_matrix, int32_t)
GET_MATRIX(get_matrix, float32_t)
GET_MATRIX(get_matrix, float64_t)
GET_MATRIX(get_matrix, int16_t)
GET_MATRIX(get_matrix, uint16_t)
#undef GET_MATRIX

void CCmdLineInterface::get_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CCmdLineInterface::get_sparse_matrix(SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	const char* filename=get_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to read SPARSE REAL matrix.\n");

	CCSVFile f((char*) filename, 'r');
	f.get_sparse_matrix(matrix, num_feat, num_vec);
}


void CCmdLineInterface::get_string_list(SGString<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_string_list(SGString<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	const char* filename=get_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to read CHAR string list.\n");

	CCSVFile f((char*) filename, 'r');
	f.get_string_list(strings, num_str, max_string_len);
}

void CCmdLineInterface::get_string_list(SGString<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_string_list(SGString<int16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_string_list(SGString<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}


void CCmdLineInterface::get_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* &attrs)
{
	attrs=NULL;
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

#define SET_SCALAR(fname, mfname, sg_type)	\
void CCmdLineInterface::fname(sg_type scalar)	\
{												\
	const char* filename=set_arg_increment();	\
	CCSVFile f((char*) filename, 'w');		\
	f.mfname(&scalar, 1, 1);					\
}
SET_SCALAR(set_int, set_matrix, int32_t)
SET_SCALAR(set_real, set_matrix, float64_t)
SET_SCALAR(set_bool, CFile::set_matrix, bool)
#undef SET_SCALAR

#define SET_VECTOR(fname, mfname, sg_type)	\
void CCmdLineInterface::fname(const sg_type* vec, int32_t len)	\
{																\
	const char* filename=set_arg_increment();					\
	if (!filename)												\
		SG_ERROR("No filename given to write vector.\n");		\
																\
	CCSVFile f((char*) filename, 'w');						\
	f.mfname(vec, len, 1);										\
}
SET_VECTOR(set_vector, set_matrix, uint8_t)
SET_VECTOR(set_vector, set_matrix, char)
SET_VECTOR(set_vector, set_matrix, int32_t)
SET_VECTOR(set_vector, set_matrix, float32_t)
SET_VECTOR(set_vector, set_matrix, float64_t)
SET_VECTOR(set_vector, set_matrix, int16_t)
SET_VECTOR(set_vector, set_matrix, uint16_t)
#undef SET_VECTOR

#define SET_MATRIX(fname, sg_type)	\
void CCmdLineInterface::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec)	\
{																						\
	const char* filename=set_arg_increment();											\
	if (!filename)																		\
		SG_ERROR("No filename given to write matrix.\n");								\
																						\
	CCSVFile f((char*) filename, 'w');												\
	f.fname(matrix, num_feat, num_vec);										\
}
SET_MATRIX(set_matrix, uint8_t)
SET_MATRIX(set_matrix, char)
SET_MATRIX(set_matrix, int32_t)
SET_MATRIX(set_matrix, float32_t)
SET_MATRIX(set_matrix, float64_t)
SET_MATRIX(set_matrix, int16_t)
SET_MATRIX(set_matrix, uint16_t)

void CCmdLineInterface::set_sparse_matrix(const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz)
{
	const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write SPARSE REAL matrix.\n");

	CCSVFile f((char*) filename, 'w');
	f.set_sparse_matrix(matrix, num_feat, num_vec);
}

#define SET_STRING_LIST(fname, sg_type)	\
void CCmdLineInterface::fname(const SGString<sg_type>* strings, int32_t num_str)		\
{																						\
	const char* filename=set_arg_increment();											\
	if (!filename)																		\
		SG_ERROR("No filename given to write CHAR string list.\n");						\
																						\
	CCSVFile f((char*) filename, 'w');												\
	f.fname(strings, num_str);															\
}
SET_STRING_LIST(set_string_list, uint8_t)
SET_STRING_LIST(set_string_list, char)
SET_STRING_LIST(set_string_list, int32_t)
SET_STRING_LIST(set_string_list, int16_t)
SET_STRING_LIST(set_string_list, uint16_t)
#undef SET_STRING_LIST

void CCmdLineInterface::set_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* attrs)
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

	char* skipped=SGIO::skip_blanks((char*) line);
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

	init_shogun(&cmdline_print_message, &cmdline_print_warning,
			&cmdline_print_error, &cmdline_cancel_computations);
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

			SG_UNREF(interface);
			exit_shogun();
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

			SG_UNREF(interface);
			exit_shogun();
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
			SG_UNREF(interface);
			exit_shogun();
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
				SG_UNREF(interface);
				exit_shogun();
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
					SG_UNREF(interface);
					exit_shogun();
					return 1;
				}

				fclose(file);
				SG_UNREF(interface);
				exit_shogun();
				return 0;
			}
		}

	}
	catch (std::bad_alloc)
	{
		SG_SPRINT("Out of memory error.\n");
		SG_UNREF(interface);
		exit_shogun();
		return 2;
	}
	catch (ShogunException e)
	{
		SG_SPRINT("%s", e.get_exception_string());
		SG_UNREF(interface);
		exit_shogun();
		return 3;
	}
	catch (...)
	{
		SG_SPRINT("Returning from SHOGUN unknown error.");
		SG_UNREF(interface);
		exit_shogun();
		return 4;
	}
}
