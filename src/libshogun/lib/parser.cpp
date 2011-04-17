#include "parser.h"

#include <stdio.h>
#include <string.h>

#define MAX_LABEL_LENGTH 3
#define MAX_FLOAT_LENGTH 10
#define MAX_NUMBER_OF_FEATURES 100


int32_t read_label(FILE* input_file, int &val)
{				// Read an int from file into val

	char *text = new char[MAX_LABEL_LENGTH + 1];
	int32_t characters = 0;
	while ( ( *text=fgetc(input_file) ) )
	{
		if (*text == ' ' || *text == '\n' || *text == EOF)
			break;
		characters++;
		text++;
	}

	*text='\0';
	text = text - characters;
	val = atoi(text);

	return characters;
}


int32_t read_float_vector(FILE* input_file, float64_t* &vector_ref, bool store_by_ref)
{
	// Read a vector of floats from file upto \n or EOF
	// store by ref should be set true if you want to change the
	// pointer location to point to a new space allocated for the features
	// 'false' assumes that you've already allocated enough memory for the features while passing

	char *text = new char[MAX_FLOAT_LENGTH + 1];
	if (store_by_ref)
		vector_ref = new float64_t[MAX_NUMBER_OF_FEATURES];

	int32_t characters_float = 0;
	int32_t number_of_floats = 0;
	int32_t end_proc = 0;

	while ( (*text = fgetc(input_file)) )
	{
		if (*text == '\n' || *text == EOF)
			end_proc = 1;

		if (*text == ' ' || end_proc == 1)
		{

			if (characters_float == 0)
			{
				if (end_proc)
				{
					vector_ref -= number_of_floats;
					return number_of_floats;
				}
				else
					continue;
			}

			else
			{
				*text = '\0';
				text = text - characters_float;

				*vector_ref = atof(text);
				vector_ref++;
				number_of_floats++;

				characters_float = 0;

				if (end_proc)
				{
					vector_ref -= number_of_floats;
					return number_of_floats;
				}

				else
					continue;
			}
		}

		text++;
		characters_float++;	// Number of chars in the current float
	}

	return number_of_floats;
	// should never reach here
}




input_parser::input_parser()
{
	init(NULL);
}

input_parser::~input_parser()
{
	free(examples_buff);
}

void input_parser::init(FILE* input_file)
{
	input_source = input_file;
	examples_buff = NULL;
	parsing_done = false;
	reading_done = false;
	is_example_used = NULL;
	number_of_vectors_parsed = 0;
	number_of_vectors_read = 0;
	buffer_size = 100;

	current_number_of_features = -1;
	current_label = -1;
	current_feature_vector = NULL;
	buffer_write_index = 0;
	buffer_read_index = -1;

	is_example_used = new example_used_t[buffer_size];
	example_in_use_condition = new pthread_cond_t[buffer_size];
	example_in_use_mutex = new pthread_mutex_t[buffer_size];

	for (int i=0; i<buffer_size; i++)
	{
		is_example_used[i] = EMPTY;
		pthread_cond_init(&example_in_use_condition[i], NULL);
		pthread_mutex_init(&example_in_use_mutex[i], NULL);
	}


}

void input_parser::start_parser()
{
	// examples_buff = new SldFeatures[buffer_size]; // initialize the parser's buffer
	// The buffer should be created after the first example when the size of the object is actually known

	printf("Starting parser..\n");
	pthread_create(&parse_thread, NULL, parse_loop_entry_point, this); // start the parse thread
}

void* input_parser::parse_loop_entry_point(void* params)
{
	// 'this' object should be passed as params
	printf("In entry point..\n");
	((input_parser *) params)->main_parse_loop(params);

	return NULL;
}

bool input_parser::is_running()
{
	// Return true if parsing or reading is still incomplete
	// May require a mutex lock

	if (parsing_done)
	{
		if (reading_done)
			return false;
		else
			return true;
	}
	else
	{
		return false;
	}

}

int32_t input_parser::get_number_of_features()
{
	// Get the number of features in the line
	// Assumes it is used only at the first line of input
	// WILL fseek BACK TO zero!

	int32_t ret, label, label_chars;
	float64_t* feature_vector;

	label_chars = read_label(input_source, label);

	if (label_chars == 0)
	{
		return -1;
	}

	ret = read_float_vector(input_source, feature_vector, true);
	// the address of feature_vector will be changed so that it points to the new location of the features

	fseek(input_source, 0, SEEK_SET); // Seek back to zero for further processing
	return ret;

}

int32_t input_parser::get_label_and_vector(int32_t &label, float64_t* &feature_vector)
{
	// Reads a line from file, gets label and vector
	// Assume label is 1st int of the line
	int32_t label_chars;
	int32_t number_of_floats;

	label_chars = read_label(input_source, label); // Get label

	if (label_chars == 0)
	{
		parsing_done = true; // temporarily, assume that parsing done when file read
		return -1;
	}

	number_of_floats = read_float_vector(input_source, feature_vector, false);

	if (number_of_floats < 1)
	{
		parsing_done = true;
		return -1;
	}

	return number_of_floats;
}

void input_parser::buffer_increment_write_index()
{
	// increment the write index, making the buffer behave in a circular fashion
	// Have to change this so that it doesn't increment the index if the example
	// in the next index hasn't been processed yet.

	buffer_write_index = (buffer_write_index + 1) % buffer_size;
}

void input_parser::buffer_increment_read_index()
{
	buffer_read_index = (buffer_read_index + 1) % buffer_size;
}

void input_parser::copy_example_into_buffer(SldFeatures* example)
{
	// Copy the example byte-by-byte into the buffer
	// Assume buffer already points to the available memory location

	// begin SldFeatures object
	//	float64_t* feature_vector
	//	int32_t dimensions
	//	int32_t label
	// end of SldFeatures object
	// list of float64_ts

	// First we should check if the example can be overwritten or not
	// in case the same buffer space is being used
	// wait on a cond to be true, else lock the code here


	// if the ex. is not used, then lock
	// if already used, it is safe to overwrite the location

	if (is_example_used[buffer_write_index] == NOT_USED)
		pthread_mutex_lock(&example_in_use_mutex[buffer_write_index]);

	while (is_example_used[buffer_write_index] == NOT_USED)
	{
		pthread_cond_wait(&example_in_use_condition[buffer_write_index], &example_in_use_mutex[buffer_write_index]);
	}



	// Find where to store the example in the buffer
	SldFeatures* current_example_loc = (SldFeatures *) ((char *)examples_buff + buffer_write_index*example_memsize);

	current_example_loc->feature_vector = (float64_t *) ((char *) current_example_loc + sizeof(float64_t*) + sizeof(int32_t) + sizeof(int32_t));

	current_example_loc->dimensions = example->dimensions;
	current_example_loc->label = example->label;

	// Now copy the features

	for (int i=0; i<current_example_loc->dimensions; i++)
	{
		current_example_loc->feature_vector[i] = example->feature_vector[i];
	}

	is_example_used[buffer_write_index] = NOT_USED; // set the example to unused
	pthread_mutex_unlock(&example_in_use_mutex[buffer_write_index]);
}


void* input_parser::main_parse_loop(void* params)
{
	// Read the examples into current_* objects
	// Instead of allocating mem for new objects each time

	input_parser* this_obj = (input_parser *) params;
	this->input_source = this_obj->input_source;

	while (!parsing_done)
	{
		// Get number of features from the first example parsed
		if (number_of_vectors_parsed == 0)
		{
			// Get number of features, allocate mem
			current_number_of_features = get_number_of_features();
			number_of_features = current_number_of_features;

			// Now allocate mem for buffer
			example_memsize = sizeof(float64_t*) + sizeof(int32_t) + sizeof(int32_t) + sizeof(float64_t)*number_of_features;

			current_example = (SldFeatures*) malloc(example_memsize);
			examples_buff = (SldFeatures *) malloc(example_memsize*buffer_size);

			// make it point to the list of floats in current_example
			current_feature_vector = (float64_t*) ((char *) current_example + sizeof(float64_t*) + sizeof(int32_t) + sizeof(int32_t));

		}

		// Get the example
		current_number_of_features = get_label_and_vector(current_label, current_feature_vector);

		if (current_number_of_features < 0)
		{
			parsing_done = true;
			return NULL;
		}


		current_example->feature_vector = current_feature_vector;
		current_example->dimensions = current_number_of_features;
		current_example->label = current_label;

		// Now copy the example into the buffer
		copy_example_into_buffer(current_example);

		buffer_increment_write_index();
		number_of_vectors_parsed++;


		printf("\n*************************\n");
		printf("Example number %d (Parsing)\n", number_of_vectors_parsed);
		for (int i=0; i<number_of_features; i++)
			printf("fv[%d] = %f\t", i, current_example->feature_vector[i]);
		printf("\n");

	}

	return NULL;

}

SldFeatures* input_parser::get_next_example()
{
	// Return the next unused example from the buffer

	if (buffer_read_index < 0)
	{
		if (number_of_vectors_parsed > 0)
			buffer_increment_read_index();
		else
			return NULL;
	}


	if (parsing_done)
	{
		if (number_of_vectors_read == number_of_vectors_parsed)
		{
			reading_done = true;
		}
	}


	if (number_of_vectors_read == number_of_vectors_parsed)
	{
		return NULL;
	}


	if (is_example_used[buffer_read_index] == NOT_USED)
	{
		pthread_mutex_lock(&example_in_use_mutex[buffer_read_index]);
		SldFeatures* example = (SldFeatures *) ((char *) examples_buff + example_memsize * buffer_read_index);
		pthread_mutex_unlock(&example_in_use_mutex[buffer_read_index]);
		number_of_vectors_read++;
		return example;
	}

	else
	{
		return NULL;
	}

}

int32_t input_parser::get_next_example(float64_t* &feature_vector, int32_t &length, int32_t &label)
{
	// if reading is done, no more examples can be fetched. return 0
	// else, if example can be read, get the example and return 1
	// otherwise, wait for further parsing, get the example and return 1


	SldFeatures *example;

	while (1)
	{
		if (reading_done)
			return 0;

		example = get_next_example();
		if (example == NULL)
			continue;
		else
			break;
	}


	feature_vector = example->feature_vector;
	length = example->dimensions;
	label = example->label;

	return 1;
}

void input_parser::set_buffer_size(int32_t size)
{
	buffer_size = size;
}

void input_parser::finalize_example()
{
	is_example_used[buffer_read_index] = USED;
	pthread_cond_signal(&example_in_use_condition[buffer_read_index]);
	buffer_increment_read_index();
}


SldFeatures* input_parser::geao(int32_t offset) // get example at offset
{
	return (SldFeatures*) ((char *) examples_buff + example_memsize*offset);
}


void input_parser::end_parser()
{
	pthread_join(parse_thread, NULL);
	fclose(input_source);
}


