#include "lib/common.h"
#include <pthread.h>


class SldFeatures
{				// Sample features class
public:
	float64_t* feature_vector;
	int32_t dimensions;		// No. of features
	int32_t label;		// Label


};

enum example_used_t
{
	EMPTY = 1,
	NOT_USED = 2,
	USED = 3
};


class input_parser
{
 private:
	static void* parse_loop_entry_point(void* params);

protected:
	FILE* input_source;		// input source
	pthread_t parse_thread;	// parse thread

	SldFeatures* examples_buff; // features stored as they are read
	int32_t buffer_write_index; // where current example will be stored in buffer
	int32_t buffer_read_index;	// from where next example will be read
	example_used_t* is_example_used;	/* list, indicating whether example is in use or not */

	pthread_cond_t* example_in_use_condition;
	pthread_mutex_t* example_in_use_mutex;

	int32_t example_memsize;  // Size of example in the memory
	int32_t buffer_size;		   // Number of examples to store in buffer

	int32_t number_of_features; // Assume constant

	int32_t number_of_vectors_parsed;
	int32_t number_of_vectors_read;

	float64_t* current_feature_vector; // should point to a location in the buffer
	int32_t current_label;		   // -- same --
	int32_t current_number_of_features; // -- same --

	SldFeatures* current_example;


	// Storage of examples in buffer:
	// example 1:
	//	float64_t* feature_vector - 4 bytes
	//	int32_t dimensions - 4 bytes
	//	int32_t label - 4 bytes
	//	<feature_vector> list of 'dimensions' number of floats - dimensions*sizeof(float64_t) bytes
	// example 2...
	// ... and so on
	// The ptr to the feature vector will have to be set through the code
	// to point to the list of floats
	// So, effectively, feature_vector will point to the address 12 bytes after it.
	// This will be constant with time for each example in the buffer.
	// Thus, the feature vector should not be overwritten unless the example has been processed.


public:

	bool parsing_done;
	bool reading_done;

	input_parser();		// Constructor

	~input_parser();

	void init(FILE* input_file);

	bool is_running();

	int32_t get_number_of_features();

	int32_t get_label_and_vector(int32_t &label, float64_t* &feature_vector);

	void start_parser();

	void copy_example_into_buffer(SldFeatures* example);

	void set_buffer_size(int32_t size);
	void buffer_increment_write_index();
	void buffer_increment_read_index();

	void* main_parse_loop(void* params);

	SldFeatures* get_next_example();
	int32_t get_next_example(float64_t* &feature_vector, int32_t &length, int32_t &label);

	SldFeatures* geao(int32_t offset); /* get eg at offset, debug function */

	void finalize_example();	/* indicate that the current example has been processed, so make that mem loc writable */
	void end_parser();
};

