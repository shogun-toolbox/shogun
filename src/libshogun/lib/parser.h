#ifndef __INPUT_PARSER_H__
#define __INPUT_PARSER_H__

#include "lib/io.h"
#include "lib/StreamingFile.h"
#include "lib/common.h"
#include <pthread.h>

namespace shogun
{
/// Unlabelled example class
class UnlabelledExample
{
public:
	float64_t* feature_vector;
	int32_t dimensions;
};

/// Labelled example class
class LabelledExample
{
public:
	float64_t* feature_vector;
	int32_t dimensions;
	float64_t label;
};

enum example_used_t
{
	EMPTY = 1,
	NOT_USED = 2,
	USED = 3
};

enum example_type_t
{
	LABELLED = 1,
	UNLABELLED = 2
};

class input_parser
{
private:
	/** 
	 * Entry point for the parse thread.
	 * 
	 * @param params this object
	 * 
	 * @return NULL
	 */
	static void* parse_loop_entry_point(void* params);

protected:
	
	CStreamingFile* input_source; /**< Input source,
				       * CStreamingFile object */
	
	pthread_t parse_thread;	/**< Parse thread */

	void* examples_buff;	/**< Buffer for examples, behaves
				 * like a ring.
				 * Examples are stored and retrieved
				 * from this buffer.*/
	
	int32_t buffer_write_index; /**< Where next example will be
				     * written into the buffer. */
	
	int32_t buffer_read_index; /**< Where next example will be
				    * read from the buffer */
	
	example_used_t* is_example_used; /**< Indicates state of examples
					  * in buffer - used, not
					  * used, or empty. */

	pthread_cond_t* example_in_use_condition;
	pthread_mutex_t* example_in_use_mutex;

	int32_t example_memsize; /**< Size of example object */
	int32_t buffer_size;	/**< Number of examples to store in buffer */
	
	int32_t number_of_features;

	int32_t number_of_vectors_parsed;
	int32_t number_of_vectors_read;

	float64_t* current_feature_vector; /**< Points to feature
					    * vector of last read example */
	
	float64_t current_label; /**< Label of last read example */
	
	int32_t current_number_of_features; /**< Features in last
					     * read example */

	void* current_example;	/**< Points to current example in buffer */


public:

	bool parsing_done;	/**< true if all input is parsed */
	bool reading_done;	/**< true if all examples are fetched */

	example_type_t example_type; /**< LABELLED or UNLABELLED */

	/** 
	 * Constructor
	 * 
	 */
	input_parser();

	/** 
	 * Destructor
	 * 
	 */
	~input_parser();

	/** 
	 * Initializer
	 * 
	 * Sets initial or default values for members.
	 * is_example_used is initialized to EMPTY.
	 * example_type is LABELLED by default.
	 * 
	 * @param input_file CStreamingFile object
	 */
	void init(CStreamingFile* input_file);

	/** 
	 * Test if parser is running.
	 * 
	 * @return true if running, false otherwise.
	 */
	bool is_running();

	/** 
	 * Get number of features from example.
	 * Currently reads first line of input to infer.
	 * 
	 * @return Number of features
	 */
	int32_t get_number_of_features();

	/** 
	 * Gets feature vector, length and label.
	 * Sets their values by reference.
	 * Uses method for reading the vector defined in CStreamingFile.
	 * 
	 * @param feature_vector Pointer to feature vector
	 * @param length Features in vector
	 * @param label Label of example
	 * 
	 * @return 1 on success, 0 on failure.
	 */
	int32_t get_vector_and_label(float64_t* &feature_vector,
				     int32_t &length,
				     float64_t &label);

	/** 
	 * Gets feature vector and length by reference.
	 * Assumes examples are unlabelled.
	 * Uses method for reading the vector defined in CStreamingFile.
	 * 
	 * @param feature_vector Pointer to feature vector
	 * @param length Features in vector
	 * 
	 * @return 1 on success, 0 on failure
	 */
	int32_t get_vector_only(float64_t* &feature_vector, int32_t &length);

	/** 
	 * Starts the parser, creating a new thread.
	 * 
	 * main_parse_loop is the parsing method.
	 */
	void start_parser();

	/** 
	 * Set buffer size in units of number of examples.
	 * 
	 * @param size Size of buffer
	 */
	void set_buffer_size(int32_t size);

	/** 
	 * Increment buffer write index.
	 * 
	 */
	void buffer_increment_write_index();

	/** 
	 * Increment buffer read index.
	 * 
	 */
	void buffer_increment_read_index();

	/** 
	 * Main parsing loop. Reads examples from source and stores
	 * them in the buffer.
	 * 
	 * @param params 'this' object
	 * 
	 * @return NULL
	 */
	void* main_parse_loop(void* params);

	/** 
	 * Copy example into the buffer.
	 *
	 * Buffer space:
	 * -Example n-
	 * float64_t* feature_vector
	 * int32_t length
	 * float64_t label (if LabelledExample)
	 * list of float64_ts (Feature vector)
	 * -Example n+1-
	 * ..
	 *
	 * Currently, constant dimensionality is assumed!
	 *
	 * @param example Example to be copied.
	 */
	void copy_example_into_buffer(void* example);

	/** 
	 * Gets the next unused example from the buffer.
	 * 
	 * @return NULL if no unused example is available, else return
	 * a pointer to the example in the buffer.
	 */
	void* get_next_example();

	/** 
	 * Gets the next example, assuming it to be labelled.
	 *
	 * Waits till get_next_example returns a valid example, or
	 * returns if reading is done already.
	 * 
	 * @param feature_vector Feature vector pointer
	 * @param length Length of feature vector
	 * @param label Label of example
	 * 
	 * @return 1 if an example could be fetched, 0 otherwise
	 */
	int32_t get_next_example_labelled(float64_t* &feature_vector,
					  int32_t &length,
					  float64_t &label);

	/** 
	 * Gets the next example, assuming it to be unlabelled.
	 *
	 * Waits till get_next_example returns a valid example, or
	 * returns if reading is done already.
	 * 
	 * @param feature_vector Feature vector pointer
	 * @param length Length of feature vector
	 * 
	 * @return 1 if an example could be fetched, 0 otherwise
	 */
	int32_t get_next_example_unlabelled(float64_t* &feature_vector,
					    int32_t &length);

	/** 
	 * Finalize the current example, indicating that the buffer
	 * position it occupies may be overwritten by the parser.
	 *
	 * Should be called when the example has been processed by the
	 * external algorithm.
	 */
	void finalize_example();

	/** 
	 * End the parser, closing the parse thread.
	 * 
	 */
	void end_parser();
};
}
#endif // __INPUT_PARSER_H__
