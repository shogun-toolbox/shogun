/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __INPUTPARSER_H__
#define __INPUTPARSER_H__

#include <shogun/lib/common.h>
#ifdef HAVE_PTHREAD

#include <shogun/io/SGIO.h>
#include <shogun/io/streaming/StreamingFile.h>
#include <shogun/io/streaming/ParseBuffer.h>
#include <pthread.h>

#define PARSER_DEFAULT_BUFFSIZE 100

namespace shogun
{
	/// Type of example, either E_LABELLED
	/// or E_UNLABELLED
	enum E_EXAMPLE_TYPE
	{
		E_LABELLED = 1,
		E_UNLABELLED = 2
	};

/** @brief Class CInputParser is a templated class used to
 * maintain the reading/parsing/providing of examples.
 *
 * Parsing is done in a thread separate from the learner.
 *
 * Note that parsing is not done directly by this class,
 * but by the Streaming*File classes. This class only calls
 * the required get_vector* functions from the StreamingFile
 * object. (Exactly which function should be called is set
 * through the set_read_vector* functions)
 *
 * The template type should be the type of feature vector the
 * parser should return. Eg. CInputParser<float32_t> means it
 * will expect a float32_t* vector to be returned from the get_vector
 * function. Other parameters returned are length of feature vector
 * and the label, if applicable.
 *
 * If the vectors cannot be directly represented as say float32_t*
 * one can instantiate eg. CInputParser<VwExample> and it looks for
 * a get_vector function which returns a VwExample, which may contain
 * any kind of data, including label, vector, weights, etc. It is then
 * up to the external algorithm to handle such objects.

 * The parser should first be started with a call to the start_parser()
 * function which starts a new thread for continuous parsing of examples.
 *
 * Parsing is done through the CParseBuffer object, which in its
 * current implementation is a ring of a specified number of examples.
 * It is the task of the CInputParser object to ensure that this ring
 * is being updated with new parsed examples.
 *
 * CInputParser provides mainly the get_next_example function which
 * returns the next example from the CParseBuffer object to the caller
 * (usually a StreamingFeatures object). When one is done using
 * the example, finalize_example() should be called, leaving the
 * spot free for a new example to be loaded.
 *
 * The parsing thread should be joined with a call to end_parser().
 * exit_parser() may be used to cancel the parse thread if needed.
 *
 * Options are provided for automatic SG_FREEing of example objects
 * after each finalize_example() and also on CInputParser destruction.
 * They are set through the set_free_vector* functions.
 * Do not free vectors on finalize_example() if you intend to reuse the
 * same vector memory locations for different examples.
 * Do not free vectors on destruction if you are bound to free them
 * manually later.
 */
template <class T> class CInputParser
{
public:

    /**
     * Constructor
     *
     */
    CInputParser();

    /**
     * Destructor
     *
     */
    ~CInputParser();

    /**
     * Initializer
     *
     * Sets initial or default values for members.
     * is_example_used is initialized to EMPTY.
     * example_type is LABELLED by default.
     *
     * @param input_file CStreamingFile object
     * @param is_labelled Whether example is labelled or not (bool), optional
     * @param size Size of the buffer in number of examples
     */
    void init(CStreamingFile* input_file, bool is_labelled = true, int32_t size = PARSER_DEFAULT_BUFFSIZE);

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
    int32_t get_number_of_features() { return number_of_features; }

    /**
     * Sets the function used for reading a vector from
     * the file.
     *
     * The function must be a member of CStreamingFile,
     * taking a T* as input for the vector, and an int for
     * length, setting both by reference. The function
     * returns void.
     *
     * The argument is a function pointer to that function.
     */
    void set_read_vector(void (CStreamingFile::*func_ptr)(T* &vec, int32_t &len));

    /**
     * Sets the function used for reading a vector and
     * label from the file.
     *
     * The function must be a member of CStreamingFile,
     * taking a T* as input for the vector, an int for
     * length, and a float for the label, setting all by
     * reference. The function returns void.
     *
     * The argument is a function pointer to that function.
     */
    void set_read_vector_and_label(void (CStreamingFile::*func_ptr)(T* &vec, int32_t &len, float64_t &label));

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
    int32_t get_vector_and_label(T* &feature_vector,
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
    int32_t get_vector_only(T* &feature_vector, int32_t &length);

    /**
     * Sets whether to SG_FREE() the vector explicitly
     * after it has been used
     *
     * @param free_vec whether to SG_FREE() or not, bool
     */
    void set_free_vector_after_release(bool free_vec);

    /**
     * Sets whether to free all vectors that were
     * allocated in the ring upon destruction of the ring.
     *
     * @param destroy free all vectors on destruction
     */
    void set_free_vectors_on_destruct(bool destroy);

    /**
     * Starts the parser, creating a new thread.
     *
     * main_parse_loop is the parsing method.
     */
    void start_parser();

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
     * @param ex Example to be copied.
     */
    void copy_example_into_buffer(Example<T>* ex);

    /**
     * Retrieves the next example from the buffer.
     *
     *
     * @return The example pointer.
     */
    Example<T>* retrieve_example();

    /**
     * Gets the next example, assuming it to be labelled.
     *
     * Waits till retrieve_example returns a valid example, or
     * returns if reading is done already.
     *
     * @param feature_vector Feature vector pointer
     * @param length Length of feature vector
     * @param label Label of example
     *
     * @return 1 if an example could be fetched, 0 otherwise
     */
    int32_t get_next_example(T* &feature_vector,
                 int32_t &length,
                 float64_t &label);

    /**
     * Gets the next example, assuming it to be unlabelled.
     *
     * @param feature_vector
     * @param length
     *
     * @return 1 if an example could be fetched, 0 otherwise
     */
    int32_t get_next_example(T* &feature_vector,
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
     * End the parser, waiting for the parse thread to complete.
     *
     */
    void end_parser();

    /** Terminates the parsing thread
     */
    void exit_parser();

    /**
     * Returns the size of the examples ring
     *
     * @return ring size in terms of number of examples
     */
    int32_t get_ring_size() { return ring_size; }

private:
    /**
     * Entry point for the parse thread.
     *
     * @param params this object
     *
     * @return NULL
     */
    static void* parse_loop_entry_point(void* params);

public:
    bool parsing_done;	/**< true if all input is parsed */
    bool reading_done;	/**< true if all examples are fetched */

    E_EXAMPLE_TYPE example_type; /**< LABELLED or UNLABELLED */

protected:
    /**
     * This is the function pointer to the function to
     * read a vector from the input.
     *
     * It is called while reading a vector.
     */
    void (CStreamingFile::*read_vector) (T* &vec, int32_t &len);

    /**
     * This is the function pointer to the function to
     * read a vector and label from the input.
     *
     * It is called while reading a vector and a label.
     */
    void (CStreamingFile::*read_vector_and_label) (T* &vec, int32_t &len, float64_t &label);

    /// Input source, CStreamingFile object
    CStreamingFile* input_source;

    /// Thread in which the parser runs
    pthread_t parse_thread;

    /// The ring of examples, stored as they are parsed
    CParseBuffer<T>* examples_ring;

    /// Number of features in dataset (max of 'seen' features upto point of access)
    int32_t number_of_features;

    /// Number of vectors parsed
    int32_t number_of_vectors_parsed;

    /// Number of vectors used by external algorithm
    int32_t number_of_vectors_read;

    /// Example currently being used
    Example<T>* current_example;

    /// Feature vector of current example
    T* current_feature_vector;

    /// Label of current example
    float64_t current_label;

    /// Number of features in current example
    int32_t current_len;

    /// Whether to SG_FREE() vector after it is used
    bool free_after_release;

    /// Size of the ring of examples
    int32_t ring_size;

    /// Mutex which is used when getting/setting state of examples (whether a new example is ready)
    pthread_mutex_t examples_state_lock;

    /// Condition variable to indicate change of state of examples
    pthread_cond_t examples_state_changed;

};

template <class T>
    void CInputParser<T>::set_read_vector(void (CStreamingFile::*func_ptr)(T* &vec, int32_t &len))
{
    // Set read_vector to point to the function passed as arg
    read_vector=func_ptr;
}

template <class T>
    void CInputParser<T>::set_read_vector_and_label(void (CStreamingFile::*func_ptr)(T* &vec, int32_t &len, float64_t &label))
{
    // Set read_vector_and_label to point to the function passed as arg
    read_vector_and_label=func_ptr;
}

template <class T>
    CInputParser<T>::CInputParser()
{
	/* this line was commented out when I found it. However, the mutex locks
	 * have to be initialised. Otherwise uninitialised memory error */
	//init(NULL, true, PARSER_DEFAULT_BUFFSIZE);
	pthread_mutex_init(&examples_state_lock, NULL);
	pthread_cond_init(&examples_state_changed, NULL);
	examples_ring=NULL;
	parsing_done=true;
	reading_done=true;
}

template <class T>
    CInputParser<T>::~CInputParser()
{
	pthread_mutex_destroy(&examples_state_lock);
	pthread_cond_destroy(&examples_state_changed);

	SG_UNREF(examples_ring);
}

template <class T>
    void CInputParser<T>::init(CStreamingFile* input_file, bool is_labelled, int32_t size)
{
    input_source = input_file;

    if (is_labelled == true)
        example_type = E_LABELLED;
    else
        example_type = E_UNLABELLED;

    examples_ring = new CParseBuffer<T>(size);
    SG_REF(examples_ring);

    parsing_done = false;
    reading_done = false;
    number_of_vectors_parsed = 0;
    number_of_vectors_read = 0;

    current_len = -1;
    current_label = -1;
    current_feature_vector = NULL;

    free_after_release=true;
    ring_size=size;
}

template <class T>
    void CInputParser<T>::set_free_vector_after_release(bool free_vec)
{
    free_after_release=free_vec;
}

template <class T>
    void CInputParser<T>::set_free_vectors_on_destruct(bool destroy)
{
	examples_ring->set_free_vectors_on_destruct(destroy);
}

template <class T>
    void CInputParser<T>::start_parser()
{
	SG_SDEBUG("entering CInputParser::start_parser()\n")
    if (is_running())
    {
        SG_SERROR("Parser thread is already running! Multiple parse threads not supported.\n")
    }

    SG_SDEBUG("creating parse thread\n")
    pthread_create(&parse_thread, NULL, parse_loop_entry_point, this);

    SG_SDEBUG("leaving CInputParser::start_parser()\n")
}

template <class T>
    void* CInputParser<T>::parse_loop_entry_point(void* params)
{
    ((CInputParser *) params)->main_parse_loop(params);

    return NULL;
}

template <class T>
    bool CInputParser<T>::is_running()
{
	SG_SDEBUG("entering CInputParser::is_running()\n")
    bool ret;

    pthread_mutex_lock(&examples_state_lock);

    if (parsing_done)
        if (reading_done)
            ret = false;
        else
            ret = true;
    else
        ret = false;

    pthread_mutex_unlock(&examples_state_lock);

    SG_SDEBUG("leaving CInputParser::is_running(), returning %d\n", ret)
    return ret;
}

template <class T>
    int32_t CInputParser<T>::get_vector_and_label(T* &feature_vector,
                              int32_t &length,
                              float64_t &label)
{
    (input_source->*read_vector_and_label)(feature_vector, length, label);

    if (length < 1)
    {
        // Problem reading the example
        return 0;
    }

    return 1;
}

template <class T>
    int32_t CInputParser<T>::get_vector_only(T* &feature_vector,
                         int32_t &length)
{
    (input_source->*read_vector)(feature_vector, length);

    if (length < 1)
    {
        // Problem reading the example
        return 0;
    }

    return 1;
}

template <class T>
    void CInputParser<T>::copy_example_into_buffer(Example<T>* ex)
{
    examples_ring->copy_example(ex);
}

template <class T> void* CInputParser<T>::main_parse_loop(void* params)
{
    // Read the examples into current_* objects
    // Instead of allocating mem for new objects each time
#ifdef HAVE_PTHREAD
    CInputParser* this_obj = (CInputParser *) params;
    this->input_source = this_obj->input_source;

    while (1)
	{
		pthread_mutex_lock(&examples_state_lock);
		if (parsing_done)
		{
			pthread_mutex_unlock(&examples_state_lock);
			return NULL;
		}
		pthread_mutex_unlock(&examples_state_lock);

		pthread_testcancel();

		current_example = examples_ring->get_free_example();
		current_feature_vector = current_example->fv.vector;
		current_len = current_example->fv.vlen;
		current_label = current_example->label;

		if (example_type == E_LABELLED)
			get_vector_and_label(current_feature_vector, current_len, current_label);
		else
			get_vector_only(current_feature_vector,	current_len);

		if (current_len < 0)
		{
			pthread_mutex_lock(&examples_state_lock);
			parsing_done = true;
			pthread_cond_signal(&examples_state_changed);
			pthread_mutex_unlock(&examples_state_lock);
			return NULL;
		}

		current_example->label = current_label;
		current_example->fv.vector = current_feature_vector;
		current_example->fv.vlen = current_len;

		examples_ring->copy_example(current_example);

		pthread_mutex_lock(&examples_state_lock);
		number_of_vectors_parsed++;
		pthread_cond_signal(&examples_state_changed);
		pthread_mutex_unlock(&examples_state_lock);
	}
#endif /* HAVE_PTHREAD */
    return NULL;
}

template <class T> Example<T>* CInputParser<T>::retrieve_example()
{
    /* This function should be guarded by mutexes while calling  */
    Example<T> *ex;

    if (parsing_done)
    {
        if (number_of_vectors_read == number_of_vectors_parsed)
        {
            reading_done = true;
            /* Signal to waiting threads that no more examples are left */
            pthread_cond_signal(&examples_state_changed);
            return NULL;
        }
    }

    if (number_of_vectors_parsed <= 0)
        return NULL;

    if (number_of_vectors_read == number_of_vectors_parsed)
    {
        return NULL;
    }

    ex = examples_ring->get_unused_example();
    number_of_vectors_read++;

    return ex;
}

template <class T> int32_t CInputParser<T>::get_next_example(T* &fv,
        int32_t &length, float64_t &label)
{
    /* if reading is done, no more examples can be fetched. return 0
       else, if example can be read, get the example and return 1.
       otherwise, wait for further parsing, get the example and
       return 1 */

    Example<T> *ex;

    while (1)
    {
        if (reading_done)
            return 0;

        pthread_mutex_lock(&examples_state_lock);
        ex = retrieve_example();

        if (ex == NULL)
        {
            if (reading_done)
            {
                /* No more examples left, return */
                pthread_mutex_unlock(&examples_state_lock);
                return 0;
            }
            else
            {
                /* Examples left, wait for one to become ready */
                pthread_cond_wait(&examples_state_changed, &examples_state_lock);
                pthread_mutex_unlock(&examples_state_lock);
                continue;
            }
        }
        else
        {
            /* Example ready, return the example */
            pthread_mutex_unlock(&examples_state_lock);
            break;
        }
    }

    fv = ex->fv.vector;
    length = ex->fv.vlen;
    label = ex->label;

    return 1;
}

template <class T>
    int32_t CInputParser<T>::get_next_example(T* &fv, int32_t &length)
{
    float64_t label_dummy;

    return get_next_example(fv, length, label_dummy);
}

template <class T>
    void CInputParser<T>::finalize_example()
{
    examples_ring->finalize_example(free_after_release);
}

template <class T> void CInputParser<T>::end_parser()
{
	SG_SDEBUG("entering CInputParser::end_parser\n")
	SG_SDEBUG("joining parse thread\n")
    pthread_join(parse_thread, NULL);
    SG_SDEBUG("leaving CInputParser::end_parser\n")
}

template <class T> void CInputParser<T>::exit_parser()
{
	SG_SDEBUG("cancelling parse thread\n")
    pthread_cancel(parse_thread);
}
}

#endif /* HAVE_PTHREAD */

#endif // __INPUTPARSER_H__
