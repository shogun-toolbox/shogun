#include "buffer.h"

ParseBuffer::ParseBuffer(int32_t size)
{
	buffer_size=size*1024*1024;
	ex_buff=new example[buffer_size];
	fv_buff=new float64_t[buffer_size/sizeof(float64_t)];
	ex_used=new E_IS_EXAMPLE_USED[buffer_size];
	
	ex_in_use_mutex=new pthread_mutex_t[buffer_size];
	ex_in_use_cond=new pthread_cond_t[buffer_size];
	
	ex_write_index=0;
	ex_read_index=-1;

	fv_write_index=0;

	for (int32_t i=0; i<buffer_size; i++)
	{
		ex_used[i] = E_EMPTY;
		pthread_cond_init(&ex_in_use_cond[i], NULL);
		pthread_mutex_init(&ex_in_use_mutex[i], NULL);
	}
}

ParseBuffer::~ParseBuffer()
{
	delete[] ex_buff;
	delete[] fv_buff;
	delete[] ex_used;

	for (int32_t i=0; i<buffer_size; i++)
	{
		pthread_mutex_destroy(&ex_in_use_mutex[i]);
		pthread_cond_destroy(&ex_in_use_cond[i]);
	}
}

inline void ParseBuffer::inc_read_index()
{
	ex_read_index=(ex_read_index + 1) % buffer_size;
}

inline void ParseBuffer::inc_write_index(int32_t len)
{
	ex_write_index=(ex_write_index + 1) % buffer_size;
	fv_write_index=fv_write_index + len;
}

int32_t ParseBuffer::write_example(example *ex)
{
	ex_buff[ex_write_index].label = ex->label;
	ex_buff[ex_write_index].feature_vector = &fv_buff[fv_write_index];
	ex_buff[ex_write_index].len = ex->len;
	
	//Write feature vector into the fv buffer
	//First we should check if the remaining length is enough to accommodate the fv
	//Then realloc/expand if necessary

	for (int i=0; i<ex->len; i++)
	{
		fv_buff[fv_write_index+i] = ex->feature_vector[i];
	}

	ex_used[ex_write_index] = E_NOT_USED;
	inc_write_index(ex->len);

	return 1;					// Should check for size and return 0 if insufficient
}

example* ParseBuffer::get_example()
{
	example* ex;
	
	if (ex_read_index >= 0)
	{
		ex = &ex_buff[ex_read_index];
		inc_read_index();
		
		return ex;
	}
	else
		return NULL;
}

inline example* ParseBuffer::fetch_example()
{
	example *ex;

	pthread_mutex_lock(&ex_in_use_mutex[ex_read_index]);
	int32_t read_index=ex_read_index;
	// Because read index will change after get_example
	
	if (ex_used[read_index] == E_NOT_USED)
		ex = get_example();
	else
		ex = NULL;
	
	pthread_mutex_unlock(&ex_in_use_mutex[read_index]);

	return ex;
}
	
inline int32_t ParseBuffer::copy_example(example *ex)
{
	// Check this mutex call.. It should probably be locked regardless of ex in use

	int32_t ret;

	pthread_mutex_lock(&ex_in_use_mutex[ex_write_index]);

	while (ex_used[ex_write_index] == E_NOT_USED)
	{
		pthread_cond_wait(&ex_in_use_cond[ex_write_index], &ex_in_use_mutex[ex_write_index]);
	}
	
	ret=write_example(ex);

	if (ex_read_index < 0)
		ex_read_index = 0;
	
	pthread_mutex_unlock(&ex_in_use_mutex[ex_write_index]);

	return ret;
}

inline void ParseBuffer::finalize_example()
{
	pthread_mutex_lock(&ex_in_use_mutex[ex_read_index]);
	ex_used[ex_read_index] = E_USED;
	pthread_cond_signal(&ex_in_use_cond[ex_read_index]);
	pthread_mutex_unlock(&ex_in_use_mutex[ex_read_index]);

	inc_read_index();
}
