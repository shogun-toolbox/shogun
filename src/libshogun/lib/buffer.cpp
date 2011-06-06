/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/buffer.h"

using namespace shogun;

ParseBuffer::ParseBuffer(int32_t size)
{
	int32_t buffer_size_feat=size*1024*1024;
	buffer_size=100;
	ex_buff=new example[buffer_size];
	fv_buff=new float64_t[buffer_size_feat/sizeof(float64_t)];
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

void ParseBuffer::inc_read_index()
{
	ex_read_index=(ex_read_index + 1) % buffer_size;
}

void ParseBuffer::inc_write_index(int32_t len)
{
	ex_write_index=(ex_write_index + 1) % buffer_size;
	fv_write_index=fv_write_index + len;
}

int32_t ParseBuffer::write_example(example *ex)
{
  //printf("write_example... write_index=%d.\n", ex_write_index);
	ex_buff[ex_write_index].label = ex->label;
	ex_buff[ex_write_index].fv.vector = &fv_buff[fv_write_index];
	ex_buff[ex_write_index].fv.length = ex->fv.length;
		
	printf("write_index=%d\n", ex_write_index);
	//printf("label=%f, len=%d.\n", ex->label, ex->len);

	//Write feature vector into the fv buffer
	//First we should check if the remaining length is enough to accommodate the fv
	//Then realloc/expand if necessary

	for (int i=0; i<ex->fv.length; i++)
	{
		fv_buff[fv_write_index+i] = ex->fv.vector[i];
	}

	ex_used[ex_write_index] = E_NOT_USED;
	inc_write_index(ex->fv.length);

	return 1;					// Should check for size and return 0 if insufficient
}

example* ParseBuffer::get_example()
{
	example* ex;
	
	if (ex_read_index >= 0)
	{
	  printf("In get_example.. read_index=%d.\n", ex_read_index);
		ex = &ex_buff[ex_read_index];
		//inc_read_index();
		printf("In get_example.. inc_read_index=%d.\n", ex_read_index);
		return ex;
	}
	else
		return NULL;
}

example* ParseBuffer::fetch_example()
{
	example *ex;
	int32_t current_index = ex_read_index;
	//printf("In fetch_example. ex_read_index=%d.\n", ex_read_index);

	pthread_mutex_lock(&ex_in_use_mutex[current_index]);

	//printf("Locked the mutex!\n");
	int32_t read_index=ex_read_index;
	// Because read index will change after get_example
	printf("In fetch_example.. read_index=%d\n", current_index);
	
	if (ex_used[current_index] == E_NOT_USED)
	  {
	    printf("NOT USED!\n");
		ex = get_example();
	  }
	else
	  {
	    printf("NULL!\n");
		ex = NULL;
	  }
	
	pthread_mutex_unlock(&ex_in_use_mutex[current_index]);
	//printf("Fetch_example... ex_read_index=%d.\n", ex_read_index);
	return ex;
}
	
int32_t ParseBuffer::copy_example(example *ex)
{
	// Check this mutex call.. It should probably be locked regardless of ex in use

	int32_t ret;
	int32_t current_index = ex_write_index;
	//printf("locking mutex for index: %d\n", current_index);

	pthread_mutex_lock(&ex_in_use_mutex[current_index]);
	printf("mutex locked for index: %d\n", current_index);
	while (ex_used[ex_write_index] == E_NOT_USED)
	{
	  //printf("Waiting for example to be USED...\n");
		pthread_cond_wait(&ex_in_use_cond[ex_write_index], &ex_in_use_mutex[ex_write_index]);
	}
	
	ret=write_example(ex);

	if (ex_read_index < 0)
		ex_read_index = 0;
	
	pthread_mutex_unlock(&ex_in_use_mutex[current_index]);
	//printf("unlocked mutex for index:%d\n", current_index);
	return ret;
}

void ParseBuffer::finalize_example()
{
  printf("finalizing ex for index: %d..\n", ex_read_index);
	pthread_mutex_lock(&ex_in_use_mutex[ex_read_index]);
	ex_used[ex_read_index] = E_USED;
	pthread_cond_signal(&ex_in_use_cond[ex_read_index]);
	pthread_mutex_unlock(&ex_in_use_mutex[ex_read_index]);
	
	printf("finalized ex for index: %d.. \n", ex_read_index);
	inc_read_index();

}
