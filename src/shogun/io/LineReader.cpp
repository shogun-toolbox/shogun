/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <cstdio>

#include <shogun/io/LineReader.h>

using namespace shogun;

int32_t CLineReader::getdelim(char **lineptr, int32_t *n, char delimiter, FILE *stream)
{
	if (lineptr==NULL || n==NULL || stream==NULL)
	{
		SG_SERROR("Invalid arguments");
		return -1;
	}

	if (ferror(stream))
	{
		SG_SERROR("Error reading file");
		return -1;
	}

	if (feof(stream))
		return -1;

	int32_t buffer_size=chunk_size;
	char* buffer=SG_MALLOC(char, buffer_size);
	char* delim_pos = 0;

	int32_t position=0;
	int32_t bytes_read=0;

	while (1)
	{
		bytes_read=fread(buffer+position, sizeof(char), chunk_size, stream);
		if (ferror(stream))
		{
			SG_SERROR("Error reading file");
			return -1;
		}

		// find delimiter in read data
		delim_pos=(char*) memchr(buffer+position, delimiter, bytes_read);
		if (delim_pos)
		{
			position=delim_pos-buffer;
			break;
		}

		position+=bytes_read;
		if (feof(stream))
			break;

		if (position==buffer_size)
		{
			buffer_size*=2;
			SG_REALLOC(char, buffer, position, buffer_size);
		}
	}

	if (position==0)
		return -1;

	fseek(stream, position-bytes_read+1, SEEK_CUR);

	if (*lineptr==NULL || *n<position+1)
	{
		*n=position+1;
		*lineptr=SG_MALLOC(char, *n);
		if (*lineptr==NULL)
		{
			SG_SERROR("Out of Memory");
			return -1;
		}
	}

	memcpy(*lineptr, buffer, position);
	SG_FREE(buffer);

	(*lineptr)[position]='\0'; // NUL-terminate
	return position; // return the number of chars read
}

int32_t CLineReader::getline(char **lineptr, int32_t *n, FILE *stream)
{
	return getdelim(lineptr, n, '\n', stream);
}
