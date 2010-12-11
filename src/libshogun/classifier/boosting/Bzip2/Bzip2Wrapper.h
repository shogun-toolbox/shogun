#ifndef __BZIP2WRAPPER_H
#define __BZIP2WRAPPER_H

#include "bzlib.h"
#include <fstream>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#define BUFLEN 32768

using namespace std;

class Bzip2WrapperReader
{
private:
	FILE* f;
	BZFILE* b;
	int bzerror;
	unsigned int bytesOut, bytesIn;
	char buf[ BUFLEN ];
	int bufPos;
	string delim;
	long int filePos;
	string _fname; 

public:
	/////////////////////////////////////////////////////////////////
	// constructors
	Bzip2WrapperReader(void){ 
		sprintf( buf, ", " );
		setDelim( buf ); 
		bufPos = 0;
		filePos = 0;
	}

	Bzip2WrapperReader( const char* fname ) {
		sprintf( buf, ", " );
		setDelim( buf ); 
		bufPos = 0;
		filePos = 0;

		open( fname );
		_fname = fname;
	}


	/////////////////////////////////////////////////////////////////
	// open&close
	void open( const char* fname ) {
		bufPos = 0;
		filePos = 0;

		f = fopen ( fname, "rb" );
		if ( !f ) {
			cout << "cannot open bz2 file!!!!" << endl;
		}

		b = BZ2_bzReadOpen ( &bzerror, f, 0, 0, NULL, 0 );
		if ( bzerror != BZ_OK ) {
			BZ2_bzReadClose ( &bzerror, b );
			cout << "error code " << bzerror << endl;
		}

		_fname = fname;
	}

	void close( void ) {
		if ( bzerror != BZ_STREAM_END ) {
			BZ2_bzReadClose ( &bzerror, b );
		} else {
			BZ2_bzReadClose ( &bzerror, b );
		}
		if ( f ) fclose( f );
	}

	///////////////////////////////////////////////////////////////
	// read data
	void putback( const char c ) {
		buf[bufPos++] = c;
		filePos--;
	}
	
	friend int getline( Bzip2WrapperReader& bzr, string& str, char delim = 10 ) {
		int i = 0;
		int bzerror = 0;
		char tmpBuf[ 65536 ];
		BZFILE* b = bzr.getBzFilePointer();
		char* buf = bzr.getBuffer();
		int nbuf = 0;

		while ( 1 ) {
			int intChar = (char) bzr.get();
			//nbuf = BZ2_bzRead ( &bzerror, b, tmpBuf, 1 );
			//bzerror = BZ2_bzread( b, tmpBuf, 10 );
			if ( intChar != -1 ) {
				tmpBuf[ i ] = (char) intChar;
				if ( ( tmpBuf[ i ] == delim ) || ( tmpBuf[ i ] == 10 ) ) {
					tmpBuf[i] = '\0';
					break;
				}				
			} else {
				bzr.setError( BZ_STREAM_END );
				tmpBuf[i] = '\0';
				break;
			}

			i++;
		}
		
		str.clear();
		str = tmpBuf;
		
		return 1;
	}

	 
	friend Bzip2WrapperReader &operator>>( Bzip2WrapperReader& bzr, int& data ) {
		string str; 
		if ( bzr.getStringNextDelimiter( str ) ) {
			data = atoi( str.c_str() );
		} else {
			data = 0;
		}

		return bzr;
	}

	friend Bzip2WrapperReader &operator>>( Bzip2WrapperReader& bzr, double& data ) {
		string str; 
		if ( bzr.getStringNextDelimiter( str ) ) {
			data = atof( str.c_str() );
		} else {
			data = 0.0;
		}

		return bzr;
	}

	friend Bzip2WrapperReader &operator>>( Bzip2WrapperReader& bzr, float& data ) {
		string str; 
		if ( bzr.getStringNextDelimiter( str ) ) {
			data = atof( str.c_str() );
		} else {
			data = 0.0;
		}

		return bzr;
	}

	friend Bzip2WrapperReader &operator>>( Bzip2WrapperReader& bzr, string& data ) {
		bzr.getStringNextDelimiter( data );
		return bzr;
	}
	
	int get( void ) {
		int bzerror = 0;
		char tmpBuf[ 8 ];
		int nbuf = 0;
		int retval = -1;
		
		if ( bufPos > 0 ) {
			retval = (int) buf[0];
			for( int i = 0; i < bufPos-1; i++ ) buf[i] = buf[i+1];
			bufPos--;
		} else {
			nbuf = BZ2_bzRead ( &bzerror, b, tmpBuf, 1 );
			if ( bzerror == BZ_OK ) {
				retval = (int) tmpBuf[0];
			} else if ( bzerror == BZ_STREAM_END ) { 
				setError( BZ_STREAM_END );
			} else {
				setError( BZ_STREAM_END );
			}
		}
		filePos++;

		return retval;
	}
	


	int getStringNextDelimiter( string& str ) {
		int i = 0;
		int bzerror = 0;
		char tmpBuf[ 65536 ];
		int nbuf = 0;
		int retval = 1;
		

		while ( 1 ) {
			int intChar = (char) get();
			//nbuf = BZ2_bzRead ( &bzerror, b, tmpBuf, 1 );
			//bzerror = BZ2_bzread( b, tmpBuf, 10 );
			if ( ( intChar == 32 ) && ( i == 0 ) ) continue;
			if ( intChar != -1 ) {
				tmpBuf[ i ] = (char) intChar;
				size_t t = delim.find( tmpBuf[i] );
				if ( t != string::npos ) {
					tmpBuf[i] = '\0';
					break;
				}				
			} else {
				setError( BZ_STREAM_END );
				tmpBuf[i] = '\0';
				break;
			}
			i++;
		}

		str.clear();
		str = tmpBuf;
		
		return retval;
	}

	/////////////////////////////////////////////////////////////////////
	// destructors
	~Bzip2WrapperReader(void);


	bool is_open() {
		if ( b ) return true;
		else return false;
	}

	bool eof() {
		if ( bzerror == BZ_STREAM_END ) return true;
		else return false;
	}


	/////////////////////////////////////////////////////////////////////
	// getters and setters
	char* getBuffer() {
		return buf;
	}

	BZFILE * getBzFilePointer() {
		if ( is_open() ) {
			return b;
		} else {
			return 0;
		}
	}

	void setError( const int code ) {
		bzerror = code;
	}
	
	void setDelim( const string s ) {
		char tmpBuf[1024];
		sprintf( tmpBuf, "%s\n", s.c_str() );
		delim = s;
	}

	void setDelim( const char* s ) {
		char tmpBuf[1024];
		sprintf( tmpBuf, "%s\n", s );
		delim = tmpBuf;
	}
	
	int getBufPos() { return bufPos; }
	void setBufPos( int b ) { bufPos = b; }
	void incrementBufPos() { bufPos++; }

	int getFilePos() { return filePos; }
	void setFilePos( int b ) { filePos = b; }


	int remainingRowNum() {
		int rowCount = 0;
		string tmpString;

		int tmpFilePos = filePos;

		getline( *this, tmpString );	
		while( ! eof() ) {
			rowCount++;
			getline( *this, tmpString );	
		}

		setPos( tmpFilePos );
		//getline( *this, tmpString );
		rowCount++;

		return rowCount;
	}


	void setPos( long int pos ) {
		close();
		open( _fname.c_str() );
		for( int i = 0; i < pos; i++ ) get();
	}
};

class Bzip2WrapperWriter
{
private:
	FILE* f;
	BZFILE* b;
	int bzerror;
	unsigned int bytesOut, bytesIn;

public:
	//////////////////////////////////////////////////////////////////////
	// constructors
	Bzip2WrapperWriter(void);

	Bzip2WrapperWriter( const char * fname, bool append = false ) {
		open( fname, append );
	}


	//////////////////////////////////////////////////////////////////////
	void close( void ) {
		if ( f ) { 
			BZ2_bzWriteClose( &bzerror, b, 0, &bytesOut, &bytesIn );
			if (bzerror == BZ_IO_ERROR) {
				cout << "error while close file" << endl;
			}

			fclose( f );
		}
	}
	//////////////////////////////////////////////////////////////////////

	void open( const char* fname, bool append = false ) {
		if ( append ) {
			const char* tmpFilename = "tmp1234567";
			
			if ( rename( fname, tmpFilename ) == 0 ) {
				string stringBuffer;
				
				
				Bzip2WrapperReader bzr;
				bzr.open( tmpFilename );			
				if ( bzr.is_open() ) {
					cout << "The compressed file is opened!" << endl; 
				}
				
				this->open( fname, false );

				while ( ! bzr.eof() ) {
					getline( bzr, stringBuffer );
					//stringBuffer.append( endl );
					(*this) << stringBuffer; 
					stringBuffer = '\n';
					(*this) << stringBuffer; 
					
					//int tmpVal;
					//bzr >> tmpVal;
					//out << tmpVal << " ";
					
					//out << stringBuffer << endl;
				}

				bzr.close();


				int removeFlag = remove( tmpFilename );
			} else {
				this->open( fname, false );
			}
		} else {
			f = fopen ( fname , "wb" );

			if ( ! f ) {
				std::cout << " handle error " << std::endl;
				exit(-1);	
			}

			b = BZ2_bzWriteOpen( &bzerror, f, 9, 0, 0 );
			if (bzerror != BZ_OK) {
				BZ2_bzWriteClose ( &bzerror, b, 0, &bytesOut, &bytesIn );
			
				std::cout << "cannot open output fiel" << std::endl;
				std::cout << "error code: " << bzerror << std::endl;
			}

		}

	}
	
	//////////////////////////////////////////////////////////////////////
	void writeCharSequence( const char* str ) {
		BZ2_bzWrite ( &bzerror, b, (void*)str, (int )strlen(str) );

		if (bzerror == BZ_IO_ERROR) {
			BZ2_bzWriteClose ( &bzerror, b, 0, &bytesOut, &bytesIn );
			cout << "error while close file" << endl;
		}
	}



	///////////////////////////////////////////////////////////////////////////////
	// operators 
	friend Bzip2WrapperWriter &operator<<( Bzip2WrapperWriter& bzw, string& str ) {
		return bzw << str.c_str();
	}
	

	friend Bzip2WrapperWriter &operator<<( Bzip2WrapperWriter& bzw, const char* str ) {
		bzw.writeCharSequence( str );
		return bzw;
	}

	////////////////////////////////////////////////////////////////////////////////
	// destructor
	~Bzip2WrapperWriter(void);
};




#endif

