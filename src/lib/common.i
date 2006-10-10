%{
 #include "lib/common.h" 
%}

%include "lib/common.h"

%include "cpointer.i"
%pointer_class(int, iptr);
%pointer_class(double, dptr);
%pointer_class(char, cptr);

%include "carrays.i"
%array_class(int, intArray);
%array_class(double, doubleArray);
%array_class(char, charArray);

%typemap(in) char ** {
  /* Check if is a list */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (char **) malloc((size+1)*sizeof(char *));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyString_Check(o))
   $1[i] = PyString_AsString(PyList_GetItem($input,i));
      else {
   PyErr_SetString(PyExc_TypeError,"list must contain strings");
   free($1);
   return NULL;
      }
    }
    $1[i] = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%pythoncode %{

def createDoubleArray(list):
   array = doubleArray(len(list))
   for i in range(len(list)):
      array[i] = list[i]
   return array

def createIntArray(list):
   array = intArray(len(list))
   for i in range(len(list)):
      array[i] = list[i]
   return array

def createCharArray(list):
   array = charArray(len(list))
   for i in range(len(list)):
      array[i] = list[i]
   return array

%}
