#ifdef HAVE_MINDY
%{
 #include <shogun/features/MindyGramFeatures.h>
%}

%rename(MindyGramFeatures) CMindyGramFeatures;

%include <shogun/features/MindyGramFeatures.h>

%template(import_from_char) CMindyGramFeatures::import_features<char>;
%template(import_from_byte) CMindyGramFeatures::import_features<uint8_t>;
#endif
