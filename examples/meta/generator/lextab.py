# lextab.py. This file automatically created by PLY (version 3.8). Don't edit!
_tabversion   = '3.8'
_lextokens    = set(['SHOGUNTYPE', 'COMMENT', 'PRINTKEYWORD', 'NEWLINE', 'RPAREN', 'BASICTYPE', 'STRINGLITERAL', 'BOOLLITERAL', 'ENUMKEYWORD', 'EQUALS', 'COMMA', 'LPAREN', 'NUMERAL', 'IDENTIFIER', 'DOT'])
_lexreflags   = 0
_lexliterals  = ''
_lexstateinfo = {'INITIAL': 'inclusive'}
_lexstatere   = {'INITIAL': [('(?P<t_IDENTIFIER>[a-zA-Z][a-zA-Z0-9-_]*)|(?P<t_NEWLINE>\\n)|(?P<t_NUMERAL>([1-9][0-9]*(\\.[0-9]+)?)|0\\.[0-9]+)|(?P<t_STRINGLITERAL>"[^"\n]*")|(?P<t_COMMENT>\\#.*\\n)|(?P<t_DOT>\\.)|(?P<t_LPAREN>\\()|(?P<t_RPAREN>\\))|(?P<t_COMMA>,)|(?P<t_EQUALS>=)', [None, ('t_IDENTIFIER', 'IDENTIFIER'), ('t_NEWLINE', 'NEWLINE'), (None, 'NUMERAL'), None, None, (None, 'STRINGLITERAL'), (None, 'COMMENT'), (None, 'DOT'), (None, 'LPAREN'), (None, 'RPAREN'), (None, 'COMMA'), (None, 'EQUALS')])]}
_lexstateignore = {'INITIAL': ' \t'}
_lexstateerrorf = {'INITIAL': 't_error'}
_lexstateeoff = {}
