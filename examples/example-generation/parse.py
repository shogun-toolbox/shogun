import sys
import ast
import pyparsing as pp
import json
# Memoization speed boost
pp.ParserElement.enablePackrat()
pp.ParserElement.setDefaultWhitespaceChars(' \t')

"""
grammar (mixed BNF and regex notation)
--------------
identifier -> [a-zA-Z][a-zA-Z0-9\-_]
numeral -> (integers | decimals)  no leading zeros. tracing zeros allowed
stringLiteral -> ".*" 
boolLiteral -> True | False

basicType -> int | bool | float | string | RealVector
type -> Shogun type | basicType

argumentList -> expr | expr, argumentList
methodCall -> identifier . identifier '(' argumentList | empty ')'
expr -> stringLiteral | boolLiteral | numeral | methodCall | identifier

assignment -> identifier = expr
initialisation -> type identifier ('(' argumentList | epsilon ')' | (= expr))

statement -> (initialisation | assignment | expr)

grammar -> statement*
"""

def grammar():
    identifier = pp.Word(initChars=pp.alphas, bodyChars=pp.alphanums+'_-')
    numeral = pp.Regex('([1-9][0-9]*(\.[0-9]*)?)|0\.[0-9]*')
    stringLiteral = pp.QuotedString('"')
    boolLiteral = pp.Literal('True') ^ pp.Literal('False')
    
    basicType = pp.Literal('int') ^ pp.Literal('bool') ^ pp.Literal('float') ^ pp.Literal('string') ^ pp.Literal('RealVector')
    objectType = shogunType()
    type = objectType ^ basicType

    expr = pp.Forward()
    argumentList = pp.Forward()
    argumentList << (expr ^ (expr + ',' + argumentList))
    methodCall = identifier + '.' + identifier + '(' + (argumentList ^ pp.empty) + ')'
    expr << (stringLiteral ^ boolLiteral ^ numeral ^ methodCall ^ identifier)
    
    assignment = identifier + '=' + expr
    initialisation = type + identifier + (('(' + (argumentList ^ pp.empty) + ')') ^ ('=' + expr))

    statement = ((initialisation ^ assignment ^ expr) + pp.lineEnd) ^ pp.lineEnd

    grammar = pp.Forward()
    grammar << pp.ZeroOrMore(statement)

    # Connect grammar to ast data structure
    grammar.setParseAction(ast.Program)
    statement.setParseAction(ast.Statement)
    initialisation.setParseAction(ast.Init)
    assignment.setParseAction(ast.Assign)
    expr.setParseAction(ast.Expr)
    methodCall.setParseAction(ast.MethodCall)
    numeral.setParseAction(ast.NumberLiteral)
    stringLiteral.setParseAction(ast.StringLiteral)
    boolLiteral.setParseAction(ast.BoolLiteral)
    basicType.setParseAction(ast.BasicType)
    objectType.setParseAction(ast.ObjectType)
    identifier.setParseAction(ast.Identifier)
    argumentList.setParseAction(ast.ArgumentList)

    return grammar

def shogunType():
    type = pp.Forward()
    with open("types/typelist", "r") as typelist:
        for line in typelist:
            # Make sure the line is not an empty string
            if line.replace('\n', '') != '':
                type = type ^ pp.Literal(line.replace('\n', ''))

    return type

if __name__ == "__main__":
    program = grammar().parseFile(sys.argv[1], parseAll=True)[0]
    print json.dumps(program, cls=ast.JSONEncoder, indent=2)
