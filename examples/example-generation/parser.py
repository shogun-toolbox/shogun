import sys
import parsetree as tree
import pyparsing as pp
import json
# Memoization speed boost
pp.ParserElement.enablePackrat()

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

assignment -> identiger = expr
initialisation -> type identifier ('(' argumentList | epsilon ')' | (= expr))

statement -> (initialisation | assignment | expr) ;

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

    statement = (initialisation ^ assignment ^ expr) + pp.lineEnd

    grammar = pp.Forward()
    grammar << pp.ZeroOrMore(statement)

    # Connect grammar to parse tree data structure
    grammar.setParseAction(tree.Program)
    statement.setParseAction(tree.Statement)
    initialisation.setParseAction(tree.Init)
    assignment.setParseAction(tree.Assign)
    expr.setParseAction(tree.Expr)
    methodCall.setParseAction(tree.MethodCall)
    numeral.setParseAction(tree.NumberLiteral)
    stringLiteral.setParseAction(tree.StringLiteral)
    boolLiteral.setParseAction(tree.BoolLiteral)
    basicType.setParseAction(tree.BasicType)
    objectType.setParseAction(tree.ObjectType)
    identifier.setParseAction(tree.Identifier)
    argumentList.setParseAction(tree.ArgumentList)

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
    print json.dumps(program, cls=tree.JSONEncoder, indent=2)
