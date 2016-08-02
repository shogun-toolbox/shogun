#!/usr/bin/env python

def Tag_params_modular():

    from modshogun import Obj, GaussianKernel, RealVector
    from modshogun import TagKernel, TagInt, TagFloat, TagRealVector
    import sys, random

    obj = Obj()
    int_tag = TagInt("int")
    float_tag = TagFloat("float")
    kernel_tag = TagKernel("gaussian")
    vector_tag = TagRealVector("vector")
    random_tag = TagInt("random")

    try:
        random_val = obj.gets(random_tag)
    except SystemError as err:
        print err

    try:
        random_val = obj.getsKernel("random")
    except SystemError as err:
        print err

    integer = random.randint(-100, 100)
    obj.setsInt("int", integer)
    if integer != obj.getsInt("int") or integer != obj.gets(int_tag):
        sys.exit(1)
    integer = random.randint(-100, 100)
    obj.sets(int_tag, integer)
    if integer != obj.getsInt("int") or integer != obj.gets(int_tag):
        sys.exit(1)

    decimal = random.random()
    obj.setsFloat("float", decimal)
    if decimal != obj.getsFloat("float") or decimal != obj.gets(float_tag):
        sys.exit(1)
    decimal = random.random()
    obj.sets(float_tag, decimal)
    if decimal != obj.getsFloat("float") or decimal != obj.gets(float_tag):
        sys.exit(1)

    kernel = GaussianKernel(5)
    obj.setsKernel("gaussian", kernel)
    if kernel != obj.getsKernel("gaussian") or kernel != obj.gets(kernel_tag):
        sys.exit(1)
    kernel = GaussianKernel(6)
    obj.sets(kernel_tag, kernel)
    if kernel != obj.getsKernel("gaussian") or kernel != obj.gets(kernel_tag):
        sys.exit(1)

    size = random.randint(1, 10)
    real_vector = RealVector(size)
    for i in range(size):
        real_vector[i] = random.random()
    obj.setsRealVector("vector", real_vector)
    if real_vector != obj.getsRealVector("vector") or real_vector != obj.gets(vector_tag):
        sys.exit(1)
    for i in range(size):
        real_vector[i] = random.random()
    obj.sets(vector_tag, real_vector)
    if real_vector != obj.getsRealVector("vector") or real_vector != obj.gets(vector_tag):
        sys.exit(1)

if __name__=='__main__':
    print('Tag-Parameters')
    Tag_params_modular()
