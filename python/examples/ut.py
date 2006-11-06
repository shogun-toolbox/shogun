#!/usr/bin/env python

class TestCase:
   """
   This class is a replacement for those people who do not
   have the python unittest package installed.
   """

   def setUp():
      pass

   def tearDown():
      pass

   def __init__(self):  
      self.setUp()

   def __del__(self):
      """
      Note that this doesn't work as thought. After the 
      "del obj" statement the garbage collector may defer
      the call of this method up to any time in the future.
      """
      self.tearDown()
   
   def run(self):
      pass

