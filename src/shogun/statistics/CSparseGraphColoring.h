/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Mikhail Belous
 */

#ifndef __CSparseGraphColoring_H_
#define __CSparseGraphColoring_H_
#include<vector>
#include<algorithm>
#include<cstdio>


class CSparseGraphColoring 
{
  private:
    std::vector<std::vector<int> > edges;
    std::vector<int> colors;
    std::vector<std::vector<int> > possibleColors;
    std::vector<int> possibleColorsNum;
    bool dfs(int v);
  public:
    CSparseGraphColoring(std::vector<std::vector<int> > _edges){
      edges = _edges;
    }
    std::vector<int> findColoring(int maxColor);
	  bool validateColoring(std::vector<int> colors);
}; 
#endif