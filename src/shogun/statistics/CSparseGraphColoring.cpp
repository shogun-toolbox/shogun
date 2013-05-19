#include<vector>
#include<algorithm>
#include<cstdio>

using namespace std;

class CSparseGraphColoring 
{
  vector<vector<int> > edges;
  CSparseGraphColoring(vector<vector<int> > _edges){
    edges = _edges;
  }
  vector<int> colors;
  vector<vector<int> > possibleColors;
  vector<int> possibleColorsNum;
  bool dfs(int v){
    //Trying to color vectex V
    if(colors[v] != -1)
      return true;

    for(int i = 0; i < (int)possibleColors[v].size();i++)
      if(possibleColors[v][i] == 1)
      {
        colors[v] = i;// Let v vertex be colored in color i;
        for(int j = 0; j < (int)edges[v].size(); j++){
          if((--possibleColors[edges[v][j]][i]) == 0)  //No more related vertex can have this color
            possibleColorsNum[edges[v][j]]--;
        }
        vector<pair<int, int > > next;
        for(int j = 0; j < (int)edges[v].size();j++)
          next.push_back(make_pair(possibleColorsNum[edges[v][j]], edges[v][j]));
        
        //Trying to color related nodes from weakest one to strongest one 
        sort(next.begin(), next.end());

        bool goodColoring = true;
        for(int j = 0; j < (int) next.size() ; j++)
          if(!dfs(next[j].second)){
            goodColoring = false;
            break;
          }
         
        for(int j = 0; j < (int)edges[v].size(); j++){
          if((++possibleColors[edges[v][j]][i]) == 1)  //Cancel this coloring
            possibleColorsNum[edges[v][j]]++;
        }
        if(goodColoring)// No issue found coloring mathes
          return true;
        colors[v] = -1;
      }
    return false;
  }
  vector<int> findColoring(int maxColor){
    colors = vector<int> (edges.size(), -1);
    for(int i = 0; i < (int)edges.size(); i++)
      if(colors[i] == -1){ //Vertex wasn't colored yet
        for(int colorNum = 1; colorNum < maxColor; colorNum++){ //Iterative deaping
          possibleColorsNum = vector<int>(edges.size(), colorNum);
          possibleColors = vector<vector<int> > (edges.size(), vector<int> (colorNum, 1)); // Possibly lots of useless memory
          if(dfs(i)){
            //We found needed coloring
            break;
          }
        }
        if(colors[i] == -1){
          fprintf(stderr,"No coloring found, try biger maxColor\n");
          return colors;
        }
      }
  }
  bool validateColoring(vector<int> colors){
    for(int i = 0; i < (int)edges.size(); i++)
      for(int j = 0; j < (int)edges[i].size(); j++)
        if(colors[i] == colors[edges[i][j]])
          return false;
    return true;
  } 
}; 