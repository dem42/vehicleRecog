#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <limits>


using namespace std;

//bag of words representation of an image
//similar features are summarized
//and the image is represented as a codebook
//where the codewords are the the similar descriptors + their counts
class BoW
{
public:
  static void kmeans(int K, const vector<vector<double> >&data,
               vector<vector<double> >&kcenters, vector<int>& kcounts);
  static double euclidDist(const vector<double>& v1, const vector<double>& v2)
  {
    double sum = 0;
    
    for(int i=0;i<v1.size();i++)
      {
       sum += (v1[i] - v2[i])*(v1[i] - v2[i]);
      }
    
    return sqrt(sum);
  }
};

void BoW::kmeans(int K, const vector<vector<double> >&data, vector<vector<double> >&kcenters,
                  vector<int>& kcounts)
{
  vector<vector<double> >old_kcenters(kcenters.size());
  srand(time(NULL));

  cout << data.size() << endl;

  for(int i=0; i<kcenters.size(); i++)
    {
        int random = rand() % data.size();          
        for(int j=0; j<kcenters[i].size(); j++)
        {
                kcenters[i][j] = data[random][j];
                old_kcenters[i].push_back(kcenters[i][j]);
         }
         kcounts[i] = 0;
    }

  // for(int i=0;i<K;i++)
  //  {
  //          for(int j=0;j<kcenters[i].size();j++)
  //                  cout << kcenters[i][j] << " ";
  //          cout << endl;
  //          cout << "with kcounts = " << kcounts[i] << endl;
  //  }
  //  char c;
  //  //cin >> c; 

  int MAXITER = 10000;
  int iter = 0;

  int codebook[data.size()];
  for(int i=0;i<data.size();i++)
    codebook[i] = 0;

  double maxMove = 0;
  
  do
  {
      iter++;
      maxMove = numeric_limits<double>::min();

      for(int k=0;k<data.size();k++)
      {
         double minDist = numeric_limits<double>::max();
         for(int i=0;i<kcenters.size();i++)
         {                
            double dist = BoW::euclidDist(data[k], kcenters[i]);
            //cout << k << " " << i << " " << dist << " vs " << minDist <<  endl;
            
            if(dist < minDist)
            {
               codebook[k] = i;               
               minDist = dist;
            }

          }
      }
      
      // for(int i=0;i<data.size();i++)
      //         cout << codebook[i] << " ";
      // cout << endl;        
      
      
      //cout << "before remembering old mean" << endl;
      for(int i=0;i<kcenters.size();i++)
      {
          for(int j=0; j<kcenters[i].size();j++)
          {
              old_kcenters[i][j] = kcenters[i][j];
              kcenters[i][j] = 0;
              kcounts[i] = 0;
          }
      }
      
//      cout << "calculating new means for new groups" << endl;
      for(int i=0;i<data.size();i++)
      {
          for(int j=0; j<data[i].size();j++)
          {
            kcenters[codebook[i]][j] += data[i][j];
          }
          kcounts[codebook[i]]++;
      }
      
      //cout << "averaging new means" << endl;
      for(int i=0;i<kcenters.size();i++)
      {
         if(kcounts[i] != 0)
         {     
               for(int j=0;j<kcenters[i].size();j++)
                       kcenters[i][j] /= static_cast<double>(kcounts[i]);
         }
         
         double dist = BoW::euclidDist(kcenters[i], old_kcenters[i]);
         if(dist > maxMove)
         {
              maxMove = dist;
         }
      }      
      cout << maxMove << " <-- max move " << endl;
  } while(iter < MAXITER && maxMove > 0.00001);
   

}

int main()
{
   vector<vector<double> >data;

   int num = 64;
   int num_v = 1020;
   int num_b = 3014;
   fstream v("vehicles.out", fstream::in);

   fstream b("back.out", fstream::in);
   for(int i=0;i<num_v;i++)
     {
       vector<double> car(num);
       for(int j=0;j<num;j++)
	 v >> car[j];
       data.push_back(car);
     }

   for(int i=0;i<num_b;i++)
     {
       vector<double> nocar(num);
       for(int j=0;j<num;j++)
	 b >> nocar[j];
       data.push_back(nocar);
     }
   
   int K = 40;
   vector<vector<double> >kcenters;
   for(int i=0;i<K;i++)
     kcenters.push_back(vector<double>(num));
   
   vector<int>kcounts(K); 
   
   BoW::kmeans(K, data, kcenters, kcounts);
   
   for(int i=0;i<K;i++)
   {
     for(int j=0;j<kcenters[i].size();j++)
       cout << kcenters[i][j] << " ";
     cout << endl;
   }
}
