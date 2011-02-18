#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
// #include "cv.h"
// #include "highgui.h"
#include <ctime>
#include <cstdlib>
#include <limits>

#define PI 3.14519265

using namespace std;

struct IPoint
{
  double x;
  double y;
  double s;
};

struct P
{
  short w;
  short h;
  
  short col;
  short row;

  short r;
  short g;
  short b;
};

struct GS
{
  short col;
  short row;

  short intensity;
};

class VehicleRecognition
{
public:
  double classifyImage(vector<int> image);
  void train();
};


class ImageUtils
{
public:
  static void init_arr(const string& dir, vector<vector<P> >& arr, int num);
  static void readImg(const string& dir, int imgNum, vector<P>& img);
  static vector<GS> grayScaleIt(const vector<P>& color);
};

class MathUtils
{
public:
  static double det3x3(const vector<vector<double> >& mat);
  static void adjugate3x3(const vector<vector<double> >& mat, vector<vector<double> >& out);
  static void inverse3x3(const vector<vector<double> >& mat, vector<vector<double> >& out);
};
//find patches of interest (features)
class Detector
{
public:
  //just a fancy stack of scales
  struct OctaveLayer
  {
    vector<vector<double> > responseMap;
    double scale;
    int filterSize;
  };

  struct Octave
  {
    OctaveLayer layer[3];
  };

  vector<IPoint> getFeaturePatches(const vector<vector<long> >& integral, int W, int H, double threshold) const;
  static long calcDyy(const vector<vector<long> >& img, int i, int j, int sx, int sy);
  static long calcDxy(const vector<vector<long> >& img, int i, int j, int sx, int sy);
  static long calcDxx(const vector<vector<long> >& img, int i, int j, int sx, int sy);

  static bool isLocalMax(const Octave& o, int max, unsigned int i, 
			 unsigned int j, unsigned int k, unsigned int l);
  static IPoint interpolateMax(const Octave& o, const IPoint& max, unsigned int i, 
			     unsigned int j, unsigned int k, unsigned int l);

  

  static void initOctave(Octave& o, int W, int H); 
};

//convert patches into numerical vectors
class Descriptor
{
public:
   //2D gaussian of two independant variables x y who both have var = sigma
   //and mean  = 0 
   //cov matrix is therefore [1/sigma, 0; 0 1/sigma] and D = 2
   //therefore we get
  static double gaussian(double x, double y, double sigma)
  {
    return (1.0 / (2.0*PI*sigma*sigma)) * exp( -(x*x + y*y) / (2.0*sigma*sigma));        
  }      
   
  static vector<double> getDescriptor(const IPoint& p, const vector<vector<long> >& integralImage);

  static double haarX(int x, int y, int width, const vector<vector<long> >& integralImage)
  {
     int imgHeight = integralImage.size() - 1;    
     int imgWidth = integralImage[0].size() - 1;
     
     int l_x = min(imgWidth, max(0, x - width/2 - 1));
     int r_x = max(0, min(imgWidth, x + width/2));
     int u_y = min(imgHeight, max(0, y - width/2 - 1));
     int b_y = max(0, min(imgHeight, y + width/2));
     int m_y = min(max(0, y-1), imgHeight);
     
     long sum = 0;
     sum += integralImage[b_y][r_x] + integralImage[u_y][r_x] + 2*integralImage[m_y][l_x];
     sum += -2*integralImage[m_y][r_x] - integralImage[b_y][l_x] - integralImage[u_y][l_x];
     
     return sum;
  }
  
  static double haarY(int x, int y, int width, const vector<vector<long> >& integralImage)
  {
     int imgHeight = integralImage.size() - 1;
     int imgWidth = integralImage[0].size() - 1;

     int l_x = min(imgWidth, max(0, x - width/2 - 1));
     int r_x = max(0, min(imgWidth, x + width/2));
     int u_y = min(imgHeight, max(0, y - width/2 - 1));
     int b_y = max(0, min(imgHeight, y + width/2));
     
     int m_x = min(max(0, x-1),imgWidth);
   
     long sum = 0;
     sum += integralImage[b_y][r_x] - integralImage[u_y][r_x] - 2*integralImage[b_y][m_x];
     sum += 2*integralImage[u_y][m_x] + integralImage[b_y][l_x] - integralImage[u_y][l_x];
     
     return sum;
  }
};

vector<double> Descriptor::getDescriptor(const IPoint& p, const vector<vector<long> >& integralImage)
{
  double x_c = p.x;
  double y_c = p.y;
  double s = p.s;
  double iHeight = static_cast<double>(integralImage.size()) - 1;
  double iWidth = static_cast<double>(integralImage[0].size()) - 1;

  int window_lx = static_cast<int>( floor(max(x_c - 10*s, 0.0)) );
//  int window_rx = static_cast<int>( ceil(min(x_c + 10*s, iWidth)) );
  int window_uy = static_cast<int>( floor(max(y_c - 10*s, 0.0)) );
//int window_by = static_cast<int>( ceil(min(y_c + 10*s, iHeight)) );

int step = static_cast<int>( round(5.0*s) );
  int substep = static_cast<int>( floor(s) );
  //25 in each subregion

  vector<double> descriptor;
int count =0;


  for(int i=0; i < 4; i++)
    {
      for(int j=0; j < 4; j++)
	{
	  double vector_dx = 0;
	  double vector_dy = 0;
	  double vector_abs_dx = 0;
	  double vector_abs_dy = 0;
	  bool showStuff = false;
	  count++;
	  for(int k = 0; k<5; k++)
	    {
	      for(int l = 0; l<5; l++)
		{
		  int start_x = window_lx+j*step + l*substep;
		  int start_y = window_uy+i*step + k*substep;

		  //3.3 sigma gaussian centered at Ipoint
		  double blur = Descriptor::gaussian(start_x - x_c, 
						     start_y - y_c,
						     3.3*s);

		  
		  double waveResp = blur * Descriptor::haarX(start_x, 
							    start_y,
							    2*s,
							    integralImage);
		  
		  vector_dx += waveResp;
		  vector_abs_dx += abs(waveResp);

		  waveResp = blur * Descriptor::haarY(start_x, 
						     start_y,
						     2*s,
						     integralImage);
		  vector_dy += waveResp;
		  vector_abs_dy += abs(waveResp);
		}
	      
	    }
	  //normalize the vector to make it contrast invariant
	  double norm = vector_dx*vector_dx + vector_dy*vector_dy;
	  norm += vector_abs_dx*vector_abs_dx + vector_abs_dy*vector_abs_dy;
	  norm = sqrt(norm);

	  descriptor.push_back(vector_dx / norm);
	  descriptor.push_back(vector_dy / norm);
	  descriptor.push_back(vector_abs_dx / norm);
	  descriptor.push_back(vector_abs_dy / norm);
	}
    }

  //the order in the descriptor vector 
  //will be unique since it will always be from left to right top to bottom
  return descriptor;
}

//bag of words representation of an image
//similar features are summarized
//and the image is represented as a codebook
//where the codewords are the the similar descriptors + their counts
class BoW
{
public:
  void crossValid(const vector<vector<double> >&kcenters, const string& back_file, 
		  const string& veh_file)
  {

  }

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

  static vector<int> getCodebook(const vector<vector<double> >&data, const vector<vector<double> >& kcenters)
  {
    vector<int> codebook(data.size());
    
    for(int k=0;k<data.size();k++)
      {
	double minDist = numeric_limits<double>::max();
	for(int i=0;i<kcenters.size();i++)
	  {                
	    double dist = BoW::euclidDist(data[k], kcenters[i]);
            
	    if(dist < minDist)
	      {
		codebook[k] = i;               
		minDist = dist;
	      }
	      
	  }
      }

    vector<int> histogram(kcenters.size());
    for(int i=0;i<codebook.size();i++)
      {                
	histogram[codebook[i]]++;
      }
    return histogram;
  }
};

void BoW::kmeans(int K, const vector<vector<double> >&data, vector<vector<double> >&kcenters,
		 vector<int>& kcounts)
{
  vector<vector<double> >old_kcenters(kcenters.size());
  srand(time(NULL));
  
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
  //   {
  //     for(int j=0;j<kcenters[i].size();j++)
  // 	cout << kcenters[i][j] << " ";
  //   }

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
            
	      if(dist < minDist)
		{
		  codebook[k] = i;               
		  minDist = dist;
		}
	      
	    }
	}
      
      for(int i=0;i<kcenters.size();i++)
	{
          for(int j=0; j<kcenters[i].size();j++)
	    {
              old_kcenters[i][j] = kcenters[i][j];
              kcenters[i][j] = 0;
              kcounts[i] = 0;
	    }
	}
      
      for(int i=0;i<data.size();i++)
	{
          for(int j=0; j<data[i].size();j++)
	    {
	      kcenters[codebook[i]][j] += data[i][j];
	    }
          kcounts[codebook[i]]++;
	}
      
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
    } while(iter < MAXITER && maxMove > 0.001);
}

inline double MathUtils::det3x3(const vector<vector<double> >& mat)
{
  double m1 = mat[0][0]*(mat[1][1]*mat[2][2] - mat[2][1]*mat[1][2]);
  double m2 = -mat[0][1]*(mat[1][0]*mat[2][2] - mat[2][0]*mat[1][2]);
  double m3 = mat[0][2]*(mat[1][0]*mat[2][1] - mat[2][0]*mat[1][1]);

  return m1+m2+m3;
}

inline void MathUtils::adjugate3x3(const vector<vector<double> >& mat, vector<vector<double> >& out)
{
  //minora coefficients of mat transposed 
  //with {+ - +} {- + -} {+ - +} applied
  out[0][0] = mat[1][1]*mat[2][2] - mat[1][2]*mat[2][1];
  out[0][1] = mat[0][2]*mat[2][1] - mat[0][1]*mat[2][2];
  out[0][2] = mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1];

  out[1][0] = mat[1][2]*mat[2][0] - mat[1][0]*mat[2][2];
  out[1][1] = mat[0][0]*mat[2][2] - mat[0][2]*mat[2][0];
  out[1][2] = mat[0][2]*mat[1][0] - mat[0][0]*mat[1][2];

  out[2][0] = mat[1][0]*mat[2][1] - mat[1][1]*mat[2][0];
  out[2][1] = mat[0][1]*mat[2][0] - mat[0][0]*mat[2][1];
  out[2][2] = mat[0][0]*mat[1][1] - mat[1][0]*mat[0][1];
}

inline void MathUtils::inverse3x3(const vector<vector<double> >& mat, vector<vector<double> >& out)
{
  double fac = 1.0 / MathUtils::det3x3(mat);

  MathUtils::adjugate3x3(mat, out);

  for(unsigned int i=0;i<mat.size();i++)
    {
      for(unsigned int j=0;j<mat.size();j++)
	{
	  out[i][j] *= fac;
	}
    }

}

inline void Detector::initOctave(Octave& o, int W, int H)
{
  for(int i=0;i<3;i++)
    for(int k=0;k<H;k++)
      {
	vector<double> v;
	for(int l=0;l<W;l++)
	  {
	    v.push_back(0.0);
	  }
	o.layer[i].responseMap.push_back(v);
	o.layer[i].scale = 0;
	o.layer[i].filterSize = 0;
      }
}

inline long Detector::calcDxx(const vector<vector<long> >& img, int i, int j, int sx, int sy)
{
  //8 points from integral image for Dyy
  //dxx = (i4 - i3 - i2 + i1) - 2*(i6 - i4 - i5 + i2) + (i8 - i6 - i7 + i5);
  long sum = 0;

  if(j - 5 - 3*sx > 0)
    {
      //i1
      sum += img[i-3-sy][j-5-3*sx];
      //i3
      sum -= img[i+2+sy][j-5-3*sx];
    }
	      
  //i2
  sum -= 3*img[i-3-sy][j-2-sx];
  //i4
  sum += 3*img[i+2+sy][j-2-sx];
  //i5
  sum += 3*img[i-3-sy][j+1+sx];

  //i6
  sum -= 3*img[i+2+sy][j+1+sx];
  //i7
  sum -= img[i-3-sy][j+4+3*sx];
  //i8
  sum += img[i+2+sy][j+4+3*sx];


  return sum;
}

inline long Detector::calcDyy(const vector<vector<long> >& img, int i, int j, int sx, int sy)
{
  //8 points from integral image for Dyy
  //dyy = (i4 - i3 - i2 + i1) - 2*(i6 - i5 - i4 + i3) + (i8 - i7 - i6 + i5);
  long sum = 0;

  if(i - 5 - 3*sy > 0)
    {
      //i1
      sum += img[i-5-3*sy][j-3-sx];
      //i2
      sum -= img[i-5-3*sy][j+2+sx];
    }
	      

  //i3
  sum -= 3*img[i-2-sy][j-3-sx];
  //i5
  sum += 3*img[i+1+sy][j-3-sx];
  //i7
  sum -= img[i+4+3*sy][j-3-sx];
  
  //i4
  sum += 3*img[i-2-sy][j+2+sx];
  //i6
  sum -= 3*img[i+1+sy][j+2+sx];
  //i8
  sum += img[i+4+3*sy][j+2+sx];


  return sum;
}

inline long Detector::calcDxy(const vector<vector<long> >& img, int i, int j, int sx, int sy)
{
  //16 points from integral image for Dxy
  //dxy = (i4 - i3 - i2 + i1) - (i8 - i7 - i6 + i5) 
  //dxy += -1*(i12 - i11 - i10 + i9) + (i16 - i15 - i14 + i13);
  long sum = 0;
  //i1
  sum += img[i-4-sy][j-4-sx];
  //i2
  sum -= img[i-4-sy][j-1];
  //i3
  sum -= img[i-1][j-4-sx];
  //i4 -> stays the same with upscaling
  sum += img[i-1][j-1];

  //i5
  sum -= img[i-4-sy][j];
  //i6
  sum += img[i-4-sy][j+3+sx];  
  //i7 -> stays
  sum += img[i-1][j];
  //i8
  sum -= img[i-1][j+3+sx];

  //i9
  sum -= img[i][j-4-sx];
  //i10
  sum += img[i][j-1];
  //i11
  sum += img[i+3+sy][j-4-sx];
  //i12
  sum -= img[i+3+sy][j-1];

  //i13
  sum += img[i][j];
  //i14
  sum -= img[i][j+3+sx];  
  //i15
  sum -= img[i+3+sy][j];
  //i16
  sum += img[i+3+sy][j+3+sx];

  return sum;
}

IPoint Detector::interpolateMax(const Octave& o, const IPoint& max, unsigned int i,
			      unsigned int j, unsigned int k, unsigned int l)
{
  double ds = (o.layer[l+1].responseMap[i][j] - o.layer[l-1].responseMap[i][j]) / 2.0;
  double dx = (o.layer[l].responseMap[i][j+1] - o.layer[l].responseMap[i][j-1]) / 2.0;
  double dy = (o.layer[l].responseMap[i+1][j] - o.layer[l].responseMap[i-1][j]) / 2.0;

  double dxx = o.layer[l].responseMap[i][j+1] - 2*o.layer[l].responseMap[i][j];
  dxx += o.layer[l].responseMap[i][j-1];

  double dxy = (o.layer[l].responseMap[i+1][j+1] - o.layer[l].responseMap[i+1][j-1]);
  dxy += (o.layer[l].responseMap[i-1][j-1] - o.layer[l].responseMap[i-1][j+1]);
  dxy /= 4.0;
  double dxs = (o.layer[l+1].responseMap[i][j+1] - o.layer[l+1].responseMap[i][j-1]);
  dxs += (o.layer[l-1].responseMap[i][j-1] - o.layer[l-1].responseMap[i][j+1]);
  dxs /= 4.0;
  double dys = (o.layer[l+1].responseMap[i+1][j] - o.layer[l+1].responseMap[i-1][j]);
  dys += (o.layer[l-1].responseMap[i-1][j] - o.layer[l-1].responseMap[i+1][j]);
  dys /= 4.0;
  double dss = o.layer[l+1].responseMap[i][j] - 2*o.layer[l].responseMap[i][j];
  dss += o.layer[l-1].responseMap[i][j];
  double dyy = o.layer[l].responseMap[i+1][j] - 2*o.layer[l].responseMap[i][j];
  dyy += o.layer[l].responseMap[i-1][j];

  vector<vector<double> >hessian;
  hessian.push_back(vector<double>(3));
  hessian.push_back(vector<double>(3));
  hessian.push_back(vector<double>(3));

  hessian[0][0] = dxx;
  hessian[0][1] = dxy;
  hessian[0][2] = dxs;

  hessian[1][0] = dxy;
  hessian[1][1] = dyy;
  hessian[1][2] = dys;

  hessian[2][0] = dxs;
  hessian[2][1] = dys;
  hessian[2][2] = dss;

  vector<vector<double> >iHess;
  iHess.push_back(vector<double>(3));
  iHess.push_back(vector<double>(3));
  iHess.push_back(vector<double>(3));
  MathUtils::inverse3x3(hessian, iHess);

  //due to taylor expansion these are displacements to the optimum
  double xopt = -1*(iHess[0][0]*dx + iHess[0][1]*dy + iHess[0][2]*ds);
  double yopt = -1*(iHess[1][0]*dx + iHess[1][1]*dy + iHess[1][2]*ds);
  double sopt = -1*(iHess[2][0]*dx + iHess[2][1]*dy + iHess[2][2]*ds);

  IPoint interpolated;
  interpolated.x = max.x;
  interpolated.y = max.y;
  interpolated.s = max.s;
  if(fabs(xopt) < 0.5 && fabs(yopt) < 0.5 && fabs(sopt) < 0.5)
    {
  
      IPoint interpolated;
      interpolated.x = xopt;
      interpolated.y = yopt;
      interpolated.s = sopt;
    }
  return interpolated;
}

bool Detector::isLocalMax(const Octave& o, int max, unsigned int i, 
			  unsigned int j, unsigned int k, unsigned int l)
{
  if(max == 0)
    return false;
  for(int m=-1;m<=1;m++)
    {
      for(int n=-1;n<=1;n++)
	{
	  for(int p=-1;p<=1;p++)
	    {
	      double val = o.layer[l+m].responseMap[i+n][j+p];

	      if(val >= max && !(m == 0 && n==0 && p==0))
		{				  
		  return false;
		}
	    } 
	}
    }
  return true;
}

vector<IPoint> Detector::getFeaturePatches(const vector<vector<long> >& integral, 
					   int W, int H, double threshold) const
{
  Detector::Octave o[4];
  for(int i=0;i<4;i++)
    Detector::initOctave(o[i], W, H);
  double area_squared = 0.;
  
  int filter_sizes[4][4] = {{9,15,21,27},{15,27,39,51},{27,51,75,99},{51,99,147,195}};
  
  for(unsigned int k=0;k<4;k++)
    {
      int border = (filter_sizes[k][3] / 2) + 1;
      for(unsigned int l=0;l<4;l++)
	{
	  for(int i=border;i<H-border;i++)
	    {
	      for(int j=border;j<W-border;j++)
		{
		  int filter_size = filter_sizes[k][l];
		  
		  area_squared = filter_size*filter_size*filter_size*filter_size;
		  int sx = (filter_size / 3) - 3;
		  int sy = sx / 2;
		  
		  double dyy = static_cast<double>(Detector::calcDyy(integral, i, j, sx, sy));
		  double dxy = static_cast<double>(Detector::calcDxy(integral, i, j, sx, sx));
		  double dxx = static_cast<double>(Detector::calcDxx(integral, i, j, sy, sx));
		  //filtered[i][j] = dyy;
		  
		  o[k].layer[l].scale = (1.2 / 9.0) * static_cast<double>(filter_size);
		  o[k].layer[l].filterSize = filter_size;
		  
		  double resp = (dxx*dyy - 0.81*dxy*dxy) / (area_squared);
		  
		  o[k].layer[l].responseMap[i][j] = resp;
		}
	    }
	}
    }
  // for(unsigned int k=0;k<4;k++)
  //   {
  //     for(unsigned int l=0;l<3;l++)
  // 	{
  // 	  for(unsigned int i=0;i<H;i++)
  // 	    {
  // 	      for(unsigned int j=0;j<W;j++)
  // 		{
  // 		  cout << o[k].layer[l].responseMap[i][j] << " ";
  // 		}
  // 	      cout << endl;
  // 	    }
  // 	  cout << "----layer----" << endl;
  // 	}
  //     cout << "****octave****" << endl;
  //   }


  vector<IPoint> result;
  
  //non-maximum suppression
  for(unsigned int i=1;i<H-1;i++)
    {
      for(unsigned int j=1;j<W-1;j++)
	{
  	  for(unsigned int k=0;k<4;k++)
	    {
	      for(unsigned int l=1;l<2;l++)
		{
		  double max = o[k].layer[l].responseMap[i][j];
		  
		  if(max > threshold && Detector::isLocalMax(o[k], max, i, j, k, l))
		    {
		      IPoint maxPoint;
		      maxPoint.y = static_cast<double>(i);
		      maxPoint.x = static_cast<double>(j);
		      maxPoint.s = o[k].layer[l].scale;

		      IPoint interpolated = interpolateMax(o[k], maxPoint, i, j, k, l);

		      result.push_back(interpolated);
		      //result.push_back(maxPoint);
		    }
		}
	    }
	}
    }
  
  // IplImage *grayImg2 = cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);
  // int step = grayImg2->widthStep;
  // uchar* data2 = (uchar*)grayImg2->imageData;

  // for(int i=0;i<H;i++) 
  //   for(int j=0;j<W;j++) 
  // 	{
  // 	  data2[i*step+j]=filtered[i][j];
  // 	}

  // cvShowImage("hello2", grayImg2);

  return result;
}

double VehicleRecognition::classifyImage(vector<int> image)
{
  
  return 0.5;
}
	
void VehicleRecognition::train()
{
  
}


vector<GS> ImageUtils::grayScaleIt(const vector<P>& color)
{
  vector<GS> grayScale;
  grayScale.reserve(color.size());

  for(int i=0;i<color.size();i++)
    {
      GS gs;
      gs.row = color[i].row;
      gs.col = color[i].col;

      //assume input is already gamma corrected
      gs.intensity = 0.30*color[i].r + 0.59*color[i].g + 0.11*color[i].b;

      grayScale.push_back(gs);
    }

  return grayScale;
}

void ImageUtils::init_arr(const string& dir, 
				  vector<vector<P> >& arr, int num)
{
  for(int i=0;i<num;i++)
    {
      arr.push_back( vector<P>() );
      ImageUtils::readImg(dir, i, arr[i]);
    } 
}

void ImageUtils::readImg(const string& dir, int imgNum, vector<P>& destImg)
{
  stringstream ss(stringstream::in | stringstream::out);

  ss << dir << "img" << imgNum << ".raw";
  fstream img(ss.str().c_str(), fstream::in);
  int w = 0;
  int h = 0;

  img >> w >> h;
      
  for(int j=0;j<w;j++)
    {
      for(int k=0;k<h;k++)
	{
	  P point;
	  point.w = w;
	  point.h = h;
	  point.col = j;
	  point.row = k;
	  img >> point.r;
	  img >> point.g;
	  img >> point.b;
	  destImg.push_back(point);
	}
    }
}

// void drawPoints(IplImage *img, vector<IPoint> &ipts)
// {
//   double s, o;
//   int r1, c1;

//   for(unsigned int i = 0; i < ipts.size(); i++) 
//   {
//     s = 3;
//     r1 = (int)floor(ipts[i].y + 0.5);
//     c1 = (int)floor(ipts[i].x + 0.5);

//     cvCircle(img, cvPoint(c1,r1), 3, cvScalar(123,12,14), -1);
//     cvCircle(img, cvPoint(c1,r1), 4, cvScalar(0,111,0), 2);
//   }
// }


int main(int argc, char** argv)
{
  stringstream ss("", stringstream::in | stringstream::out);

  for(int i=1;i < argc;i++)
    ss << argv[i] << " ";

  string testDir = "/home/martin/workspace/train/vehicles_proc/";

  int num = 0;
  double threshold;
  if(argc == 4)
    {
      ss >> num;
      ss >> threshold;
      ss >> testDir;
    }

  
  vector<P> points;
  ImageUtils::readImg(testDir,num,points);

  int height = points[0].h;
  int width = points[0].w;
  int channels = 3;
    
  // IplImage* img = cvLoadImage(oh.c_str());
  
  // IplImage *noCar = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,channels);

  // int step = noCar->widthStep;
  // uchar* data = (uchar*)noCar->imageData;

  // for(int i=0;i<height;i++) 
  //   for(int j=0;j<width;j++) 
  // 	{
  // 	  data[i*step+j*channels]=points[j*height+i].b;
  // 	  data[i*step+j*channels+1]=points[j*height+i].g;
  // 	  data[i*step+j*channels+2]=points[j*height+i].r;
  // 	}

  
  //cvShowImage("hello",noCar);

  vector<GS> gray = ImageUtils::grayScaleIt(points);



  vector<vector<long> > integral;
  vector<vector<int> > gray_intensities;
  
  for(unsigned int i=0;i<height;i++)
    {
      vector<int> v;
      vector<long> i;
      for(int j=0;j<width;j++)
	{
	  v.push_back(0);
	  i.push_back(0L);
	}
      gray_intensities.push_back(v);
      integral.push_back(i);
    }

  for(unsigned int i=0;i<gray.size();i++)
    {
      gray_intensities[gray[i].row][gray[i].col] = gray[i].intensity;
    }

   for(unsigned int i=0;i<height;i++)
    for(unsigned int j=0;j<width;j++)
      {
	if(i == 0 && j == 0)
	  integral[i][j] = gray_intensities[i][j];
	else if(i == 0)
	  integral[i][j] = gray_intensities[i][j] + integral[i][j-1];
	else if(j == 0)
	  integral[i][j] = gray_intensities[i][j] + integral[i-1][j];
	else
	  {
	    integral[i][j] = gray_intensities[i][j];
	    integral[i][j] += integral[i-1][j];
	    integral[i][j] += integral[i][j-1];
	    integral[i][j] -= integral[i-1][j-1];
	  }
      }

  Detector d;
  vector<IPoint> ipoints = d.getFeaturePatches(integral, width, height, threshold);
  //cout << "here " << ipoints.size() << endl;
  vector<vector<double> >descs;
  for(int i=0;i<ipoints.size();i++)
    {

      //cout << ipoints[i].x << " " << ipoints[i].y << " " << ipoints[i].s << endl;
      vector<double> desc = Descriptor::getDescriptor(ipoints[i], integral);
      descs.push_back(desc);


      // for(int j=0;j<desc.size();j++)
      // 	cout << desc[j] << " ";
      // cout << endl;
    }

  //transfrom into codebook
  vector<vector<double> >kcenters;
  int centers = 64;
  int num_k = 40;
  fstream kc("kcenters.txt", fstream::in);

   for(int i=0;i<num_k;i++)
     {
       vector<double> kcv(centers);
       for(int j=0;j<centers;j++)
	 kc >> kcv[j];
       kcenters.push_back(kcv);
     }

   vector<int> kcounts = BoW::getCodebook(descs, kcenters);

   for(int i=0;i<num_k;i++)
     cout << i << " " << kcounts[i] << " ";
   cout << endl;



  //cout << ipoints.size() << endl;
  //drawPoints(image, ipoints);


  // IplImage *image = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
  // int step = image->widthStep;
  // uchar* data2 = (uchar*)image->imageData;


  // for(int i=0;i<height;i++) 
  //   for(int j=0;j<width;j++) 
  // 	{
  // 	  data2[i*step+j]=gray[j*height+i].intensity;
  // 	}
  // cvShowImage("hello2", image);
  // cvWaitKey(0);

  

  // cvReleaseImage(&noCar);
  // cvDestroyWindow("hello");
  // cvReleaseImage(&image);
  // cvDestroyWindow("hello2");  
}

