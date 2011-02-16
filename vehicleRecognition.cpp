#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include "cv.h"
#include "highgui.h"
#include <ctime>

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
  static void grayScaleTriplet(const vector<P>& color, vector<GS>& rg, vector<GS>& gg, vector<GS>& bg);
  static vector<GS> grayScaleItGamma(const vector<P>& color);
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

  vector<IPoint> getFeaturePatches(const vector<GS>& gray, int W, int H, double threshold) const;
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
  calcHaarResponse(const vector<IPoint>& ipoints)
  {
    
    for(int x = -6*s; x<=6*s; x+=s)
      {
	for(int y = -6*s; y<=6*s; y+=s)
	  {


};

//bag of words representation of an image
//similar features are summarized
//and the image is represented as a codebook
//where the codewords are the the similar descriptors + their counts
class BoW
{
public:

};

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

  if(j > 4 + 3*sx)
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

  if(i > 4 + 3*sy)
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

  cout << dxx << " " << dxy << " " << dxs << endl;
  cout << dxy << " " << dyy << " " << dys << endl;
  cout << dxs << " " << dys << " " << dss << endl;

  vector<vector<double> >iHess;
  iHess.push_back(vector<double>(3));
  iHess.push_back(vector<double>(3));
  iHess.push_back(vector<double>(3));
  MathUtils::inverse3x3(hessian, iHess);

  cout << iHess[0][0] << " " << iHess[0][1] << " " << iHess[0][2] << endl;
  cout << iHess[1][0] << " " << iHess[1][1] << " " << iHess[1][2] << endl;
  cout << iHess[2][0] << " " << iHess[2][1] << " " << iHess[2][2] << endl;

  
  //due to taylor expansion these are displacements to the optimum
  double xopt = -1*(iHess[0][0]*dx + iHess[0][1]*dy + iHess[0][2]*ds);
  double yopt = -1*(iHess[1][0]*dx + iHess[1][1]*dy + iHess[1][2]*ds);
  double sopt = -1*(iHess[2][0]*dx + iHess[2][1]*dy + iHess[2][2]*ds);
  cout << xopt << " " << yopt << " " << sopt << endl;
  //if(fabs(xopt) < 0.5 && fabs(yopt) < 0.5 && fabs(sopt) < 0.5)
  IPoint interpolated;
  interpolated.x = max.x + xopt;
  interpolated.y = max.y + yopt;
  interpolated.s = max.s + sopt;
  return interpolated;
}

bool Detector::isLocalMax(const Octave& o, int max, unsigned int i, 
			  unsigned int j, unsigned int k, unsigned int l)
{
  if(max == 0)
    return false;
  for(unsigned int m=-1;m<=1;m++)
    {
      for(unsigned int n=-1;n<=1;n++)
	{
	  for(unsigned int p=-1;p<=1;p++)
	    {
	      double val = o.layer[l+m].responseMap[i+n][j+p];
	      if(val > max)
		{				  
		  return false;
		}
	    } 
	}
    }
  return true;
}

vector<IPoint> Detector::getFeaturePatches(const vector<GS>& gray, int W, int H, double threshold) const
{
  vector<vector<long> > integral;
  vector<vector<int> > grayImg;

  cout << H << " " << W << endl;

  for(unsigned int i=0;i<H;i++)
    {
      vector<int> v;
      vector<long> i;
      for(int j=0;j<W;j++)
	{
	  v.push_back(0);
	  i.push_back(0L);
	}
      grayImg.push_back(v);
      integral.push_back(i);
    }

  for(unsigned int i=0;i<gray.size();i++)
    {
      grayImg[gray[i].row][gray[i].col] = gray[i].intensity;
    }

   for(unsigned int i=0;i<H;i++)
    for(unsigned int j=0;j<W;j++)
      {
	if(i == 0 && j == 0)
	  integral[i][j] = grayImg[i][j];
	else if(i == 0)
	  integral[i][j] = grayImg[i][j] + integral[i][j-1];
	else if(j == 0)
	  integral[i][j] = grayImg[i][j] + integral[i-1][j];
	else
	  {
	    integral[i][j] = grayImg[i][j];
	    integral[i][j] += integral[i-1][j];
	    integral[i][j] += integral[i][j-1];
	    integral[i][j] -= integral[i-1][j-1];
	  }
      }


  Detector::Octave o[4];
  for(int i=0;i<4;i++)
    Detector::initOctave(o[i], W, H);
  double area_squared = 0.;

  int filter_sizes[4][4] = {{9,15,21,27},{15,27,39,51},{27,51,75,99},{51,99,147,195}};

  cout << "building response map" << endl;

  for(unsigned int i=0;i<H;i++)
    {
      for(unsigned int j=0;j<W;j++)
	{
	  for(unsigned int k=0;k<4;k++)
	    {
	      for(unsigned int l=0;l<4;l++)
		{
		  int filter_size = filter_sizes[k][l];
		  int half = filter_size / 2;
		  if(i >= half && i < (H - half) && j >=half && j < (W - half))
		    {
		      area_squared = filter_size*filter_size*filter_size*filter_size;
		      int sx = (filter_size / 3) - 3;
		      int sy = sx / 2;;
		      
		      double dyy = static_cast<double>(Detector::calcDyy(integral, i, j, sx, sy));
		      double dxy = static_cast<double>(Detector::calcDxy(integral, i, j, sx, sx));
		      double dxx = static_cast<double>(Detector::calcDxx(integral, i, j, sy, sx));
		      //filtered[i][j] = dyy;

		      o[k].layer[l].scale = (1.2 / 9.0) * static_cast<double>(filter_size);
		      o[k].layer[l].filterSize = filter_size;

		      double resp = (dxx*dyy - 0.81*dxy*dxy) / (area_squared);
		      
		      if(resp > threshold)
			{
			  cout << threshold << " " << resp << endl;
			  cout << i  << " " << j << " " << o[k].layer[l].scale << endl;
			  o[k].layer[l].responseMap[i][j] = resp;
			}
		      else
			{
			  o[k].layer[l].responseMap[i][j] = 0.0;
			}
		    }
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


  cout << "suppresing and interpolating" << endl;

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
		  
		  if(Detector::isLocalMax(o[k], max, i, j, k, l))
		    {
		      IPoint maxPoint;
		      maxPoint.y = static_cast<double>(i);
		      maxPoint.x = static_cast<double>(j);
		      maxPoint.s = o[k].layer[l].scale;
		      //cout << maxPoint.x << " " << maxPoint.y << " " << maxPoint.s << endl;
		      //IPoint interpolated = interpolateMax(o[k], maxPoint, i, j, k, l);
		      //cout << interpolated.x << " " << interpolated.y << " " << interpolated.s << endl;
		      //result.push_back(interpolated);
		      result.push_back(maxPoint);
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

void ImageUtils::grayScaleTriplet(const vector<P>& color, vector<GS>& rg, vector<GS>& gg, vector<GS>& bg)
{
  for(int i=0;i<color.size();i++)
    {
      GS gs;
      gs.row = color[i].row;
      gs.col = color[i].col;
      
      gs.intensity = color[i].r;
      rg.push_back(gs);
      gs.intensity = color[i].g;
      gg.push_back(gs);
      gs.intensity = color[i].b;
      bg.push_back(gs);
    }
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

vector<GS> ImageUtils::grayScaleItGamma(const vector<P>& color)
{
  vector<GS> grayScale;
  grayScale.reserve(color.size());

  for(int i=0;i<color.size();i++)
    {
      GS gs;
      gs.row = color[i].row;
      gs.col = color[i].col;
      
      //gamma decode
      int re = pow(color[i].r, 1.0/2.2);
      int rg = pow(color[i].g, 1.0/2.2);
      int rb = pow(color[i].b, 1.0/2.2);

      gs.intensity = 0.30*re + 0.59*rg + 0.11*rb;

      //gamma encode
      gs.intensity = pow(gs.intensity, 2.2);
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

void drawPoints(IplImage *img, vector<IPoint> &ipts)
{
  double s, o;
  int r1, c1;

  for(unsigned int i = 0; i < ipts.size(); i++) 
  {
    s = 3;
    r1 = (int)floor(ipts[i].y + 0.5);
    c1 = (int)floor(ipts[i].x + 0.5);

    cvCircle(img, cvPoint(c1,r1), 3, cvScalar(123,12,14), -1);
    cvCircle(img, cvPoint(c1,r1), 4, cvScalar(0,111,0), 2);
  }
}


int main()
{
  string testDir = "/home/martin/workspace/train/vehicles_proc/";
  vector<P> points;
  ImageUtils::readImg(testDir,942,points);

  string oh = "hh.jpg";
  int height = points[0].h;
  int width = points[0].w;
  int channels = 3;
    
  IplImage* img = cvLoadImage(oh.c_str());
  
  IplImage *noCar = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,channels);

  int step = noCar->widthStep;
  uchar* data = (uchar*)noCar->imageData;

  for(int i=0;i<height;i++) 
    for(int j=0;j<width;j++) 
	{
	  data[i*step+j*channels]=points[j*height+i].b;
	  data[i*step+j*channels+1]=points[j*height+i].g;
	  data[i*step+j*channels+2]=points[j*height+i].r;
	}

  
  cvShowImage("hello",noCar);





  vector<GS> test;
  for(int i=0;i<10;i++)
    for(int j=0;j<10;j++)
      {
	GS gs;
	gs.row = i;
	gs.col = j;
	gs.intensity = 1;
	test.push_back(gs);
      }


  IplImage *grayImg = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
  step = grayImg->widthStep;
  uchar* data2 = (uchar*)grayImg->imageData;
  vector<GS> gray = ImageUtils::grayScaleIt(points);

  for(int i=0;i<height;i++) 
    for(int j=0;j<width;j++) 
	{
	  data2[i*step+j]=gray[j*height+i].intensity;
	}

  double threshold = 150;
  Detector d;
  vector<IPoint> ipoints = d.getFeaturePatches(gray, points[0].w, points[0].h, threshold);

  for(int i=0;i<ipoints.size();i++)
    {

            cout << ipoints[i].x << " " << ipoints[i].y << " " << ipoints[i].s << endl;
    }

  drawPoints(grayImg, ipoints);


  cvShowImage("hello2", grayImg);
  cvWaitKey(0);

  

  cvReleaseImage(&noCar);
  cvDestroyWindow("hello");
  cvReleaseImage(&grayImg);
  cvDestroyWindow("hello2");  
}

