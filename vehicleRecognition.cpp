#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include "cv.h"
#include "highgui.h"
#include <ctime>
#include <ext/hash_map>

using namespace std;


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

//find patches of interest (features)
class Detector
{
public:
  vector<int> getFeaturePatches(const vector<GS>& gray, int W, int H) const;
  static int calcDyy(const vector<vector<long> >& img, int i, int j, int sx, int sy);
  static int calcDxy(const vector<vector<long> >& img, int i, int j, int sx, int sy);
  static int calcDxx(const vector<vector<long> >& img, int i, int j, int sx, int sy);

  //just a fancy stack of scales
  struct Octave
  {
    vector<vector<double> > responseMap[3];
  };

  static void initOctave(Octave& o, int W, int H); 
};

//convert patches into numerical vectors
class Descriptor
{
public:

};

//bag of words representation of an image
//similar features are summarized
//and the image is represented as a codebook
//where the codewords are the the similar descriptors + their counts
class BoW
{
public:

};

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
	o.responseMap[i].push_back(v);
      }
}

inline int Detector::calcDxx(const vector<vector<long> >& img, int i, int j, int sx, int sy)
{
  //8 points from integral image for Dyy
  //dxx = (i4 - i3 - i2 + i1) - 2*(i6 - i4 - i5 + i2) + (i8 - i6 - i7 + i5);
  int sum = 0;

  if(j > 4)
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

inline int Detector::calcDyy(const vector<vector<long> >& img, int i, int j, int sx, int sy)
{
  //8 points from integral image for Dyy
  //dyy = (i4 - i3 - i2 + i1) - 2*(i6 - i5 - i4 + i3) + (i8 - i7 - i6 + i5);
  int sum = 0;

  if(i > 4)
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

inline int Detector::calcDxy(const vector<vector<long> >& img, int i, int j, int sx, int sy)
{
  //16 points from integral image for Dxy
  //dxy = (i4 - i3 - i2 + i1) - (i8 - i7 - i6 + i5) 
  //dxy += -1*(i12 - i11 - i10 + i9) + (i16 - i15 - i14 + i13);
  int sum = 0;
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


vector<int> Detector::getFeaturePatches(const vector<GS>& gray, int W, int H) const
{
  vector<vector<long> > integral;
  vector<vector<int> > grayImg;

  cout << H << " " << H << endl;

  for(int i=0;i<H;i++)
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

  for(int i=0;i<gray.size();i++)
    {
      grayImg[gray[i].row][gray[i].col] = gray[i].intensity;
    }

   for(int i=0;i<H;i++)
    for(int j=0;j<W;j++)
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

  // for(int i=0;i<H;i++)
  //   {
  //     for(int j=0;j<W;j++)
  // 	cout << integral[i][j] << " ";
  //     cout << endl;
  //   }

   int SIZE = 9;
   int HALF = SIZE / 2;
   int Dyy_9x9[9][9] = { {0,0,1,1,1,1,1,0,0}, 
		    {0,0,1,1,1,1,1,0,0},
		    {0,0,1,1,1,1,1,0,0},
		    {0,0,-2,-2,-2,-2,-2,0,0},
		    {0,0,-2,-2,-2,-2,-2,0,0},
		    {0,0,-2,-2,-2,-2,-2,0,0},
		    {0,0,1,1,1,1,1,0,0},
		    {0,0,1,1,1,1,1,0,0},
		    {0,0,1,1,1,1,1,0,0}
  };
   int Dxy_9x9[9][9] = { {0,0,0,0,0,0,0,0,0}, 
			 {0,1,1,1,0,-1,-1,-1,0},
			 {0,1,1,1,0,-1,-1,-1,0},
			 {0,1,1,1,0,-1,-1,-1,0},
			 {0,0,0,0,0,0,0,0,0},
			 {0,-1,-1,-1,0,1,1,1,0},
			 {0,-1,-1,-1,0,1,1,1,0},
			 {0,-1,-1,-1,0,1,1,1,0},
			 {0,0,0,0,0,0,0,0,0}
  };
   int Dxx_9x9[9][9] = { {0,0,0,0,0,0,0,0,0},
			 {0,0,0,0,0,0,0,0,0},
			 {1,1,1,-2,-2,-2,1,1,1},
			 {1,1,1,-2,-2,-2,1,1,1},
			 {1,1,1,-2,-2,-2,1,1,1},
			 {1,1,1,-2,-2,-2,1,1,1},
			 {1,1,1,-2,-2,-2,1,1,1},
			 {0,0,0,0,0,0,0,0,0},
			 {0,0,0,0,0,0,0,0,0}
  };

   //   int SIZE = 3;
   //   int HALF = SIZE / 2;
   int lap[3][3] = { {0,1,0}, {1,-4,1}, {0,1,0}};
   double con = 1.0/159.0;
   int gaus[5][5] = {{2,4,5,4,2},{4,9,12,9,4},{5,12,15,12,5},{2,4,5,4,2},{4,9,12,9,4}};

   con = 1.0;
   int sobel[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

  int filtered[H][W];
  int filtered2[H][W];

  //filter
  
  for(int i=0;i<H;i++)
    {
      for(int j=0;j<W;j++)
  	{
  	  filtered2[i][j] = 0;

  	  if(i >= HALF && i < (H - HALF) && j >=HALF && j < (W - HALF))
  	    {
  	      for(int k=0; k < SIZE;k++)
  		for(int l=0; l < SIZE; l++)
  		  {
  		    filtered2[i][j] += Dxx_9x9[k][l] * grayImg[i + k - HALF][j + l - HALF];
  		  }
  	      filtered2[i][j] *= con;
  	    }
  	}
    }

  Detector::Octave o[4];
  for(int i=0;i<4;i++)
    Detector::initOctave(o[i], W, H);
  double area_squared = 9.*9.*9.*9.;

  int areas[4][4] = {{9,15,21,27},{15,27,39,51},{27,51,75,99},{51,99,147,195}};
  __gnu_cxx::hash_map<int, pair<int,int> > displs;

  displs[9] = make_pair(0,0);
  displs[15] = make_pair(1,2);
  displs[21] = make_pair(0,0);

  for(int i=0;i<H;i++)
    {
      for(int j=0;j<W;j++)
	{
	  filtered[i][j] = 0;

  	  if(i >= HALF && i < (H - HALF) && j >=HALF && j < (W - HALF))
	    {
	      double dyy = (double)Detector::calcDxx(integral, i, j, 0, 0);
	      double dxy = (double)Detector::calcDxy(integral, i, j, 0, 0);
	      double dxx = (double)Detector::calcDxx(integral, i, j, 0, 0);
	      //filtered[i][j] = dyy;

	      o[0].responseMap[0][i][j] = (dxx*dyy - 0.81*dxy*dxy) / (area_squared);	      
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

  // IplImage *grayImg3 = cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);
  // step = grayImg3->widthStep;
  // uchar* data3 = (uchar*)grayImg3->imageData;

  // for(int i=0;i<H;i++) 
  //   for(int j=0;j<W;j++) 
  // 	{
  // 	  data3[i*step+j]=filtered2[i][j];
  // 	}

  // cvShowImage("hello4", grayImg3);

  cvWaitKey(0);

  // for(int i=0;i<H;i++)
  //   {
  //     for(int j=0;j<W;j++)
  // 	cout << filtered[i][j] << " ";
  //     cout << endl;
  //   }

  return vector<int>();
}

double VehicleRecognition::classifyImage(vector<int> image)
{
  
  return 0.5;
}
	
void VehicleRecognition::train()
{
  string bdir = "/home/martin/workspace/train/background_proc/";
  string vdir = "/home/martin/workspace/train/vehicles_proc/";


   int bNum = 3015;
   int vNum = 1021;
  // int bNum = 1;
  // int vNum = 0;
  
  vector<vector<P> >backs;
  vector<vector<P> >vehs;
  
  ImageUtils::init_arr(bdir,backs,bNum);
  ImageUtils::init_arr(vdir,vehs,vNum);
  
  // for(int i=0;i<bNum;i++)
  //   {
  //     for(int j=0;j<backs[i].size();j++)
  // 	{
  // 	  cout << backs[i][j].r << " " << backs[i][j].g 
  // 	       << " " << backs[i][j].b << endl;
  // 	}
  //   }
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

int main()
{
  string testDir = "/home/martin/workspace/train/vehicles_proc/";
  vector<P> points;
  ImageUtils::readImg(testDir,2,points);

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


  IplImage *grayImg = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
  step = grayImg->widthStep;
  uchar* data2 = (uchar*)grayImg->imageData;
  vector<GS> gray = ImageUtils::grayScaleIt(points);

  for(int i=0;i<height;i++) 
    for(int j=0;j<width;j++) 
	{
	  data2[i*step+j]=gray[j*height+i].intensity;
	}

  cvShowImage("hello2", grayImg);
  cvWaitKey(0);

  

  cvReleaseImage(&noCar);
  cvDestroyWindow("hello");
  cvReleaseImage(&grayImg);
  cvDestroyWindow("hello2");  


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

  Detector d;
  d.getFeaturePatches(gray, points[0].w, points[0].h);
}
