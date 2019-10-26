#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <functional>   // std::greater
#include <algorithm>    // std::sort
#include <windows.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;

typedef Vec<float,9> Vec9f;

struct dotLenght
{
	Point2f dot;
	float lenghtToLine;
	DMatch match;
};
struct LineDot
{
	Vec4i line;
	vector<dotLenght> dot;
	float minLenght;
	float maxLenght;
};

double angle(Point2f begin,Point2f a_end);
bool checkDotInLine(Point2f lineDotA,Point2f lineDotB,Point2f checkDot);
vector<Vec4i> findLines(Mat& imag, double rho, double theta, int threshold, double minLineLength, double maxLineGap);
float dotToLineLenght(Point2f lineDotA,Point2f lineDotB,Point2f checkDot);
vector<Vec9f> findSimilarLinesAndAngleBetweenThem(vector<LineDot> leftImgLine, vector<LineDot> rightImgLine);
void findLinesAndTheirDot(const vector<Vec4i>& linesArray, vector<LineDot>& outLineArray, int cols, int rows,const vector< DMatch >& matches, const vector<Point2f>& keypoint, float maxDistToLine);
float dotToDotLenght(Point2f dotA, Point2f dotB);
void clearDublicateMatches(vector< DMatch >& matches);
void rotation(float angle, Mat& imag);
void brightnessNormalizationEqualizeHist(const Mat& inputImg,Mat& outputImg, int type);
//void findJointLeftRightROI(vector<Point2i> left);

int main( int argc, char** argv )
{
	Mat img_left = imread( "images/1/left.jpg", CV_LOAD_IMAGE_COLOR );
	Mat img_right = imread( "images/1/right.jpg", CV_LOAD_IMAGE_COLOR ); 
    if( !img_left.data || !img_right.data ) // Проверка наличия информации в матрице изображения
    { 
		cout << "No data" << endl;
		return -1;
    }

	brightnessNormalizationEqualizeHist(img_left,img_left,2);
	brightnessNormalizationEqualizeHist(img_right,img_right,2);

	//-- Этап 1. Нахождение ключевых точек.
    int minHessian = 500;
	SurfFeatureDetector detector( minHessian,4,16,true,false);
    std::vector<KeyPoint> keypoints_left, keypoints_rigth;
	
    detector.detect( img_left, keypoints_left );
    detector.detect( img_right, keypoints_rigth );
	//-- Этап 2. Вычисление дескрипторов.
    SurfDescriptorExtractor extractor;
    Mat descriptors_left, descriptors_right;
    extractor.compute( img_left, keypoints_left, descriptors_left );
	extractor.compute( img_right, keypoints_rigth, descriptors_right );
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
	if ( descriptors_left.empty() )
		cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
	if ( descriptors_right.empty() )
		cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);
    matcher.match( descriptors_left, descriptors_right, matches );
	cout << "Count matches " << matches.size() << endl;

    Mat left_right_img,img_matches, img_line;
	vector< DMatch > m;
    //-- Нарисовать хорошие матчи
	drawMatches( img_left, keypoints_left, img_right, keypoints_rigth, m, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	img_matches.copyTo(left_right_img);
	img_matches.copyTo(img_line);
	vector<Point2f> left;
	vector<Point2f> right;
	double max_dist = 0; 
	double min_dist = 100;
	DMatch bestMatch;
	for( int i = 0; i < matches.size(); i++ )
    {
		double dist = matches[i].distance;
		if( dist < min_dist ) { min_dist = dist; bestMatch = matches[i];}
        if( dist > max_dist ) max_dist = dist;
		left.push_back( keypoints_left[ matches[i].queryIdx ].pt );
		right.push_back( keypoints_rigth[ matches[i].trainIdx ].pt );
    }
	printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
	circle(left_right_img,keypoints_left[bestMatch.queryIdx].pt,1,Scalar(50,255,80));
	circle(left_right_img,keypoints_rigth[bestMatch.trainIdx].pt + Point2f(img_left.cols,0),1,Scalar(50,255,80));
	double rho = 1;
	double theta = 180;
	int threshold = 80;
	double minLineLength = 50;
	double maxLineGap = 1;
	
	vector<Vec4i> leftLines = findLines(img_left, rho, theta, threshold, minLineLength, maxLineGap);
	cout << "count left imeg lines " << leftLines.size() << endl;

	vector<Vec4i> rightLines = findLines(img_right, rho, theta, threshold, minLineLength, maxLineGap);
	cout << "count right imeg lines " << rightLines.size() << endl;

	float maxDistToLine = 0.2;

	vector<LineDot> leftLineDot;
	findLinesAndTheirDot(leftLines, leftLineDot, img_left.cols, img_left.rows, matches, left, maxDistToLine);
	cout << "count left line with more two dot " << leftLineDot.size() << endl;
	for (int i = 0; i < leftLineDot.size(); i++)
	{
		line(img_line,Point2i(leftLineDot[i].line[0],leftLineDot[i].line[1]),Point2i(leftLineDot[i].line[2],leftLineDot[i].line[3]),Scalar(255,255,0));
	}

	vector<LineDot> rightLineDot;
	findLinesAndTheirDot(rightLines, rightLineDot, img_right.cols, img_right.rows, matches, right, maxDistToLine);
	cout << "count right line with more two dot " << rightLineDot.size() << endl;
	for (int i = 0; i < rightLineDot.size(); i++)
	{
		line(img_line,Point2i(rightLineDot[i].line[0],rightLineDot[i].line[1]) + Point2i(img_left.cols,0),Point2i(rightLineDot[i].line[2],rightLineDot[i].line[3])  + Point2i(img_left.cols,0),Scalar(255,255,0));
	}
	vector<Vec9f> result = findSimilarLinesAndAngleBetweenThem(leftLineDot,rightLineDot);
	cout << "count similar line " << result.size() << endl;
	for (int i = 0; i < result.size(); i++)
	{
		line(img_line,Point2i(result[i][0],result[i][1]),Point2i(result[i][2],result[i][3]),Scalar(80,255,255));
		line(img_line,Point2i(result[i][4],result[i][5]) + Point2i(img_left.cols,0),Point2i(result[i][6],result[i][7]) + Point2i(img_left.cols,0),Scalar(80,255,255));
		cout << "angle[" << i <<"] " << result[i][8] << endl;
		cout << "line[" << i <<"] " << result[i] << endl;
	}
	
	drawMatches( img_left, keypoints_left, img_right, keypoints_rigth, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	imshow("show img_matches before filter",img_matches);

	float angleOfRotation;
	if (result.size() > 0)
		angleOfRotation = result[0][8];
	else
	{
		waitKey(0);
		cout << "Haven't Similar Lines" << endl;
		system("pause");
		exit(-1);
	}

//****************************************************************************************************************************************************
//****************************************************************************************************************************************************
//****************************************************************************************************************************************************

	clearDublicateMatches(matches);
	cout << "Count matches after clean duplicate " << matches.size() << endl;
	vector<DMatch> filteredMatches;
	double thresholdSelectionAngle = 1;
	for (int i = 0; i < matches.size(); i++)
	{
		double a = angle(keypoints_left[bestMatch.queryIdx].pt,keypoints_left[matches[i].queryIdx].pt);
		double b = angle(keypoints_rigth[bestMatch.trainIdx].pt,keypoints_rigth[matches[i].trainIdx].pt);
		double resAngle = abs(b - a - angleOfRotation);
		if ( resAngle <  thresholdSelectionAngle)
		{
			filteredMatches.push_back(matches[i]);
		}
	}
	cout << "Count filtered matches " << filteredMatches.size() << endl;

	DMatch minFilteredMatche;
	DMatch minDisparityMatch;
	DMatch maxDisparityMatch;
	int max_disparity = 0;
	int min_disparity = img_right.cols;
	double minThresholdSelection = thresholdSelectionAngle;

	for (int i = 0; i < filteredMatches.size(); i++)
	{
		double temp_disparity_y = (int)std::abs(keypoints_left[filteredMatches[i].queryIdx].pt.y - keypoints_rigth[filteredMatches[i].trainIdx].pt.y);
		double temp_disparity_x = (int)std::abs(keypoints_left[filteredMatches[i].queryIdx].pt.x - keypoints_rigth[filteredMatches[i].trainIdx].pt.x);
		double temp_disparity = sqrt(pow(temp_disparity_y,2) + pow(temp_disparity_x,2));
		if ( max_disparity < temp_disparity)
		{
			max_disparity = temp_disparity;
			maxDisparityMatch = filteredMatches[i];
		}
		if ( min_disparity > temp_disparity)
		{
			min_disparity = temp_disparity;
			minDisparityMatch = filteredMatches[i];
		}

		if ((filteredMatches[i].queryIdx != bestMatch.queryIdx) && (filteredMatches[i].trainIdx != bestMatch.trainIdx))
		{
			double a = angle(keypoints_left[bestMatch.queryIdx].pt,keypoints_left[matches[i].queryIdx].pt);
			double b = angle(keypoints_rigth[bestMatch.trainIdx].pt,keypoints_rigth[matches[i].trainIdx].pt);
			double resAngle = abs(b - a - angleOfRotation);
			if (resAngle < minThresholdSelection)
			{
				minThresholdSelection = resAngle;
				minFilteredMatche = matches[i];
			}
		}
	}
	
	circle(img_line, Point2f(img_left.cols,0) + keypoints_rigth[minDisparityMatch.trainIdx].pt,6,Scalar(255,100,255));
	circle(img_line, Point2f(img_left.cols,0) + keypoints_rigth[maxDisparityMatch.trainIdx].pt,9,Scalar(40,255,255));
	
	circle(img_line, keypoints_left[minDisparityMatch.queryIdx].pt,6,Scalar(255,100,255));
	circle(img_line, keypoints_left[maxDisparityMatch.queryIdx].pt,9,Scalar(40,255,255));
	
	max_disparity++;

	cout << "Max disparity " << max_disparity << endl;
	cout << "Min disparity " << min_disparity << endl;

	vector<DMatch> jointROIDMatch(4); 
	float minX = img_left.cols;
	float minY = img_left.rows;
	float maxX = 0;
	float maxY = 0;
	for (int i = 0; i < filteredMatches.size(); i++)
	{
		if (keypoints_left[filteredMatches[i].queryIdx].pt.x < minX)
		{
			minX = keypoints_left[filteredMatches[i].queryIdx].pt.x;
			jointROIDMatch[0] = filteredMatches[i]; 
		}
		if (keypoints_left[filteredMatches[i].queryIdx].pt.y < minY)
		{
			minY = keypoints_left[filteredMatches[i].queryIdx].pt.y;
			jointROIDMatch[1] = filteredMatches[i]; 
		}
		if (keypoints_left[filteredMatches[i].queryIdx].pt.x > maxX)
		{
			maxX = keypoints_left[filteredMatches[i].queryIdx].pt.x;
			jointROIDMatch[2] = filteredMatches[i]; 
		}
		if (keypoints_left[filteredMatches[i].queryIdx].pt.y > maxY)
		{
			maxY = keypoints_left[filteredMatches[i].queryIdx].pt.y;
			jointROIDMatch[3] = filteredMatches[i]; 
		}
	}

	vector<Point2i> leftROI(4);
	vector<Point2i> rightROI(4);

	circle(img_line,keypoints_left[jointROIDMatch[0].queryIdx].pt,2,Scalar(40,100,255));
	circle(img_line,keypoints_left[jointROIDMatch[1].queryIdx].pt,2,Scalar(40,100,255));
	circle(img_line,keypoints_left[jointROIDMatch[2].queryIdx].pt,2,Scalar(40,100,255));
	circle(img_line,keypoints_left[jointROIDMatch[3].queryIdx].pt,2,Scalar(40,100,255));

	leftROI[0].x = keypoints_left[jointROIDMatch[0].queryIdx].pt.x;
	leftROI[0].y = keypoints_left[jointROIDMatch[1].queryIdx].pt.y;

	leftROI[1].x = keypoints_left[jointROIDMatch[2].queryIdx].pt.x;
	leftROI[1].y = keypoints_left[jointROIDMatch[1].queryIdx].pt.y;

	leftROI[2].x = keypoints_left[jointROIDMatch[2].queryIdx].pt.x;
	leftROI[2].y = keypoints_left[jointROIDMatch[3].queryIdx].pt.y;

	leftROI[3].x = keypoints_left[jointROIDMatch[0].queryIdx].pt.x;
	leftROI[3].y = keypoints_left[jointROIDMatch[3].queryIdx].pt.y;

	line(img_line, leftROI[0],leftROI[1],Scalar(40,255,255));
	line(img_line, leftROI[1],leftROI[2],Scalar(80,255,255));
	line(img_line, leftROI[2],leftROI[3],Scalar(120,255,255));
	line(img_line, leftROI[0],leftROI[3],Scalar(160,255,255));
	
	circle(img_line, Point2f(img_left.cols,0) + keypoints_rigth[jointROIDMatch[0].trainIdx].pt,2,Scalar(40,100,255));
	circle(img_line, Point2f(img_left.cols,0) + keypoints_rigth[jointROIDMatch[1].trainIdx].pt,2,Scalar(40,100,255));
	circle(img_line, Point2f(img_left.cols,0) + keypoints_rigth[jointROIDMatch[2].trainIdx].pt,2,Scalar(40,100,255));
	circle(img_line, Point2f(img_left.cols,0) + keypoints_rigth[jointROIDMatch[3].trainIdx].pt,2,Scalar(40,100,255));

    rightROI[0].x = keypoints_rigth[jointROIDMatch[0].trainIdx].pt.x;
	rightROI[0].y = keypoints_rigth[jointROIDMatch[1].trainIdx].pt.y;

	rightROI[1].x = keypoints_rigth[jointROIDMatch[2].trainIdx].pt.x;
	rightROI[1].y = keypoints_rigth[jointROIDMatch[1].trainIdx].pt.y;

	rightROI[2].x = keypoints_rigth[jointROIDMatch[2].trainIdx].pt.x;
	rightROI[2].y = keypoints_rigth[jointROIDMatch[3].trainIdx].pt.y;

	rightROI[3].x = keypoints_rigth[jointROIDMatch[0].trainIdx].pt.x;
	rightROI[3].y = keypoints_rigth[jointROIDMatch[3].trainIdx].pt.y;

	line(img_line, Point(img_left.cols,0) + rightROI[0], Point(img_left.cols,0) + rightROI[1],Scalar(40,80,255));
	line(img_line, Point(img_left.cols,0) + rightROI[1], Point(img_left.cols,0) + rightROI[2],Scalar(80,80,255));
	line(img_line, Point(img_left.cols,0) + rightROI[2], Point(img_left.cols,0) + rightROI[3],Scalar(120,80,255));
	line(img_line, Point(img_left.cols,0) + rightROI[0], Point(img_left.cols,0) + rightROI[3],Scalar(160,80,255));

	circle(img_line,keypoints_left[minFilteredMatche.queryIdx].pt,2,Scalar(40,100,255));
	circle(img_line,keypoints_rigth[minFilteredMatche.trainIdx].pt + Point2f(img_left.cols,0),2,Scalar(40,100,255));
	Point2f img_offset = keypoints_left[minFilteredMatche.queryIdx].pt - keypoints_rigth[minFilteredMatche.trainIdx].pt;
	cout << img_offset << endl; 
	
	imshow("show line",img_line);
	cout << "minThresholdSelection " << minThresholdSelection << endl;
	
	drawMatches( img_left, keypoints_left, img_right, keypoints_rigth, filteredMatches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	imshow("show",img_matches);

	
	Point2i left_point_start = leftROI[0];
	Point2i left_point_end = leftROI[2];

	Point2i right_point_start = rightROI[0];
	Point2i right_point_end = rightROI[2];

	int left_point_x_shift = 0;
	int left_point_y_shift = 0;
	int right_point_x_shift = 0;
	int right_point_y_shift = 0;

	double minVal; double maxVal; Point minLoc; Point maxLoc;

	max_disparity = 25;

	Mat disp(left_point_end.y - left_point_start.y, left_point_end.x - left_point_start.x,CV_32F);
	Mat disp_255(left_point_end.y - left_point_start.y, left_point_end.x - left_point_start.x,CV_32F);

	float dispar_y  = 0;
	float dispar_x  = 0;
	float dispar_0 = 0;
	float dispar_255 = 0;
	
	Mat temp_left(5,5,img_left.type(), Scalar(255,255,255));
	Mat temp_right(5,5,img_right.type(), Scalar(255,255,255));

	Rect roi_left( 0, 0, max_disparity, temp_right.rows);
	Rect roi_right( 0, 0, max_disparity, temp_left.rows);

	int start=GetTickCount();

	for (int j = right_point_start.y + 2; j < right_point_end.y - 2; j++)
	{
		roi_left.y = left_point_start.y + left_point_y_shift;
		for (int i = right_point_start.x + 2; i < right_point_end.x - 2; i++)
		{
			img_right.row(j - 2).col(i - 2).copyTo(temp_right.row(0).col(0));
			img_right.row(j - 2).col(i - 1).copyTo(temp_right.row(0).col(1));
			img_right.row(j - 2).col(i).copyTo(temp_right.row(0).col(2));
			img_right.row(j - 2).col(i + 1).copyTo(temp_right.row(0).col(3));
			img_right.row(j - 2).col(i + 2).copyTo(temp_right.row(0).col(4));

			img_right.row(j - 1).col(i - 2).copyTo(temp_right.row(1).col(0));
			img_right.row(j - 1).col(i - 1).copyTo(temp_right.row(1).col(1));
			img_right.row(j - 1).col(i).copyTo(temp_right.row(1).col(2));
			img_right.row(j - 1).col(i + 1).copyTo(temp_right.row(1).col(3));
			img_right.row(j - 1).col(i + 2).copyTo(temp_right.row(1).col(4));

			img_right.row(j).col(i - 2).copyTo(temp_right.row(2).col(0));
			img_right.row(j).col(i - 1).copyTo(temp_right.row(2).col(1));
			img_right.row(j).col(i).copyTo(temp_right.row(2).col(2));
			img_right.row(j).col(i + 1).copyTo(temp_right.row(2).col(3));
			img_right.row(j).col(i + 2).copyTo(temp_right.row(2).col(4));

			img_right.row(j + 1).col(i - 2).copyTo(temp_right.row(3).col(0));
			img_right.row(j + 1).col(i - 1).copyTo(temp_right.row(3).col(1));
			img_right.row(j + 1).col(i).copyTo(temp_right.row(3).col(2));
			img_right.row(j + 1).col(i + 1).copyTo(temp_right.row(3).col(3));
			img_right.row(j + 1).col(i + 2).copyTo(temp_right.row(3).col(4));

			img_right.row(j + 2).col(i - 2).copyTo(temp_right.row(4).col(0));
			img_right.row(j + 2).col(i - 1).copyTo(temp_right.row(4).col(1));
			img_right.row(j + 2).col(i).copyTo(temp_right.row(4).col(2));
			img_right.row(j + 2).col(i + 1).copyTo(temp_right.row(4).col(3));
			img_right.row(j + 2).col(i + 2).copyTo(temp_right.row(4).col(4));
			
			roi_left.x = left_point_start.x + left_point_x_shift - 3;
			Mat result_match_0(max_disparity + 1, temp_left.rows + 1, CV_32FC1);
			matchTemplate( img_left(roi_left),temp_right, result_match_0, CV_TM_SQDIFF_NORMED);
			normalize( result_match_0, result_match_0, 0, 1, NORM_MINMAX, -1, Mat() );
			minMaxLoc( result_match_0, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
			result_match_0.release();
			/*
			imshow("temp_right",temp_right);
			imshow("temp_left", img_left(roi_left));
			waitKey(2000);
			destroyWindow("temp_right");
			destroyWindow("temp_left");
			*/

			dispar_y  = (left_point_y_shift + minLoc.y) - (j - right_point_start.y);
			dispar_x  = (left_point_x_shift + minLoc.x) - (i - right_point_start.x);
			dispar_0 = sqrt(pow(dispar_x,2) + pow(dispar_y,2));

			if (dispar_0 >= min_disparity)
				disp.row(left_point_y_shift).col(left_point_x_shift) = dispar_0;
			else 
				disp.row(left_point_y_shift).col(left_point_x_shift) = 0;
			

			if(left_point_x_shift + left_point_start.x + max_disparity <= left_point_end.x)
				left_point_x_shift++;
		}
		if(left_point_y_shift + left_point_start.y + roi_left.height <= left_point_end.y)
			left_point_y_shift++;
		left_point_x_shift = 0;
	}
	left_point_y_shift = 0;
	int end=GetTickCount();
	std::cout <<"Reconstraction time " << (end-start)/1000.0 << std::endl;
	Mat disp8;
	imshow("disparity map",disp);
	normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
	imshow("disparity8 map",disp8);

	waitKey(0);
	system("pause");
	//-- Конец
}

void clearDublicateMatches(vector< DMatch >& matches)
{
	
	for (int i = 0; i < matches.size(); i++)
	{
		float dist = 100;
		DMatch minM = matches[i];
		map<int,DMatch,std::greater<int>> ind_array;
		for (int j = i; j < matches.size(); j++)
		{
			if (matches[i].trainIdx == matches[j].trainIdx)
			{
				if( dist > matches[j].distance ) 
				{ 
					dist = matches[j].distance; 
					minM = matches[j]; 
				}
				ind_array.insert(pair<int,DMatch>(j,matches[j]));
			}
		}
		for (auto it = ind_array.begin(); it != ind_array.end(); it++)
		{
			if ( minM.queryIdx != it->second.queryIdx )	
				matches.erase(matches.begin() + it->first);
		}
		if (ind_array.size() > 1)
			i = -1;
	}
}

float dotToDotLenght(Point2f dotA, Point2f dotB)
{
	return sqrtf(powf(dotB.x - dotA.x,2) + powf(dotB.y - dotA.y,2));
}

void findLinesAndTheirDot(const vector<Vec4i>& linesArray, vector<LineDot>& outLineArray, int cols, int rows,const vector< DMatch >& matches, const vector<Point2f>& keypoint, float maxDistToLine)
{
	for( size_t j = 0; j < linesArray.size(); j++ )
	{
		Vec4i l = linesArray[j];
		LineDot tempLine;
		tempLine.line = l;
		tempLine.maxLenght = 0; 
		tempLine.minLenght = sqrt(cols * cols + rows * rows);
		int cd = 0;
		for( int i = 0; i < matches.size(); i++ )
		{	
			if (dotToLineLenght(Point2f(l[0], l[1]),Point2f(l[2], l[3]),Point2f(keypoint[i].x,keypoint[i].y)) < maxDistToLine)
			{
				dotLenght tempDotAndLenght;
				tempDotAndLenght.dot = keypoint[i];
				tempDotAndLenght.match = matches[i];
				tempDotAndLenght.lenghtToLine = dotToLineLenght(Point2i(l[0], l[1]),Point2i(l[2], l[3]),Point2f(keypoint[i].x,keypoint[i].y));
				tempLine.minLenght = min(tempLine.minLenght,tempDotAndLenght.lenghtToLine);
				tempLine.maxLenght = max(tempLine.maxLenght,tempDotAndLenght.lenghtToLine);
				tempLine.dot.push_back(tempDotAndLenght);
				cd++;
			}
		}
		if (cd > 1)
			outLineArray.push_back(tempLine);
	}
}
vector<Vec9f> findSimilarLinesAndAngleBetweenThem(vector<LineDot> leftImgLine, vector<LineDot> rightImgLine)
{
	vector<Vec9f> result;
	for (int i = 0; i < leftImgLine.size(); i++)
	{
		for (int j = 0; j < rightImgLine.size(); j++)
		{
			int countDot = 0;
			for (int k = 0; k < leftImgLine[i].dot.size(); k++)
			{
				for (int l = 0; l < rightImgLine[j].dot.size(); l++)
				{
					int leftQueryIdx = leftImgLine[i].dot[k].match.queryIdx;
					int rightQueryIdx = rightImgLine[j].dot[l].match.queryIdx;
					int leftTrainIdx = leftImgLine[i].dot[k].match.trainIdx;
					int rightTrainIdx = rightImgLine[j].dot[l].match.trainIdx;
					if (( leftQueryIdx == rightQueryIdx) && ( leftTrainIdx == rightTrainIdx))
					{
						countDot++;
					}
				}
			}
			if (countDot >= 2) 
			{
				float ang = angle( Point2f (rightImgLine[j].line[0],rightImgLine[j].line[1]), Point2f (rightImgLine[j].line[2],rightImgLine[j].line[3])) - 
					        angle( Point2f (leftImgLine[i].line[0],leftImgLine[i].line[1]), Point2f (leftImgLine[i].line[2],leftImgLine[i].line[3]));
				result.push_back(Vec9f(leftImgLine[i].line[0],leftImgLine[i].line[1],leftImgLine[i].line[2],leftImgLine[i].line[3],
									   rightImgLine[j].line[0],rightImgLine[j].line[1],rightImgLine[j].line[2],rightImgLine[j].line[3],ang));
				leftImgLine.erase(leftImgLine.begin() + i);
				i = -1;
				rightImgLine.erase(rightImgLine.begin() + j);
				j = -1;
				break;
			}
		}
	}
	return result;
}

double angle( Point2f begin, Point2f a_end)
{
	a_end -= begin;
	begin = Point2f(0,0);
	if ((a_end.x == 0.0) && (a_end.y == 0.0))
		return -1.0;
	if (a_end.x == 0.0)
		return ((a_end.y > 0.0) ? 90 : 270);
	double theta = atan(a_end.y/a_end.x);                    // в радианах
	theta *= 360 / (2 * CV_PI);            // перевод в градусы
	if (a_end.x > 0.0)                                 // 1 и 4 квадранты
		return ((a_end.y >= 0.0) ? theta : 360 + theta);
	else                                         // 2 и З квадранты
		return (180 + theta);
}

bool checkDotInLine(Point2f lineDotA,Point2f lineDotB,Point2f checkDot)
{
	Point2f ABvector = lineDotB - lineDotA;
	if ((checkDot.y - lineDotA.y) / (lineDotB.y - lineDotA.y) == (checkDot.x - lineDotA.x) / (lineDotB.x - lineDotA.x)) 
		return true;
	return false;
}

float dotToLineLenght(Point2f lineDotA,Point2f lineDotB,Point2f checkDot)
{
	float lenght;
	float Ax,By,C;
	Point2f MN = lineDotB - lineDotA;
	Ax = MN.y;
	By = MN.x;
	C = (-lineDotA.x * MN.y) - (-lineDotA.y * MN.x);
	lenght = abs(Ax * checkDot.x + (-By) * checkDot.y + C) / sqrt( Ax * Ax + By * By);
	return lenght;
}

vector<Vec4i> findLines(Mat& imag, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0)
{
	Mat gray, edge, draw;
    cvtColor(imag, gray, CV_BGR2GRAY);
    Canny( gray, edge, 50, 250, 3);
	vector<Vec4i> out_lines;
	HoughLinesP(edge, out_lines, rho, CV_PI/theta, threshold, minLineLength, maxLineGap );
	return out_lines;
}

void rotation(float angle, Mat& imag)
{
	float theta = angle;
	Mat src,frame, frameRotated;
	src = imag;
	int diagonal = (int)sqrt(src.cols*src.cols+src.rows*src.rows);
	int newWidth = diagonal;
	int newHeight =diagonal;

	int offsetX = (newWidth - src.cols) / 2;
	int offsetY = (newHeight - src.rows) / 2;
	Mat targetMat(newWidth, newHeight, src.type());
	Point2f src_center(targetMat.cols/2.0F, targetMat.rows/2.0F);

	src.copyTo(frame);
	double radians = theta * CV_PI / 180.0;

	frame.copyTo(targetMat.rowRange(offsetY, offsetY + frame.rows).colRange(offsetX, offsetX + frame.cols));
	Mat rot_mat = getRotationMatrix2D(src_center, theta, 1.0);
	warpAffine(targetMat, frameRotated, rot_mat, targetMat.size());

	Rect_<double> bound_Rect(frame.cols,frame.rows,0,0);

	int x1 = offsetX;
	int x2 = offsetX+frame.cols;
	int x3 = offsetX;
	int x4 = offsetX+frame.cols;

	int y1 = offsetY;
	int y2 = offsetY;
	int y3 = offsetY+frame.rows;
	int y4 = offsetY+frame.rows;

	Mat co_Ordinate = (Mat_<double>(3,4) << x1, x2, x3, x4,
											y1, y2, y3, y4,
											1,  1,  1,  1 );
	Mat RotCo_Ordinate = rot_mat * co_Ordinate;

	for(int i=0;i<4;i++){
		if(RotCo_Ordinate.at<double>(0,i)<bound_Rect.x)
			bound_Rect.x=(int)RotCo_Ordinate.at<double>(0,i); //access smallest 
		if(RotCo_Ordinate.at<double>(1,i)<bound_Rect.y)
		bound_Rect.y=RotCo_Ordinate.at<double>(1,i); //access smallest y
		}

		for(int i=0;i<4;i++){
		if(RotCo_Ordinate.at<double>(0,i)>bound_Rect.width)
			bound_Rect.width=(int)RotCo_Ordinate.at<double>(0,i); //access largest x
		if(RotCo_Ordinate.at<double>(1,i)>bound_Rect.height)
		bound_Rect.height=RotCo_Ordinate.at<double>(1,i); //access largest y
		}

	bound_Rect.width=bound_Rect.width-bound_Rect.x;
	bound_Rect.height=bound_Rect.height-bound_Rect.y;

	Mat cropedResult;
	Mat ROI = frameRotated(bound_Rect);
	ROI.copyTo(cropedResult);
	cropedResult.copyTo(imag);
}

void brightnessNormalizationEqualizeHist(const Mat& inputImg,Mat& outputImg, int type)
{
	vector<Mat> channels;
	split(inputImg,channels);
	double minR, minG, minB;
	double maxR, maxG, maxB;
	minR = minG = minB = 255;
	maxR = maxG = maxB = 0;
	int rows = channels[0].rows;
	int cols = channels[0].cols;
	if (type == 1)
	{
		Point minLoc; Point maxLoc;
		minMaxLoc( channels[0], &minB, &maxB, &minLoc, &maxLoc, Mat() );
		minMaxLoc( channels[1], &minG, &maxG, &minLoc, &maxLoc, Mat() );
		minMaxLoc( channels[2], &minR, &maxR, &minLoc, &maxLoc, Mat() );

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
			{
				if (minB != 0 || maxB != 255)
					channels[0].row(i).col(j) = 255 * (channels[0].row(i).col(j) - minB) / (maxB - minB);	
				if (minG != 0 || maxG != 255)
					channels[1].row(i).col(j) = 255 * (channels[1].row(i).col(j) - minG) / (maxG - minG);
				if (minR != 0 || maxR != 255)
					channels[2].row(i).col(j) = 255 * (channels[2].row(i).col(j) - minR) / (maxR - minR);
			}
	}
	else
	{
		equalizeHist(channels[0], channels[0]);
		equalizeHist(channels[1], channels[1]);
		equalizeHist(channels[2], channels[2]);
	}
	merge(channels,outputImg);
}

/*
Мінімальна відстань між точками
float minDistanceBetweenTwoDot = sqrt(img_left.cols * img_left.cols + img_left.rows * img_left.rows);
	vector<DMatch> matchesOfMinDistanceBetweenTwoDot(2);
	for (int i = 0; i < matches.size() - 1; i++)
	{
		for (int j = i + 1; j < matches.size(); j++)
		{
			float distLeft = dotToDotLenght(keypoints_left[matches[i].queryIdx].pt,keypoints_left[matches[j].queryIdx].pt);
			float distRight = dotToDotLenght(keypoints_rigth[matches[i].trainIdx].pt,keypoints_rigth[matches[j].trainIdx].pt);
			if (abs(distLeft - distRight) < minDistanceBetweenTwoDot) 
			{ 
				minDistanceBetweenTwoDot = abs(distLeft - distRight);
				matchesOfMinDistanceBetweenTwoDot[0] = matches[i];
				matchesOfMinDistanceBetweenTwoDot[1] = matches[j];
			}
		}
	}
	circle(left_right_img,keypoints_left[matchesOfMinDistanceBetweenTwoDot[0].queryIdx].pt,2,Scalar(50,255,50));
	circle(left_right_img,Point2f(img_left.cols,0) + keypoints_rigth[matchesOfMinDistanceBetweenTwoDot[0].trainIdx].pt,2,Scalar(50,255,50));
	imshow("left_right_img",left_right_img);
	cout << minDistanceBetweenTwoDot << endl;
*/

/*
for (int j = left_point_start.x + 3; j < left_point_end.x - 3; j++)
	{
		for (int i = left_point_start.y + 3; i < left_point_end.y - 3; i++)
		{
			Mat temp_left(3,3,img_left.type(), Scalar(0,0,0));
			img_left.row(j - 1).col(i).copyTo(temp_left.row(0).col(1));
			img_left.row(j).col(i - 1).copyTo(temp_left.row(1).col(0));
			img_left.row(j).col(i).copyTo(temp_left.row(1).col(1));
			img_left.row(j).col(i + 1).copyTo(temp_left.row(1).col(2));
			img_left.row(j + 1).col(i).copyTo(temp_left.row(2).col(1));
			imshow("temp_left",temp_left);
			Mat temp_right(5,30,img_right.type(), Scalar(0,0,0));
			int right_point_y_shift = 0;
			for (int k = right_point_start.x; k < (temp_right.rows + right_point_start.x); k++)
			{
				for (int l = right_point_start.y; l < (temp_right.cols + right_point_start.y) && l <=; l++)
				{
					img_right.row(k).col(l).copyTo(temp_right.row(k - right_point_start.x).col(l - right_point_start.y));
				}
			}
			imshow("temp_right",temp_right);
			waitKey(30);
			destroyWindow("temp_left");
			destroyWindow("temp_right");
			right_point_start.y++;
			if (right_point_start.x + temp_right.rows > right_point_end.x)
				right_point_start.x++;
			cout << right_point_start.x << endl;
		}

	}
*/