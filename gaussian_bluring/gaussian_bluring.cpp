// gaussian_bluring.cpp : 기본 프로젝트 파일입니다.

#include "stdafx.h"

using namespace System;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
 int main() {
//read the image
Mat image= cv::imread("image/test1.jpg",-1);
Mat blur_result;
Mat smooth_result;
Mat gray_result;
Mat thresh_img;
Mat thresh_adt_img;
Mat edge_img;
Mat blur_edge_img;

// create image window
//namedWindow("My Image");
//display image
imshow("My Image", image);

//gray scale 변환
cvtColor(image,gray_result,CV_BGR2GRAY);
//imshow("My gray", gray_result);

//blur image
GaussianBlur(gray_result,blur_result,cv::Size(5,5),0);
//imshow("My burred", blur_result);

//smooth image
GaussianBlur(gray_result,smooth_result,cv::Size(5,5),10);
//imshow("My smooth", smooth_result);

//edge detect
Canny(gray_result,edge_img,50,100,3);
//imshow("My Edge", edge_img);
Canny(smooth_result,edge_img,50,100,3);
//imshow("My Edge2", edge_img);

////sobel edge
//	Mat sobel_result;
//    Mat sobelX;
//    Mat sobelY;
//    Sobel(gray_result, sobelX, CV_8U, 1, 0);
//    Sobel(gray_result, sobelY, CV_8U, 0, 1);
//    sobel_result = abs(sobelX) + abs(sobelY);
//	imshow("My sobel",sobel_result);

//threshold image
threshold(smooth_result,thresh_img,100,255,CV_THRESH_OTSU);
	adaptiveThreshold(smooth_result,           // input image //지역가변 쓰레쉬 홀드사용
		thresh_adt_img,                              // output image
		255,                                    // make pixels that pass the threshold full white
		ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
		THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
		15,                                     // size of a pixel neighborhood used to calculate threshold value
		2);                                     // constant subtracted from the mean or weighted mean

//imshow("My Thresh1", thresh_img);
//imshow("My Thresh2", thresh_adt_img);


//////////////////////////////////////전처리 과정 끝


//contours찾기
 Mat contour_img = thresh_adt_img.clone();
 Mat original = image.clone();
 vector<vector<Point> > contours;
 findContours( contour_img, //외곽선 검출한후 저장
	 contours, //외곽선 저장
	 RETR_LIST, //모든 윤곽선 검색
	 CHAIN_APPROX_SIMPLE);//양끝만 남김 제일위 제일아래 우측 좌측

//외곽선 긴거 삭제
 const int cmin= 300;  // 최소 외곽선 길이
 const int cmax= 500; // 최대 외곽선 길이
 vector<vector<Point>>::const_iterator itc= contours.begin();
 while (itc!=contours.end()) {//itc가 constours의 끝값이 아닌경우
  if (itc->size() < cmin || itc->size() > cmax)//itc의 크기(외곽선 크기)가 일정크기 제외삭제
   itc= contours.erase(itc);
  else 
   ++itc;
 }
 //for(int i = 0; i < contours.size(); i++){// 이것도 외곽선 크기에 따른 검출이지만 안됨
	// if (cmax> contourArea(contours[i]) >cmin){//이유 분석 필요
	//	 Rect bndRect = boundingRect(contours[i]);

	//	 rectangle(original, bndRect, cv::Scalar(0, 0, 255), 2);
	//	 
	//	 Mat roi_img = thresh_adt_img(bndRect);

	// }
 //}

//원본영상에 외곽
 //drawContours(original,contours,-1,Scalar(0,255,0),2);
 Rect bndRect = boundingRect(contours[0]);//외곽선이 하나만 남는데 그거 사각형처리
 
 rectangle(original, bndRect, cv::Scalar(0, 0, 255), 2);//영상에 표시

 Rect roiRect(0,//원하는 관심영역 추출 시작점은 가로는 0부터
	 bndRect.height/12*10,//세로는 추출해낸 외곽선의 10/12지점 부터 시작
	 bndRect.width,//원하는 roi 크기 세로는 추출한 외곽선 가로길이
	 bndRect.height/12*2);//세로길이의 2/12 크기로 추출

 Mat coi_img = image(bndRect);
 Mat coi = thresh_adt_img(bndRect);

 rectangle(coi,roiRect,cv::Scalar(255, 255, 255), 2);
 Mat roi_img = coi_img(roiRect);//
 Mat roi_th = coi(roiRect);//roi검출
 imwrite("roi_img.jpg",roi_th);
  

//imshow("관심",roi_img);
imshow("외곽",original);
////////////////////////////////////관심영역 1 끝

 //mat ocr_org = roi_img.clone();//관심영역1의 rgb영상
 //mat ocr_th = roi_th.clone();//관심영역1의 이진화 영상
 //
 //vector<vector<point> > cont_ocr;//문자 검출을 위한 외곽선변수
 //
 //findcontours( ocr_th, //외곽선 검출할 이미지
	//  cont_ocr, //외곽선 저장
	//  retr_list, //모든 윤곽선 검색
	//  chain_approx_simple);//양끝만 남김 제일위 제일아래 우측 좌측

 //int ocmin = 20;//원하는 외곽선의 크기 검출
 //int ocmax = 60;
 // vector<vector<point>>::const_iterator ito= cont_ocr.begin();
 //while (ito!=cont_ocr.end()) {
 // if (ito->size() < ocmin || ito->size() > ocmax)
 //  ito= cont_ocr.erase(ito);
 // else 
 //  ++ito;
 //}

 // for(int i = 0; i < cont_ocr.size(); i++){//원하는 크기의 외곽선들을 사각으로 감쌈

	//	 rect ocr_bnd = boundingrect(cont_ocr[i]);

	//	 rectangle(ocr_org, ocr_bnd, cv::scalar(255, 0, 255), 2);	
	//	 
	//	 mat ocr_roi = roi_th(ocr_bnd);
	//	 mat roi_re;
	//	 resize(ocr_roi,roi_re,cv::size(40,50));//size(wid,hei)
	//	 imshow("roi ocr",ocr_roi);
	//	 imshow("resize",roi_re);
	//	 imwrite("class.jpg", ocr_roi);
	//	 imshow("ocr결론",ocr_org);
	//	 int intchar = cv::waitkey(0);//입력키가 들어와야 다음 것이 실행
	//	 if (intchar == 27) {        // if esc key was pressed
	//			return(0);              // exit program
	//		}
	//	 
 //}
 //
 //
 //imshow("thresh",roi_th);
///////////////////////////관심영역 2 + 트레이닝 끝

 Mat ocr_th = imread("ocr_class/roi_img.jpg");
 Mat temp1 = imread("ocr_class/0.jpg");
 
 //vector<vector<Point> > cont_ocr;//문자 검출을 위한 외곽선변수
 //
 //findContours( ocr_th, //외곽선 검출할 이미지
	//  cont_ocr, //외곽선 저장
	//  RETR_LIST, //모든 윤곽선 검색
	//  CHAIN_APPROX_SIMPLE);//양끝만 남김 제일위 제일아래 우측 좌측

 //int ocmin = 20;//원하는 외곽선의 크기 검출
 //int ocmax = 60;
 // vector<vector<Point>>::const_iterator ito= cont_ocr.begin();
 //while (ito!=cont_ocr.end()) {
 // if (ito->size() < ocmin || ito->size() > ocmax)
 //  ito= cont_ocr.erase(ito);
 // else 
 //  ++ito;
 //}
 //
 // for(int i = 0; i < cont_ocr.size(); i++){//원하는 크기의 외곽선들을 사각으로 감쌈

	//	 Rect ocr_bnd = boundingRect(cont_ocr[i]);

	//	 rectangle(ocr_org, ocr_bnd, cv::Scalar(255, 0, 255), 2);	
	//	 
	//	 Mat ocr_roi = roi_th(ocr_bnd);
	//	 Mat roi_re;
	//	 resize(ocr_roi,roi_re,cv::Size(40,50));//size(wid,hei)
	//	 imshow("roi ocr",ocr_roi);
	//	 imshow("resize",roi_re);
	//	 imwrite("class.jpg", roi_re);
	//	 imshow("ocr결론",ocr_org);
	//	 int intChar = cv::waitKey(0);//입력키가 들어와야 다음 것이 실행
	//	 if (intChar == 27) {        // if esc key was pressed
	//			return(0);              // exit program
	//		}
 // }

  double minVal, maxVal;
  Point minLoc, maxLoc;
  Mat result;
  matchTemplate(ocr_th, temp1, result, TM_SQDIFF);
  minMaxLoc(result, &minVal, NULL, &minLoc, NULL);
  rectangle(roi_img,minLoc,Point(minLoc.x+ temp1.cols,minLoc.y+temp1.rows), Scalar(255, 0, 0), 2);
  imshow("temp",roi_img);
  
		 

 
 

//wait key
waitKey(0);

return 0;
 }
