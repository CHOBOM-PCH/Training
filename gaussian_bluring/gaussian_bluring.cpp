// gaussian_bluring.cpp : �⺻ ������Ʈ �����Դϴ�.

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

//gray scale ��ȯ
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
	adaptiveThreshold(smooth_result,           // input image //�������� ������ Ȧ����
		thresh_adt_img,                              // output image
		255,                                    // make pixels that pass the threshold full white
		ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
		THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
		15,                                     // size of a pixel neighborhood used to calculate threshold value
		2);                                     // constant subtracted from the mean or weighted mean

//imshow("My Thresh1", thresh_img);
//imshow("My Thresh2", thresh_adt_img);


//////////////////////////////////////��ó�� ���� ��


//contoursã��
 Mat contour_img = thresh_adt_img.clone();
 Mat original = image.clone();
 vector<vector<Point> > contours;
 findContours( contour_img, //�ܰ��� �������� ����
	 contours, //�ܰ��� ����
	 RETR_LIST, //��� ������ �˻�
	 CHAIN_APPROX_SIMPLE);//�糡�� ���� ������ ���ϾƷ� ���� ����

//�ܰ��� ��� ����
 const int cmin= 300;  // �ּ� �ܰ��� ����
 const int cmax= 500; // �ִ� �ܰ��� ����
 vector<vector<Point>>::const_iterator itc= contours.begin();
 while (itc!=contours.end()) {//itc�� constours�� ������ �ƴѰ��
  if (itc->size() < cmin || itc->size() > cmax)//itc�� ũ��(�ܰ��� ũ��)�� ����ũ�� ���ܻ���
   itc= contours.erase(itc);
  else 
   ++itc;
 }
 //for(int i = 0; i < contours.size(); i++){// �̰͵� �ܰ��� ũ�⿡ ���� ���������� �ȵ�
	// if (cmax> contourArea(contours[i]) >cmin){//���� �м� �ʿ�
	//	 Rect bndRect = boundingRect(contours[i]);

	//	 rectangle(original, bndRect, cv::Scalar(0, 0, 255), 2);
	//	 
	//	 Mat roi_img = thresh_adt_img(bndRect);

	// }
 //}

//�������� �ܰ�
 //drawContours(original,contours,-1,Scalar(0,255,0),2);
 Rect bndRect = boundingRect(contours[0]);//�ܰ����� �ϳ��� ���µ� �װ� �簢��ó��
 
 rectangle(original, bndRect, cv::Scalar(0, 0, 255), 2);//���� ǥ��

 Rect roiRect(0,//���ϴ� ���ɿ��� ���� �������� ���δ� 0����
	 bndRect.height/12*10,//���δ� �����س� �ܰ����� 10/12���� ���� ����
	 bndRect.width,//���ϴ� roi ũ�� ���δ� ������ �ܰ��� ���α���
	 bndRect.height/12*2);//���α����� 2/12 ũ��� ����

 Mat coi_img = image(bndRect);
 Mat coi = thresh_adt_img(bndRect);

 rectangle(coi,roiRect,cv::Scalar(255, 255, 255), 2);
 Mat roi_img = coi_img(roiRect);//
 Mat roi_th = coi(roiRect);//roi����
 imwrite("roi_img.jpg",roi_th);
  

//imshow("����",roi_img);
imshow("�ܰ�",original);
////////////////////////////////////���ɿ��� 1 ��

 //mat ocr_org = roi_img.clone();//���ɿ���1�� rgb����
 //mat ocr_th = roi_th.clone();//���ɿ���1�� ����ȭ ����
 //
 //vector<vector<point> > cont_ocr;//���� ������ ���� �ܰ�������
 //
 //findcontours( ocr_th, //�ܰ��� ������ �̹���
	//  cont_ocr, //�ܰ��� ����
	//  retr_list, //��� ������ �˻�
	//  chain_approx_simple);//�糡�� ���� ������ ���ϾƷ� ���� ����

 //int ocmin = 20;//���ϴ� �ܰ����� ũ�� ����
 //int ocmax = 60;
 // vector<vector<point>>::const_iterator ito= cont_ocr.begin();
 //while (ito!=cont_ocr.end()) {
 // if (ito->size() < ocmin || ito->size() > ocmax)
 //  ito= cont_ocr.erase(ito);
 // else 
 //  ++ito;
 //}

 // for(int i = 0; i < cont_ocr.size(); i++){//���ϴ� ũ���� �ܰ������� �簢���� ����

	//	 rect ocr_bnd = boundingrect(cont_ocr[i]);

	//	 rectangle(ocr_org, ocr_bnd, cv::scalar(255, 0, 255), 2);	
	//	 
	//	 mat ocr_roi = roi_th(ocr_bnd);
	//	 mat roi_re;
	//	 resize(ocr_roi,roi_re,cv::size(40,50));//size(wid,hei)
	//	 imshow("roi ocr",ocr_roi);
	//	 imshow("resize",roi_re);
	//	 imwrite("class.jpg", ocr_roi);
	//	 imshow("ocr���",ocr_org);
	//	 int intchar = cv::waitkey(0);//�Է�Ű�� ���;� ���� ���� ����
	//	 if (intchar == 27) {        // if esc key was pressed
	//			return(0);              // exit program
	//		}
	//	 
 //}
 //
 //
 //imshow("thresh",roi_th);
///////////////////////////���ɿ��� 2 + Ʈ���̴� ��

 Mat ocr_th = imread("ocr_class/roi_img.jpg");
 Mat temp1 = imread("ocr_class/0.jpg");
 
 //vector<vector<Point> > cont_ocr;//���� ������ ���� �ܰ�������
 //
 //findContours( ocr_th, //�ܰ��� ������ �̹���
	//  cont_ocr, //�ܰ��� ����
	//  RETR_LIST, //��� ������ �˻�
	//  CHAIN_APPROX_SIMPLE);//�糡�� ���� ������ ���ϾƷ� ���� ����

 //int ocmin = 20;//���ϴ� �ܰ����� ũ�� ����
 //int ocmax = 60;
 // vector<vector<Point>>::const_iterator ito= cont_ocr.begin();
 //while (ito!=cont_ocr.end()) {
 // if (ito->size() < ocmin || ito->size() > ocmax)
 //  ito= cont_ocr.erase(ito);
 // else 
 //  ++ito;
 //}
 //
 // for(int i = 0; i < cont_ocr.size(); i++){//���ϴ� ũ���� �ܰ������� �簢���� ����

	//	 Rect ocr_bnd = boundingRect(cont_ocr[i]);

	//	 rectangle(ocr_org, ocr_bnd, cv::Scalar(255, 0, 255), 2);	
	//	 
	//	 Mat ocr_roi = roi_th(ocr_bnd);
	//	 Mat roi_re;
	//	 resize(ocr_roi,roi_re,cv::Size(40,50));//size(wid,hei)
	//	 imshow("roi ocr",ocr_roi);
	//	 imshow("resize",roi_re);
	//	 imwrite("class.jpg", roi_re);
	//	 imshow("ocr���",ocr_org);
	//	 int intChar = cv::waitKey(0);//�Է�Ű�� ���;� ���� ���� ����
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
