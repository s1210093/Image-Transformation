#include<iostream>
#include<vector>
#include<fstream>

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

static const int src_img_rows = 640;
static const int src_img_cols = 800;

static const double R = 1;
static const double G = 1;
static const double B = 0;

static const int THRESH = 120; // しきい値

using namespace cv;
using namespace std;

void onTrackbarChanged(int thres, void*);
Point2i calculate_center(Mat);
void getCoordinates(int event, int x, int y, int flags, void* param);
Mat undist(Mat);
double get_point_distance(Point2i, Point2i);

Mat image1;
Mat src_img;
ofstream fout("out");
Mat element = Mat::ones(3, 3, CV_8UC1); // 追加　3×3の行列ですべて1　dilate必要な行列
int Ax, Ay, Bx, By, Cx, Cy, Dx, Dy;
int Tr, Tg, Tb;
Point2i pre_point;

int main(int argc, char *argv[])
{

	Mat in_img = imread("./PICTURE/78.jpg");

	for (int i = 1; i <= 1; i++){

		//画像をリサイズ(大きすぎるとディスプレイに入りきらないため)
		resize(in_img, src_img, Size(src_img_cols, src_img_rows), CV_8UC3);

		if (i == 1) {
			namedWindow("getCoordinates");
			imshow("getCoordinates", src_img);
			cvSetMouseCallback("getCoordinates", getCoordinates, NULL); //変換したい四角形の四隅の座標を取る(クリック)
			waitKey(0);
			destroyAllWindows();

		}

		//-----------------透視変換-----------------------------------------------
		Point2f pts1[] = { Point2f(Ax, Ay), Point2f(Bx, By), Point2f(Cx, Cy), Point2f(Dx, Dy) };

		Point2f pts2[] = { Point2f(0, src_img_rows), Point2f(0, 0), Point2f(src_img_cols, 0), Point2f(src_img_cols, src_img_rows) };

		//透視変換行列を計算
		Mat perspective_matrix = getPerspectiveTransform(pts1, pts2);
		Mat dst_img;
		//変換
		warpPerspective(src_img, dst_img, perspective_matrix, src_img.size(), INTER_LINEAR);

		//変換前後の座標を描画
		line(src_img, pts1[0], pts1[1], Scalar(225, 0, 225), 2, CV_AA);
		line(src_img, pts1[1], pts1[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img, pts1[2], pts1[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img, pts1[3], pts1[0], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img, pts2[0], pts2[1], Scalar(255, 0, 255), 2, CV_AA);
		line(src_img, pts2[1], pts2[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img, pts2[2], pts2[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img, pts2[3], pts2[0], Scalar(255, 255, 0), 2, CV_AA);

		//------------------グレースケール化-------------------------------
		int x, y;
		uchar r1, g1, b1, d;
		Vec3b color1;
		image1 = Mat(Size(dst_img.cols, dst_img.rows), CV_8UC1);
		for (y = 0; y < dst_img.rows; y++){
			for (x = 0; x < dst_img.cols; x++){
				color1 = dst_img.at<Vec3b>(y, x);
				r1 = color1[2];
				g1 = color1[1];
				b1 = color1[0];
				d = saturate_cast<uchar>(R * r1 + G * g1 + B * b1);
				image1.at<uchar>(y, x) = d;
			}
		}

		//2値化
		if (i == 1){
			namedWindow("binari");
			int value = 0;
			createTrackbar("value", "binari", &value, 255, onTrackbarChanged);
			setTrackbarPos("value", "binari", 128);
		}

		Mat binari_2;


		//---------------------二値化--------------------------------------
		threshold(image1, binari_2, THRESH, 255, THRESH_BINARY);
		binari_2 = ~binari_2;//ネガポジ
		dilate(binari_2, binari_2, element, Point(-1, -1), 3);//膨張処理3回　最後の引数で回数を設定

		//---------------------重心取得---------------------------------------------------------
		Point2i point = calculate_center(binari_2);//momentで白色部分の重心を求める
		cout << "position: " << point.x << " " << point.y << endl;
		if (point.x != 0){
			fout << point.x << " " << src_img_rows - point.y << endl;
		}

		Mat dst_img2 = dst_img.clone();

		// 画像、円の中心座標、半径、色、線太さ、種類(-1, CV_AAは塗りつぶし)
		circle(dst_img, Point(point.x, point.y), 5, Scalar(0, 0, 200), -1, CV_AA);

		//-----------------表示部分---------------------------------
		Mat base(src_img_rows, src_img_cols * 2, CV_8UC3);
		Mat roi1(base, Rect(0, 0, src_img.cols, src_img.rows));
		src_img.copyTo(roi1);
		Mat roi2(base, Rect(dst_img.cols, 0, dst_img.cols, dst_img.rows));
		dst_img.copyTo(roi2);

		if (i == 1){
			//namedWindow("src_dst");
			//imshow("src_dst",base);

			namedWindow("src");
			imshow("src", src_img);

			namedWindow("dst");
			imshow("dst", dst_img);

			namedWindow("dst2");
			imshow("dst2", dst_img2);

			namedWindow("gray");
			imshow("gray", image1);

			namedWindow("binari_2");
			imshow("binari_2", binari_2);

			waitKey(0);
			destroyAllWindows();
		}
		//重心位置確認用
		//namedWindow("dst") ;
		//imshow("dst", dst_img) ;
		//waitKey(0);
		//destroyAllwindows();
	}
	fout.close();
	//fout2.close() ;
	getchar();
}

double get_points_distance(Point2i point, Point2i pre_point){

	return sqrt((point.x - pre_point.x)*(point.x - pre_point.x)
		+ (point.y - pre_point.y) * (point.y - pre_point.y));
}
void onTrackbarChanged(int thres, void*)
{

	Mat image2;
	threshold(image1, image2, thres, 255, THRESH_BINARY);

	imshow("binari", image2);
}

Point2i calculate_center(Mat gray)
{
	Point2i center = Point2i(0, 0);
	Moments moment = moments(gray, true);

	if (moment.m00 != 0)
	{
		center.x = (int)(moment.m10 / moment.m00);
		center.y = (int)(moment.m01 / moment.m00);
	}

	return center;
}

void getCoordinates(int event, int x, int y, int flags, void* param)
{
	static int count = 0;
	switch (event){
	case CV_EVENT_LBUTTONDOWN:

		if (count == 0){
			Ax = x, Ay = y;
			cout << "Ax :" << x << ", Ay: " << y << endl;
		}
		else if (count == 1){
			Bx = x, By = y;
			cout << "Bx :" << x << ", By:" << y << endl;
		}
		else if (count == 2){
			Cx = x, Cy = y;
			cout << "Cx :" << x << ", Cy:" << y << endl;
		}
		else if (count == 3){
			Dx = x, Dy = y;
			cout << "Dx :" << x << ", Dy" << y << endl;
		}
		else{
			cout << "rgb(" << x << "," << y << ") ";
			//fout2 << x << " " << y << endl ;
			Vec3b target_color = src_img.at<Vec3b>(y, x);
			uchar r, g, b;
			Tr = target_color[2];
			Tg = target_color[1];
			Tb = target_color[0];
			cout << "r:" << Tr << " g:" << Tg << " b:" << Tb << endl;
		}
		count++;
		break;

	default:
		break;
	}
}

Mat undist(Mat src_img)
{
	Mat dst_img;

	//カメラマトリックス
	Mat cameraMatrix = (Mat_<double>(3, 3) << 469.96, 0, 400, 0, 467.68, 300, 0, 0, 1);
	//歪み行列
	Mat distcoeffs = (Mat_<double>(1, 5) << -0.18957, 0.037319, 0, 0, -0.00337);

	undistort(src_img, dst_img, cameraMatrix, distcoeffs);

	return dst_img;
}
