#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
int main(int argc , char** argv)
{

cv::VideoCapture capture(argv[1]);

if(!capture.isOpened())
	return 1;

double rate = capture.get(CV_CAP_PROP_FPS);
cout << "Rate of the video is " << rate;

bool stop(false);

cv::Mat frame;
cv::namedWindow("Extracted Frame");

int delay = 1000/rate;

while(!stop)
{
	static int FrameNum;
	FrameNum++;
	if(!capture.read(frame))
		break;
//	cv::imshow("Extracted Frame",frame);
	if( !(FrameNum % 30) )
	{
		cv::imwrite("../FrameExtract/" + std::to_string( (FrameNum/30) ) + ".png" , frame);
	}
	
//	if( cv::waitKey(delay)>=0 )
//		stop = true;

}

capture.release();
return 0;


}
