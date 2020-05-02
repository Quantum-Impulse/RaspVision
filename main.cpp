#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <time.h>
#include <ratio>
#include <chrono>

#include <networktables/NetworkTableInstance.h>
//#include <vision/VisionPipeline.h>
//#include <vision/VisionRunner.h>

#include <wpi/StringRef.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>
#include <wpi/raw_ostream.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cscore_oo.h>

#include "cameraserver/CameraServer.h"

#include "FilterAndProcess.h"

using namespace std;
using namespace cv;

class VideoShow{
  public:
  //Class that continuously shows a frame using a dedicated thread.
   cs::CvSource outputStream;
   cv::Mat frame;
   bool stopped = false;
   std::string name = "stream";
   
  VideoShow(int imgWidth, int imgHeight, frc::CameraServer* cameraServer, Mat frameImg){
    outputStream = cameraServer->PutVideo(name, imgHeight, imgWidth); 
    frame = frameImg;
  }
  
  void show(){
    if (!stopped){
      outputStream.PutFrame(frame);
    }
  }
  void start(){
    //return std::thread(&VideoShow::show, this);
    std::thread t2(&VideoShow::show, this);
    t2.detach();
  }

  void stop(){
    stopped = true;
  }

  void notifyError(std::string error){
    outputStream.NotifyError(error);
  }
};

class WebcamVideoStream{
  public:
  cs::UsbCamera webcam;
  bool autoExpose;
  cv::Mat img;
  cs::CvSink stream;
  std::string str = "WebcamVideoStream";
  WebcamVideoStream(cs::UsbCamera camera, frc::CameraServer *cameraServer, int frameWidth, int frameHeight){
     // initialize the video camera stream and read the first frame
     // from the stream
    webcam = camera;
    //Automatically sets exposure to 0 to track tape
    webcam.SetExposureManual(0);
    autoExpose = false;
    //Make a blank image to write on
    //get the video
    stream = cameraServer->GetVideo(camera);
    img = stream.GrabFrame(img); // might have to change the refresh rate for fps
  }
  
  void start(){
    //start the thread to read frames from the video stream
    std::thread t(&WebcamVideoStream::update, this);
    t.detach();
  }

  void update(){
    webcam.SetExposureManual(0);
    img = stream.GrabFrame(img); 
  }
   cv::Mat read(){
    return img;
  }

  std::string getError(){
    return stream.GetError();
  }
};

/////////////////////// OPENCV Processing //////////////////////////////
//math constants
double pi = 3.141592653; //PI
double convertToDegress = (180.0 / pi); // convert radians to degrees 

// threshold scalar values H S V respectively 
//for tape
Scalar tlow {44, 0, 130};
Scalar tHigh {60, 30, 255};

//Angles in radians
//image size ratioed to 16:9
int imageWidth = 256;
int imageHeight = 144;

//Center(x,y) of the image
double centerX = (imageWidth / 2) - .5;
double centerY = (imageHeight / 2) - .5;

//16:9 aspect ratio
int horizontalAspect = 16;
int verticalAspect = 9;

//Reasons for using diagonal aspect is to calculate horizontal field of view.
double diagonalAspect = hypot(horizontalAspect, verticalAspect);

//Lifecam 3000 from datasheet
//Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf
//convert degrees '68.5' to radians 
double diagonalView = 68.5 * (convertToDegress);

//Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
double horizontalView = atan(tan(diagonalView / 2) * (horizontalAspect / diagonalAspect)) * 2;
double verticalView = atan(tan(diagonalView / 2) * (verticalAspect / diagonalAspect)) * 2;

//Focal Length calculations : https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
double H_FOCAL_LENGTH = imageWidth / (2 * tan((horizontalView / 2)));
double V_FOCAL_LENGTH = imageHeight / (2 * tan((verticalView / 2)));

// Tracked objects centroid coordinates
int theObject[2] = { 0,0 };
double yaw = 0, pitch = 0;
//bounding rectangle of the object, we will use the center of this as its position
Rect objectBoundingRectangle = Rect(0, 0, 0, 0);

//Uses trig and focal length of camera to find yaw.
//Link to further explanation : https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
double calculateYaw(double pixelX, double CenterX, double hFocalLength) {
	double yaw = convertToDegress * (atan((pixelX - CenterX) / hFocalLength));
	return yaw; 
}

//Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
double calculatePitch(double pixelY, double  CenterY, double vFocalLength) {
	double pitch = convertToDegress * (atan((pixelY - CenterY) / vFocalLength));
	//just stopped working have to do this:
	//pitch *= -1.0;
	return pitch;
}

double calculateDistance(double heightOfCamera, double heightOfTarget ,double pitch) {
	double heightOfTargetFromCamera = heightOfTarget - heightOfCamera;
	/*
	// Uses trig and pitch to find distance to target
  
    d = distance
    h = height between camera and target
    a = angle = pitch
    tan a = h/d (opposite over adjacent)
    d = h / tan a
                         .
                        /|
                       / |
                      /  |h
                     /a  |
              camera -----
                       d
	*/

	double distance = fabs(heightOfTargetFromCamera / tan((pitch* pi) / 180.0));
	return distance;
}

void threshold(Mat HSV, Scalar low, Scalar high, Mat& threshold){
  inRange(HSV ,low, high, threshold);
}


void searchForMovement(Mat thresholdImage, Mat &cameraFeed) {
	/* Notice how we use the '&' operator for the camerafeed. This is because we wish
	 to take the values passed into the function and manipulate them, rather than just
	 working with a copy. eg. we draw to the cameraFeed in this function which is then
	 displayed in the main() function. */
	bool objectDetected = false;
	Mat temp;
	thresholdImage.copyTo(temp);
		// these two vectors needed for the output of findContours

	vector< vector<Point>> contours;
	vector<Vec4i> hierarchy;
		//findContours of filtered image using openCV findCOntours function
		//findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //retrieves all contours
	findContours(temp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);// retrieves external contours
	
		//if contours vector is not empty, we found some objects
	if (contours.size() > 0) {
		objectDetected = true;
	}
	else { objectDetected = false; }

	if (objectDetected) {
		//largest contour is found at the end of the contours vector
		//we will simply assume that the biggest contour is what we looking for.
		vector< vector<Point>> largestContourVec;
		largestContourVec.push_back(contours.at( contours.size() - 1));
		
		//making a bounding rectagle around the largest contour then find its centriod
		//this will be the object's final estimated position.
		objectBoundingRectangle = boundingRect(largestContourVec.at(0));
		int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width / 2;
		int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height / 2;

		//update the objects position by changing the 'theObject' array values
		theObject[0] = xpos, theObject[1] = ypos;

	}
	//make some temp x and y varibles so we don't have to type out so much
	int x = theObject[0];
	int y = theObject[1];

	//Calculates yaw of contour (horizontal position in degrees)
	//
	//Calculates pitch of contour (Vertical position in degrees)
	//

	//draw some crosshairs on the object
	//rectangle(cameraFeed, objectBoundingRectangle, Scalar(255, 0, 0), 3, 8, 0);
	circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	//draws the rotated rectangle contours and might have to blur to make drawing more 'rectangleish'
	drawContours(cameraFeed, contours, 0, Scalar(255, 0, 0), 1, LINE_AA);
  std::cout << "vision is proccedd"<< std::endl;
}

////////////////// End of OPENCV process ////////////////////////////


static const char* configFile = "/boot/frc.json";

namespace {

unsigned int team;
bool server = false;

struct CameraConfig {
  std::string name;
  std::string path;
  wpi::json config;
  wpi::json streamConfig;
};

struct SwitchedCameraConfig {
  std::string name;
  std::string key;
};

std::vector<CameraConfig> cameraConfigs;
std::vector<SwitchedCameraConfig> switchedCameraConfigs;

wpi::raw_ostream& ParseError() {
  return wpi::errs() << "config error in '" << configFile << "': ";
}

bool ReadCameraConfig(const wpi::json& config) {
  CameraConfig c;

  // name
  try {
    c.name = config.at("name").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "could not read camera name: " << e.what() << '\n';
    return false;
  }

  // path
  try {
    c.path = config.at("path").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "camera '" << c.name
                 << "': could not read path: " << e.what() << '\n';
    return false;
  }

  // stream properties
  if (config.count("stream") != 0) c.streamConfig = config.at("stream");

  c.config = config;

  cameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadSwitchedCameraConfig(const wpi::json& config) {
  SwitchedCameraConfig c;

  // name
  try {
    c.name = config.at("name").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "could not read switched camera name: " << e.what() << '\n';
    return false;
  }

  // key
  try {
    c.key = config.at("key").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "switched camera '" << c.name
                 << "': could not read key: " << e.what() << '\n';
    return false;
  }

  switchedCameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadConfig() {
  // open config file
  std::error_code ec;
  wpi::raw_fd_istream is(configFile, ec);
  if (ec) {
    wpi::errs() << "could not open '" << configFile << "': " << ec.message()
                << '\n';
    return false;
  }

  // parse file
  wpi::json j;
  try {
    j = wpi::json::parse(is);
  } catch (const wpi::json::parse_error& e) {
    ParseError() << "byte " << e.byte << ": " << e.what() << '\n';
    return false;
  }

  // top level must be an object
  if (!j.is_object()) {
    ParseError() << "must be JSON object\n";
    return false;
  }

  // team number
  try {
    team = j.at("team").get<unsigned int>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "could not read team number: " << e.what() << '\n';
    return false;
  }

  // ntmode (optional)
  if (j.count("ntmode") != 0) {
    try {
      auto str = j.at("ntmode").get<std::string>();
      wpi::StringRef s(str);
      if (s.equals_lower("client")) {
        server = false;
      } else if (s.equals_lower("server")) {
        server = true;
      } else {
        ParseError() << "could not understand ntmode value '" << str << "'\n";
      }
    } catch (const wpi::json::exception& e) {
      ParseError() << "could not read ntmode: " << e.what() << '\n';
    }
  }

  // cameras
  try {
    for (auto&& camera : j.at("cameras")) {
      if (!ReadCameraConfig(camera)) return false;
    }
  } catch (const wpi::json::exception& e) {
    ParseError() << "could not read cameras: " << e.what() << '\n';
    return false;
  }

  // switched cameras (optional)
  if (j.count("switched cameras") != 0) {
    try {
      for (auto&& camera : j.at("switched cameras")) {
        if (!ReadSwitchedCameraConfig(camera)) return false;
      }
    } catch (const wpi::json::exception& e) {
      ParseError() << "could not read switched cameras: " << e.what() << '\n';
      return false;
    }
  }

  return true;
}

std::pair<frc::CameraServer*, cs::UsbCamera > StartCamera(const CameraConfig& config) {
  wpi::outs() << "Starting camera '" << config.name << "' on " << config.path
              << '\n';
  frc::CameraServer* inst = frc::CameraServer::GetInstance();
  //cs::UsbCamera camera{config.name, config.path};
  cs::UsbCamera server = inst->StartAutomaticCapture(config.name, config.path);

  server.SetConfigJson(config.config);
  //camera.SetConnectionStrategy(cs::VideoSource::kConnectionKeepOpen);
  return  std::make_pair (inst, server); 
 } 
}



int main(int argc, char* argv[]) {
  
  std::cout << " stage 0" <<std::endl ;
  if (argc >= 2) configFile = argv[1];
  std::cout << " stage 1" ;
  // read configuration
  if (!ReadConfig()) return EXIT_FAILURE;
  
  std::cout << " stage 2" << std::endl;
  // start NetworkTables
  nt::NetworkTableInstance ntinst = nt::NetworkTableInstance::GetDefault();
  //Name of network table - this is how it communicates with robot. IMPORTANT
  std::shared_ptr<NetworkTable> DeadEye = ntinst.GetTable("Vision5572");
  ntinst.StartServer();
  std::cout << " stage 3" << std::endl ;
  
  if (server) {
    std::cout << " stage 4 1" << std::endl ;
    wpi::outs() << "Setting up NetworkTables server\n";
    ntinst.StartServer();
  } else {
    std::cout << " stage 4 0" << std::endl ;
    wpi::outs() << "Setting up NetworkTables client for team " << team << '\n';
    ntinst.StartClientTeam(team);
  }
  
  std::cout << " stage 5" << std::endl ;
  std::vector<frc::CameraServer*> streams;
  std::vector<cs::UsbCamera> cameras;
  std::pair<frc::CameraServer*, cs::UsbCamera> cams;
  
  //start cameras
  std::cout << " stage 6" << std::endl ;
  for (const auto& config : cameraConfigs){
    std::cout << " stage 6 6" << std::endl ;
     cams = StartCamera(config);
     streams.push_back(cams.first);
     cameras.push_back(cams.second);
  }
  
  std::cout << streams.size() << std::endl;
  std::cout << " stage 7" << std::endl ;
  //Get the first camera
  frc::CameraServer* webcam = streams.at(0);
  std::cout << " stage 8" << std::endl ;
  cs::UsbCamera cameraServer = cameras[0];
  std::cout << " stage 8 1" << std::endl ;
  
  // Start thread reading camera
  WebcamVideoStream cap(cameraServer , webcam , imageWidth, imageWidth);
  std::cout << " stage 9" << std::endl ;
  std::cout << cap.getError() << std::endl;
  //cap.start();
  std::cout << " stage 10" << std::endl ;
  std::cout << cap.getError() << std::endl;
  // // (optional) Setup a CvSource. This will send images back to the Dashboard
  // // Allocating new images is very expensive, always try to preallocate
  cv::Mat img;
  // //Start thread outputing stream
  VideoShow streamViewer (imageWidth, imageHeight, webcam, img);
  //streamViewer.start();
  
  cv::Mat imgHSV;
  cv::Mat imgThreshold; 

  while(true){
  std::cout << " stage 0" << std::endl;
  cap.update();
  img = cap.read();
  std::cout << " stage 1" << std::endl;
  
  std::cout << " stage 2" << std::endl;
  std::cout << cap.getError() << std::endl;
  cv::cvtColor(img, imgHSV, CV_BGR2HSV);
  
  std::cout << " stage 3" << std::endl;
  threshold(imgHSV, tlow, tHigh, imgThreshold);
  searchForMovement(imgThreshold, img);
  std::cout << " stage 4" << std::endl;
  
  streamViewer.frame = imgThreshold;
  streamViewer.show();

  std::cout << " stage 5" << std::endl;
  ntinst.Flush();
  std::cout << " stage 6" << std::endl;
  }

}

