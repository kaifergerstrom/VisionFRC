package frc.team612;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.core.MatOfPoint;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;
import javax.swing.*;

public class Main {

    /* Values for HSV filtering */
    private static int H = 0;
    private static int S = 115;
    private static int V = 45;
    private static int H2 = 126;
    private static int S2 = 255;
    private static int V2 = 255;

    private static int kernel_size = 5;
    private static int lockon_offset = 75;
    private static boolean lockon = false;

    private static int contour_x;
    private static int contour_y;
    private static int contour_width;
    private static int contour_height;

    private static int known_distance = 36;  // A known factor for distance to calculate focal length in inches
    private static int known_object_width = 490;  // The width of the contour at the known distance in pixels
    private static int known_object_height = 76;  // The height of the contour at the known distance in pixels

    private static double known_real_width = 28.5;  // The actual width of object in inches
    private static double focal_length = (known_object_width * known_distance) / known_real_width;  // Formula to calculate focal length, unit conversion factors

    private static JFrame window = new JFrame("Vision");
    private static JLabel lbl = new JLabel();

    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);  // Load and initialize OPENCV libraries
        VideoCapture camera = new VideoCapture(0);  // Initialize video capture

        while(true) {  // While loop for each of the camera frames

            Mat frame = new Mat();  // Define main frame
            Mat hsv = frame.clone(); // Create HSV frame for filtering
            Mat mask = frame.clone();  // Create Mask frame
            Mat erosion = frame.clone();  // Create Erosion frame

            /* Code for creating kernel, some random math */
            Mat kernel = new Mat(); // Define kernel frame (Tiny little filtering blob)
            Mat ones = Mat.ones(kernel_size, kernel_size, CvType.CV_32F );
            Core.multiply(ones, new Scalar(1/(double)(kernel_size*kernel_size)), kernel);

            List<MatOfPoint> contours = new ArrayList<>();  // Create a list to append contour values to

            if (camera.read(frame)) {  // Only run if there is a frame to see

                int screen_width = frame.width();
                int screen_height = frame.width();
                int screen_center_x = screen_width / 2;

                Imgproc.GaussianBlur(frame, frame, new Size(5,5), 0);  // Blur the frame to decrease noise

                Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);  // Convert frame into HSV readable
                Core.inRange(hsv, new Scalar(H, S, V), new Scalar(H2, S2, V2), mask);  // Filter out HSV ranges and apply to a mask
                Imgproc.erode(mask, erosion, kernel);  // Erode the mask to reduce more noise

                Imgproc.findContours(erosion, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);  // Find all the contours in image and append to list

                if (contours.size() > 0) {  // Only run if there is a contour to be seen

                    /* Function to sort through contours and find biggest value */
                    int maxValIdx = find_biggest(contours);

                    /* Some code that finds the bounding box of the biggest contour */
                    MatOfPoint2f approxCurve = new MatOfPoint2f();
                    MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(maxValIdx).toArray());
                    double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02;
                    Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);
                    MatOfPoint points = new MatOfPoint(approxCurve.toArray());
                    Rect rect = Imgproc.boundingRect(points);  // The returned bounding box dimensions

                    // Return values of box to static variables for later use
                    contour_x = rect.x;
                    contour_y = rect.y;
                    contour_width = rect.width;
                    contour_height = rect.height;

                    int contour_center_x = contour_x + contour_width / 2;  // Calculate center x point of contour box
                    int contour_center_y = contour_y + contour_height / 2; // Calculate center y point of contour box

                    lockon = false;  // As default, lock on to false
                    if (contour_center_x > screen_center_x + lockon_offset) { // If the contour x point is less than the range then its on the left side - Do not lock on
                        System.out.println("Right");
                        Imgproc.rectangle(frame, rect.tl(), rect.br(), Scalar.all(255), 2, 8, 0);  // Draw the rectangle!
                    } else if (contour_center_x < screen_center_x - lockon_offset) { // If the contour x point is greater than the range then its on the right side - Do not lock on
                        System.out.println("Left");
                        Imgproc.rectangle(frame, rect.tl(), rect.br(), Scalar.all(255), 2, 8, 0);  // Draw the rectangle!
                    } else { // else that means the contour is in the range to allow for lock on
                        System.out.println("Center");
                        lockon = true;
                    }

                    if (lockon) {
                        double distance = (known_real_width * focal_length) / contour_width;
                        Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0,0,255), 2, 8, 0);  // Draw the rectangle!
                        Imgproc.putText(frame, "Target locked | target is " + distance + " in away", new Point(10, 50), 3, 0.5, new Scalar(0, 0, 255, 255), 2);
                        System.out.println("Target locked | target is " + distance + " in away");
                    }

                    Imgproc.circle(frame, new Point(contour_center_x, contour_center_y), 5, new Scalar(0,0,255));

                } else {
                    System.out.println("No contours!");  // If no contours let them know!
                }

                // Some lines to show range for "lock on"
                Imgproc.line(frame, new Point(screen_center_x + lockon_offset, 0), new Point(screen_center_x + lockon_offset, screen_height), new Scalar(255,0,0), 1);
                Imgproc.line(frame, new Point(screen_center_x - lockon_offset, 0), new Point(screen_center_x - lockon_offset, screen_height), new Scalar(255,0,0), 1);

                displayImage(Mat2BufferedImage(frame));  // Display frame in JFrame
            }
        }
    }

    private static int find_biggest(List<MatOfPoint> list_input) {  // Function that get input of contour list a returns biggest index
        double maxVal = 0;
        int maxValIdx = 0;
        for (int contourIdx = 0; contourIdx < list_input.size(); contourIdx++) {
            double contourArea = Imgproc.contourArea(list_input.get(contourIdx));
            if (maxVal < contourArea) {
                maxVal = contourArea;
                maxValIdx = contourIdx;
            }
        }
        return maxValIdx;
    }

    private static BufferedImage Mat2BufferedImage(Mat m) {
        // Fastest code
        // output can be assigned either to a BufferedImage or to an Image

        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels()*m.cols()*m.rows();
        byte [] b = new byte[bufferSize];
        m.get(0,0,b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    private static void displayImage(Image img2) {
        ImageIcon icon = new ImageIcon(img2);
        window.setLayout(new FlowLayout());
        window.setSize(img2.getWidth(null)+50, img2.getHeight(null)+50);
        lbl.setIcon(icon);
        window.add(lbl);
        window.setVisible(true);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

}
