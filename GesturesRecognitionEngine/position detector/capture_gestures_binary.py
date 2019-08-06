from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse

frame_processed = 0
score_thresh = 0.2
im_num=170
   
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0
    cap_flag=True
    fold="index_and_thumb"

    cv2.namedWindow('Hand Tracker', cv2.WINDOW_NORMAL)
    
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    
    while True:
        frame= video_capture.read()[1]
        frame = cv2.flip(frame,1)
        index += 1
        key=cv2.waitKey(1) #c-append coordinates r-train SVM q-exit
        
        input_q=(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = input_q
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            X,Y,ROI=detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            # add frame annotated with bounding box to queue
            output_q=frame
            frame_processed += 1
            if key & 0xFF == ord('c'):
                cap_flag=False
            if cap_flag==True:
                if (X,Y)==(0,0):
                    continue
                else:
#                    hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
#                    imgCrop = ROI[23:28, 23:28]
#                    hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
#                    hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
#                    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
#                    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
#                    dst1 = dst.copy()
#                    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#                    cv2.filter2D(dst,-1,disc,dst)  
#                    blur = cv2.GaussianBlur(dst, (11,11), 0)
#                    blur = cv2.medianBlur(blur, 15)
#                    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#                    thresh = cv2.merge((thresh,thresh,thresh))
                    cv2.imwrite("gestures_train_images/" + str(fold) + "/" + str(im_num) + ".jpg",ROI)
                    im_num=im_num+1
                    print(im_num)
                            
                
        else:
            output_q=frame
            
        output_frame = output_q

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        # print("frame ",  index, num_frames, elapsed_time, fps)

        if (output_frame is not None):
            if (args.display > 0):
#                if (args.fps > 0):
#                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
#                                                     output_frame)
                cv2.imshow('Hand Tracker', output_frame)
                if key & 0xFF == ord('q'):
                    break  
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))
        else:
            # print("video end")
            break

    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    sess.close()
    video_capture.stop()
    cv2.destroyAllWindows()

del video_capture