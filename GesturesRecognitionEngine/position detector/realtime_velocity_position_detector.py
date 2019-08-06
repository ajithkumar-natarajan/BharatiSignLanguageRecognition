from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm
from GaussianStatistics import *
from configure import Config

Gstat  = GaussianStatistics()
config = Config()

frame_processed = 0
score_thresh = 0.2
    
def nstnbridx(output_size,weights,row_data):
        initial_dis = float("inf")
        index_bmu   = [0, 0]
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                dist_neuron = np.linalg.norm(row_data-weights[i, j, :])
                if dist_neuron < initial_dis:
                    initial_dis = dist_neuron
                    index_bmu = [i, j]# Best matching unit (BMU)
        return np.array(index_bmu)

def nbridxdis(nrad,output_size,index_bmu):
    nbrind = np.zeros(((2*nrad + 1)**2, 2))
    for i in range(2*nrad + 1):
        for j in range(2*nrad + 1):
            ix = i*(2*nrad+1) + j
            [i - nrad, j- nrad] + index_bmu
            nbrind[ix,:] = [i - nrad, j- nrad] + index_bmu

    # print (nbrind, (nbrind[:,1] >= 0))
    nbrind = nbrind[np.where((nbrind[:,0] >= 0) * (nbrind[:,1] >= 0))]
    nbrind = nbrind[np.where((nbrind[:,0] < output_size[0]) * (nbrind[:,1] < output_size[1]))]

    mm, _  = nbrind.shape
    nbrdist = np.zeros(mm)
    for i in range(mm):
        diff = nbrind[i,:] - index_bmu
        nbrdist[i] = diff.dot(diff)
    return nbrind, nbrdist
    
def response(output_size, X, wt, sig = 20):
    """
    """
    x = X.flatten('F')
    assert len(x) == wt.shape[2]
    Y = np.zeros(output_size)

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            diff = wt[i, j, :] - x
            dis  = diff.dot(diff)
            Y[i, j] = np.exp(-1.*dis / sig**2)
    return Y

def Normalize(mat):
    mat = mat / np.sum(abs(mat))
#    mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
    return mat
            
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
    coord =[]
    lab=1
    cap_flag=False
    Coord_old=[0,0]
    pos_all=[]
    vel=[]
    X,Y=0,0
    Thold=15
    som_train_flag=True
    som_test_flag=False
    som_flag=False
    pos_flag=False
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
            X,Y,ROI,det_flag=detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame,X,Y)
            # add frame annotated with bounding box to queue
            output_q=frame
            frame_processed += 1
            if key & 0xFF == ord('c'):
                cap_flag=True
            if key & 0xFF == ord('r'):
                coord=np.array(coord)
                np.save('Coordinates_labels.npy',coord)
                som_flag=True
            if key & 0xFF == ord('s'):
                cap_flag=False
                lab=lab+1
            if cap_flag==True:
                if det_flag==False:
                    continue
                else:
                    coord.append([X,Y,lab])
        else:
            output_q=frame
            
        output_frame = output_q

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        # print("frame ",  index, num_frames, elapsed_time, fps)
        
        if som_flag==True:
            coord=np.load('Coordinates_labels.npy')
            coordinates=coord[:,0:2]
            labels=coord[:,2]
            positions=[]
            for i in range(3):
                for j in range(3):
                    positions.append([i,j])
                    
            if som_train_flag:
                epochs=1000
                iterations = len(coordinates)*epochs
                output_size = [3,3]
                input_num = coordinates.shape[1]
                nrad  =1
                sig   = 3
                alpha = 0.05
                weights = np.random.rand(output_size[0], output_size[1], input_num)
                    
                for itter in tqdm(range(iterations)):
                    initial_dis = float("inf")
            #        row_index = np.random.randint(len(data))
                    row_index = np.random.randint(len(coordinates))
                    learning_rate = alpha*np.exp(-itter/iterations)
            #        row_data = data[row_index]
                    row_data = coordinates[row_index]
            #        bmu_idx  = nstnbridx(output_size,weights,row_data)
                    bmu_idx = np.array(positions[int(labels[row_index]-1)])
                    nbrind, nbrdist = nbridxdis(nrad,output_size,bmu_idx)
                    mx, _ = nbrind.shape
                    for i in range(mx):
                        idx = nbrind[i,:]
                        wt  = weights[int(idx[0]), int(idx[1]), :]
                        diff = row_data - wt
                        dis  = nbrdist[i]/sig **2
                        delta = learning_rate*np.exp(-dis)*diff
                        weights[int(idx[0]), int(idx[1]), :] = delta + wt
            #            weights=Normalize(weights)
                print ("SOM Training done!!...")
                np.save(config.SOM_weights_path, weights)
                som_train_flag=False
                som_test_flag=True
                
            if som_test_flag:
#                display = True
                weights = np.load(config.SOM_weights_path)
                X_som=np.array([X,Y])
                X_som = X_som.reshape(2, order='F')
                y_som = response(output_size,X_som, weights)
                y_som=np.reshape(y_som,[9])
                win_ind=np.where(y_som==y_som.max())
                output=int(win_ind[0])+1
                Coord=[X,Y]
                
                vel_cur=pdist([Coord_old,Coord])
                Coord_old=Coord
                if det_flag==False:
                    vel.append(Thold+10)
                else:
                    if vel_cur<Thold:
                        if pos_flag==True:
                            pos_all.append(output)
                            pos_flag=False
                    if vel_cur>Thold+20:
                        pos_flag=True
                    vel.append(vel_cur)
#                if display:
#                    plt.imshow(y_som)
#                    plt.show()

        if (output_frame is not None):
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)

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
