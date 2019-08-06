'''
This code detects the hand in real time and feeds the coordinates of the hand in 200x300 sized frame
to the 20x20 trained SOM. The SOM's output for 9 different positions is used to train the MLP. 
Once calibrated, the position of the hand is inferred based on MLP's output.
'''
from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import numpy as np
from configure import Config
#from sklearn.metrics import accuracy_score

frame_processed = 0
score_thresh = 0.2
SOM_dim=(20,20)
weights = np.load(Config.SOM_weights_path)
count=1

def SOM_response(output_size, X, wt, sig = 80):
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
    coordinates=[]
    labels=[]
    order=[1,3,5,4,2,7,6,9,8,1,5,9,3,2,4,6,8,7]
    i=0
    run_flag=False
    pos=0

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
                if (X,Y)==(0,0):
                    continue
                else:
                    coordinates.append([X,Y])
                    label=np.zeros([1,9])
                    label[:,order[i]-1]=1
                    labels.append(label)
                    i+=1
                    print(count)
                    count=count+1
            if key & 0xFF == ord('s'):
                MLP_input=[]
                coordinates=np.array(coordinates)
                for coord in coordinates:
                    X=coord
                    X = X.reshape(2, order='F')
                    SOM_output = np.array(SOM_response(SOM_dim,X, weights))
                    SOM_output=np.reshape(SOM_output,[400])
                    MLP_input.append(SOM_output)
                MLP_input=np.array(MLP_input)
                labels=np.reshape(np.array(labels),[18,9])
            if key & 0xFF == ord('t'):
                s = tf.InteractiveSession()
                
                num_classes = 9
                num_features = 400
                num_output = 9
                num_layers_0 = 512
                num_layers_1 = 256
                starter_learning_rate = 0.001
                regularizer_rate = 0.1
                
                
                # Placeholders for the input data
                input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
                input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')
                
                
                ## for dropout layer
                keep_prob = tf.placeholder(tf.float32)
                
                
                ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
                weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
                bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
                
                weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
                bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
                
                weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_output], stddev=(1/tf.sqrt(float(num_layers_1)))))
                bias_2 = tf.Variable(tf.random_normal([num_output]))
                
                
                ## Initializing weigths and biases
                hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
                hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
                
                hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0,weights_1)+bias_1)
                hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
                
                predicted_y = tf.sigmoid(tf.matmul(hidden_output_1_1,weights_2) + bias_2)
                
                
                ## Defining the loss function
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y,labels=input_y)) \
                        + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))
                
                
                ## Variable learning rate
                learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
                
                
                ## Adam optimzer for finding the right weight
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,
                                                                                         bias_0,bias_1,bias_2])
                    
                ## Metrics definition
                correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(predicted_y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                ## Training parameters
                batch_size = 1
                epochs=20
                dropout_prob = 0.6
                
                result=[]
                
                s.run(tf.global_variables_initializer())
                for epoch in range(epochs):    
                    arr = np.arange(MLP_input.shape[0])
                    np.random.shuffle(arr)
                    for index in range(0,MLP_input.shape[0],batch_size):
                        s.run(optimizer, {input_X: MLP_input[arr[index:index+batch_size]],
                                          input_y: labels[arr[index:index+batch_size]],
                                        keep_prob:dropout_prob})
                run_flag=True
            if run_flag==True:
                get_coord=np.array([[X,Y]])
                SOM_output = np.array(SOM_response(SOM_dim,get_coord, weights))
                SOM_output=np.reshape(SOM_output,[400])
                MLP_test=np.reshape(np.array(SOM_output),[1,400])
                
                pos=s.run(predicted_y, {input_X: MLP_test,keep_prob:1}).argmax(1)
             
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
                if (args.fps > 0):
#                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
#                                                     output_frame)
                    if run_flag==True:
                        detector_utils.draw_fps_on_image("POSITION : " + str(int(pos)+1),output_frame)
                    else:
                        if (count-1)==len(order):
                            cv2.putText(output_frame,"Press S and T to continue", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                        else:
                            cv2.putText(output_frame, "Go to position "+str(int(order[count-1])), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
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