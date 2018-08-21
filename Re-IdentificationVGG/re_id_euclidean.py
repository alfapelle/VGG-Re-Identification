# Overlapping and Euclidean distance with variable Tolerance with FC6
from VGG_Model import *
from VGG_Utils import *
from BB_Utils import *
import cv2

print("Loading Vgg Model")
print('. . .')
face_cascade = cv2.CascadeClassifier("FaceDetect-master/haarcascade_frontalface_default.xml")
feature_model = load_model('vgg_model_FC6.h5')
# feature_model = vgg_model()
print("Vgg Model Loading Complete!")

choice = input('Choose the source: WEBCAM (1) or VIDEO (2)')
choice = int(choice)
if choice == 1:
    cap = cv2.VideoCapture(0)

elif choice == 2:
    video_path = input('Insert video\'s path: ')
    cap = cv2.VideoCapture(video_path)

else:
    print("Wrong choice")
    exit(0)

t = 0
max_val = 15
# to set
toll = list()
mean_dist = list()
min_dist = list()
iou_perc = 0.85

dict_faces = {}
last_pos = list()
last_l_id = list()

while True:
    pos = list()
    l_id = list()
    # Capture frame-by-frame
    ret, frame = cap.read()
    f_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print("frame num ", t)

    # resize cap
    ret = cap.set(3, 240)
    ret = cap.set(4, 320)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(16, 16))
    gray = clahe.apply(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(50, 50)
    )

    print(" ")

    im = frame
    width = im.shape[1]
    height = im.shape[0]

    for i in range(0, len(faces)):

        (x, y, w, h) = faces[i, :]
        center_x = x + w / 2
        center_y = y + h / 2
        b_dim = min(max(w, h) * 1.2, width, height)

        y1 = int(center_y - b_dim / 2)
        y2 = int(center_y + b_dim / 2)
        x1 = int(center_x - b_dim / 2)
        x2 = int(center_x + b_dim / 2)

        # to avoid the crash of cv2.resize
        if y1 < 0:
            y1 = 0

        if x1 < 0:
            x1 = 0

        # crop face to (224,224)
        crop_im = im[y1:y2, x1:x2]
        crop_im = cv2.resize(crop_im, (224, 224))

        f = features(feature_model, crop_im, transform=True)

        pos.append([x1, y1, x2, y2])
        overlap = False
        if len(last_pos) > 0:
            for lp in range(0, len(last_pos)):
                iou = get_iou(pos[i], last_pos[lp])

                if iou > iou_perc:

                    dict_faces, num_key = bb_update_dict_euclidean(f, dict_faces, min_dist, max_val, last_l_id[lp])
                    l_id.append(num_key)
                    overlap = True
                    break
            if not overlap:

                dict_faces, num_key = init_update_dict_euclidean(f, dict_faces, toll, min_dist,max_val)

                l_id.append(num_key)

        if not last_pos:
            dict_faces, num_key = init_update_dict_euclidean(f, dict_faces, toll, min_dist, max_val)
            l_id.append(num_key)

        toll, mean_dist, min_dist = update_toll_euclidean(dict_faces, num_key, toll, mean_dist, min_dist)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x-1, y), (x + 60, y - 25), (0, 255, 0), thickness=cv2.FILLED)

        cv2.putText(frame, 'Id: '+str(num_key+1), (x+1, y-8),  cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), lineType=cv2.LINE_AA)

    cv2.imshow('Video', frame)

    print("Id:    N.descriptors: ")
    print(" ")

    for key in dict_faces.keys():
        print(key+1, "    ",  len(dict_faces[key]))
    print("-----------------------")
    print(" ")

    t = t+1

    last_pos = pos
    last_l_id = l_id

    cap.set(cv2.CAP_PROP_POS_FRAMES, value=5.0+f_num)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()

