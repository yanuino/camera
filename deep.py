import numpy as np
import cv2



def synset():
    all_rows = open('./models/synset_words.txt').read().strip().split("\n")
    classes = [r[r.find(' ') + 1:] for r in all_rows]

    return classes

def image_classification(net, img):
    # use Pre-trained network

    blob = cv2.dnn.blobFromImage( img, 1, size=(224,224), mean=(104, 117, 123))
    net.setInput(blob)
    outp = net.forward()

    return outp

def main():

    classes = synset()
    net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt', './models/bvlc_googlenet.caffemodel')


    cap = cv2.VideoCapture(1)



    while(True):
        ret, frame = cap.read()

        outp = image_classification(net, frame)

        r = 1
        for i in np.argsort(outp[0])[::-1][:5]:
            txt = ' "%s" probability "%.3f" ' % (classes[i], outp[0][i] * 100)
            cv2.putText(frame, txt, (0, 25 + 40*r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            r += 1

        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break    

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()