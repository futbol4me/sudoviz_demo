#@markdown functions to break down a sudoku image into its board positions and utilities to display these images
## set of sudoku image processing ultilities
## ed lee

from sudo_euler96 import Solver, Board
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf

path = "/content/sudoviz_demo/"
spath = path + "sudoku_board_images/"
tpath = path + "training_data/"
mpath = path + "models/"

def load_grayscale(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img

def show(grayimg, show_axis=False):
    plt.imshow(grayimg, cmap='gray', vmin=0, vmax=255)
    if not show_axis:
        plt.axis('off')
    plt.show()

def locate_sudoku_image_from(path, inner=True):
    ## use cv2 contours
    ## https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    ## https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    imc = cv2.imread(path)
    img = load_grayscale(path)
    imgb = cv2.GaussianBlur(img , (7, 7), 3)
    thresh = cv2.bitwise_not(
                cv2.adaptiveThreshold(imgb,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY,
                                      11,
                                      2)
    )

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # https://docs.opencv.org/4.x/dc/dcf/tutorial_js_contour_features.html
    # https://stackoverflow.com/questions/62274412/cv2-approxpolydp-cv2-arclength-how-these-works
    big_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in big_contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # look for 4 points in our contour
        if len(approx) == 4:
            board_contours = approx.reshape(4,2)
            # check if rougly square
            # opencv coordinates in x,y not r,c array indexing
            x1, x2, x3, x4 = sorted(board_contours[:,0])
            y1, y2, y3, y4 = sorted(board_contours[:,1])

            return i2square(img[y2:y3,x2:x3])
    # not found
    assert False, "sudoku board boundaries not found"


def i2square(img):
    ''' transforms any image to square dimensions '''
    dim = max(img.shape)
    return cv2.resize(img,(dim,dim),interpolation = cv2.INTER_CUBIC)

def contrast(grayimg):
    thresh = cv2.threshold(grayimg, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh

def clear_gridlines(gray,black_background=True):
    ''' clears gridline areas of a sudoku to black... assumes background image is black (reversed)
    '''
    gray = gray.copy()
    LINE_PCT = 0.25
    dim = gray.shape[0]
    widthf=dim/9
    ldim = int(dim * LINE_PCT / 10 +.5)
    hblock = np.zeros((ldim,dim), np.uint8)
    vblock = hblock.reshape((dim,ldim))
    gray[:ldim,:] = hblock
    gray[dim-ldim:,:]=hblock
    for i in range(1,9):
        r = int(i*widthf+.5)-(ldim//2)
        gray[r:r+ldim,:]=hblock
    gray[:,:ldim] = vblock
    gray[:,dim-ldim:]=vblock
    for i in range(1,9):
        c = int(i*widthf+.5)-(ldim//2)
        gray[:,c:c+ldim]=vblock
    return gray


def full_square(img, r,c ):
    ''' takes grayscale image and r,c parameters and returns corresponding
        section of the image.  may contain borders
    '''
    BORDERFACTOR = 0.015

    dim = img.shape[0]

    borderpixels = int(dim * BORDERFACTOR)
    width = dim - borderpixels
    boxdim = width/9
    oneborder = borderpixels // 2

    i = int(oneborder + r * boxdim)
    j = int(oneborder + c * boxdim)

    return img[i:min(i+int(boxdim+.5),dim), j:min(dim,j+int(boxdim+.5))]

def get_square(img,r,c,shrink=None,jitter=False):
    ''' gets img square at r,c  shrink reduces the frame of the image, effectively zooming in
        this calls full_square then cuts away the edges of the image to focus on the center
    '''

    sq=full_square(img,r,c)
    if shrink is None:
        return sq

    if jitter:
        shrink = shrink + (np.random.rand()-0.5)*.25
    dim = sq.shape[0]
    pad = int(dim * shrink / 2)
    if not jitter:
        jr, jc = 0,0
    else:
        padrange = int(pad /2.5)
        jr = random.randint(-padrange,padrange)
        jc = random.randint(-padrange,padrange)
    padr = pad + jr
    padc = pad + jc
    crop = sq[padr:dim-padr,padc:dim-padc]
    a,b = crop.shape
    if a==b:
        return crop
    return cv2.resize(crop,(min(a,b),min(a,b)))



def replace_square(img, r,c , value, digits=None):
    ''' takes greyscale image and rc parameters and replaces part of image with value; returns the
        mutated image.
    '''
    BORDERFACTOR = 0.015
    SHRINK = .35

    img = img.copy()
    dim = img.shape[0]

    if digits is None:
        digits = build_digits_dict()

    bpixels = int(dim * BORDERFACTOR)
    width = dim - bpixels
    boxdim = width/9
    b1 = bpixels // 2
    shrinkpix = int(boxdim*SHRINK)//2

    i = int(b1 + r * boxdim)
    j = int(b1 + c * boxdim)

    rl, rh, cl, ch = i + shrinkpix, i+int(boxdim+.5)-shrinkpix, j+shrinkpix, j+int(boxdim+.5)-shrinkpix


    cell_dims = rh-rl,ch-cl
    img[rl:rh, cl:ch] = cv2.resize(digits[value],cell_dims)
    return img

def build_digits_images()-> dict:
    ''' utility to build 28x28 digit images used in replace_square
        ...needs file training0.png
    '''
    g = locate_sudoku_image_from(tpath+'training0.png')

    keymap_for_train0 = {
            1:(1,7), 2:(2,7), 3:(6,3),
            4:(1,5), 5:(1,6), 6:(1,3),
            7:(2,6), 8:(2,5), 9:(1,8), 0:(1,4),
    }
    return {digit: cv2.resize(
                            get_square(g, *keymap_for_train0[digit], shrink=.2),
                            (28,28)
                            ) for digit in range(10)
            }

DIGIT_IMAGES = build_digits_images()


def color2gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

def view28(x):
    ''' utility function to display 12 rows x whatever columns of keras-formatted training data (shape and magnitude)
    '''
    import math
    n,r,c,_ = x.shape
    x = x.reshape(n,r,c)
    cols = int(math.ceil(n/12))
    img = np.zeros((12 * 30, cols * 30))
    i = 0
    for g in x:
        c,r = divmod(i,12)
        img[r*30:r*30+28,c*30:c*30+28]=g
        i += 1
    show(img)


#@markdown functions to build our down training data... currently requires 10 specific training files in training directory.

# import sys
# sys.path.insert(0, path[:-1])
# from sudo_train import TRAINING_DB
#imports training data labels for training images.
#TRAINING_DB is dict of {filename: list of labels}

TRAINING_DB={'training1.png': [
    '263147985',
    '547893261',
    '891562473',
    '136485729',
    '759621834',
    '428379156',
    '385714692',
    '612958347',
    '974236518'
    ],
    'training2.jpg': [
    '000000008',
    '180002300',
    '060057001',
    '070960000',
    '090704010',
    '000081040',
    '600240080',
    '004500093',
    '500000000'
    ],
    'training3.png': [
    '869135742',
    '521478936',
    '347926185',
    '678341529',
    '134592867',
    '952867314',
    '793284651',
    '416753298',
    '285619473'
    ],
    'training4.png':[
    '793864512',
    '654123897',
    '128975364',
    '586219743',
    '279346185',
    '341758629',
    '835491276',
    '912637458',
    '467582931'],
    'training5.png':[
    '163297458',
    '275348196',
    '894651237',
    '349862571',
    '751934682',
    '628175349',
    '532786914',
    '486519723',
    '917423865'],
    'training6.png':[
    '185926700',
    '273148605',
    '409507182',
    '891754236',
    '007269800',
    '026800400',
    '938602514',
    '702000368',
    '654381927'],
    'training7.png':[
    '672531948',
    '831649275',
    '549827631',
    '157496823',
    '396218754',
    '284754169',
    '763185492',
    '415962387',
    '928374516'],
    'training8.jpg':
    ['700000000',
     '001400370',
     '003062000',
     '050006980',
     '104000006',
     '000003000',
     '009070001',
     '000094800',
     '240010009'],
    'training9.png':
    ['487691235',
     '521348679',
     '963275184',
     '278954361',
     '146783592',
     '359162748',
     '812436957',
     '734519826',
     '695827413'],
    'training10.jpg':
     ['269378105', 
      '581029673', 
      '403560298', 
      '090832517', 
      '812745936', 
      '350196804', 
      '135980762', 
      '728613459', 
      '946257301']

   }


def build_data_set(tdict:dict, db:list = [], jitter=False, mypath=False) -> list:
    '''
    reads dict of sudoku image files and labels (coded as list of strings per euler96)
    returns db, which is a list of tuples with np.array image and label
    this function can be rerun to add  more training data...
    then this list should be further processed before training
    '''
    db=db.copy()
    for file in tdict:
        label_list = tdict[file]
        filepath = spath if mypath else tpath
        gray = locate_sudoku_image_from(tpath+file)
        # invert, high contrast, cleargridlines
        cgray = clear_gridlines(contrast(gray))
        for r in range(9):
            for c in range(9):
                for i in range(19*jitter+1): #repeat 20x if jitter
                    label = int(label_list[r][c])
                    s = cv2.resize(get_square(cgray,r,c,shrink=.2,jitter=jitter),(28,28),interpolation = cv2.INTER_CUBIC)
                    db.append((s,label))
                    if label == 0 and i>3:
                        break  #no need to repeat blanks too often
    return db

def build_training_set_for_tf(db:list, zeros=True):
    ''' inputs db from build_data_set and forms to tuple of arrays suitable for tensorflow training
    '''
    xtrain = np.array([im for im,label in db if zeros or label])

    d1,d2,d3=xtrain.shape
    xtrain = xtrain.reshape(d1,d2,d3,1) # add a dimension ??? not sure why; perhaps for bias
    labels = tf.keras.utils.to_categorical(np.array([label for im,label in db if zeros or label]))
    return xtrain, labels


#@markdown functions that connect imaging processing to sudoku logic
def read_sudoku_array(impath, model=None, display=True):
    ''' accepts filename and tensorflow model... gets image and reads into a 9x9 array
    '''

    if model is None:
        model=tf.keras.models.load_model(mpath+'sudoku_model_9.keras')
    gray = locate_sudoku_image_from(impath)
    if display:
      show(gray)
    m=model
    board = clear_gridlines(contrast(gray))
    p = []
    for i in range(9):
        for j in range(9):
            s = cv2.resize(get_square(board,i,j,shrink=.2),(28,28),interpolation = cv2.INTER_CUBIC)
            d = s.astype('float')
            d = tf.keras.preprocessing.image.img_to_array(d)
            p.append(d)
    pnp = np.array(p)
    p_array = m.predict(pnp,verbose=0)
    p=np.argmax(p_array,axis=1).reshape(9,9)
    return p

def read_sudoku_strlist(imgfile, model=None):
    ''' covenience function to translate an imagefile into string representation
    '''
    a = read_sudoku_array(imgfile, model)
    s = ''
    for r in a:
        for c in r:
            s+=str(c)
        s+=','
    return s[:-1].split(',')

def read_sudoku_board(imgfile,model=None, display=True):
    ''' reads file into a Board object
    '''
    return Board(read_sudoku_array(imgfile,model,display))

def solve_sudoku_image(imgfile, model=None, history=False, randomize=False):
    initial_board = read_sudoku_board(imgfile,model,False)
    gray = locate_sudoku_image_from(imgfile)
    result = Solver.solve(initial_board,history,randomize=randomize)
    og = gray.copy()
    if history:
      solved_board,_ = result[-1]
    else:
      solved_board = result
    diff = (solved_board.board - initial_board.board).reshape(9,9)
    for i in range(9):
        for j in range(9):
            if diff[i,j]:
                gray = replace_square(gray,i,j,diff[i,j],DIGIT_IMAGES)

    dim = og.shape[0]
    bar = 255-np.zeros((dim,dim//20))
    show(np.hstack((og,bar,gray)))
    return result




