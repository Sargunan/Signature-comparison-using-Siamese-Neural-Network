# from pathlib import Path
import json, pdb, os, numpy as np, cv2, threading, math #collections, random
# import pickle, sys, itertools, string, sys, re, datetime, time, shutil, copy
from urllib.request import urlopen
# from tempfile import NamedTemporaryFile

import torch
from torch import nn, cuda, backends, FloatTensor, LongTensor, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from model.SigNet import main

from cocoapp import app

cats = {
    1: 'ground',
    2: 'coconut_tree'
}

id2cat = list(cats.values())
sz = 224

def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


# getting val_tfms to work without fastai import

from enum import IntEnum

class TfmType(IntEnum):
    """ Type of transformation.
    Parameters
        IntEnum: predefined types of transformations
            NO:    the default, y does not get transformed when x is transformed.
            PIXEL: x and y are images and should be transformed in the same way.
                   Example: image segmentation.
            COORD: y are coordinates (i.e bounding boxes)
            CLASS: y are class labels (same behaviour as PIXEL, except no normalization)
    """
    NO = 1
    PIXEL = 2
    COORD = 3
    CLASS = 4

class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3
    GOOGLENET = 4
    
class ChannelOrder():
    '''
    changes image array shape from (h, w, 3) to (3, h, w). 
    tfm_y decides the transformation done to the y element. 
    '''
    def __init__(self, tfm_y=TfmType.NO): self.tfm_y=tfm_y

    def __call__(self, x, y):
        x = np.rollaxis(x, 2)
        #if isinstance(y,np.ndarray) and (len(y.shape)==3):
        if self.tfm_y==TfmType.PIXEL: y = np.rollaxis(y, 2)
        elif self.tfm_y==TfmType.CLASS: y = y[...,0]
        return x,y
    
class Transforms():
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms + [crop_tfm, normalizer, ChannelOrder(tfm_y)]
    def __call__(self, im, y=None): return compose(im, y, self.tfms)
    def __repr__(self): return str(self.tfms)    
    
def A(*a): return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

class Denormalize():
    """ De-normalizes an image, returning it to original format.
    """
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m


class Normalize():
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original image """
    def __init__(self, m, s, tfm_y=TfmType.NO):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
        self.tfm_y=tfm_y

    def __call__(self, x, y=None):
        x = (x-self.m)/self.s
        if self.tfm_y==TfmType.PIXEL and y is not None: y = (y-self.m)/self.s
        return x,y

class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """
    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.store = threading.local()

    def set_state(self): pass
    def __call__(self, x, y):
        self.set_state()
        x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y in (TfmType.PIXEL, TfmType.CLASS)
                else self.transform_coord(x,y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x),y

    def transform(self, x, y=None):
        x = self.do_transform(x,False)
        return (x, self.do_transform(y,True)) if y is not None else x

#     @abstractmethod
#     def do_transform(self, x, is_y): raise NotImplementedError
    
class CoordTransform(Transform):
    """ A coordinate transform.  """

    @staticmethod
    def make_square(y, x):
        r,c,*_ = x.shape
        y1 = np.zeros((r, c))
        y = y.astype(np.int)
        y1[y[0]:y[2], y[1]:y[3]] = 1.
        return y1

    def map_y(self, y0, x):
        y = CoordTransform.make_square(y0, x)
        y_tr = self.do_transform(y, True)
        return to_bb(y_tr, y)

    def transform_coord(self, x, ys):
        yp = partition(ys, 4)
        y2 = [self.map_y(y,x) for y in yp]
        x = self.do_transform(x, False)
        return x, np.concatenate(y2)

class Scale(CoordTransform):
    """ A transformation that scales the min size to sz.

    Arguments:
        sz: int
            target size to scale minimum size.
        tfm_y: TfmType
            type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.sz_y = sz,sz_y

    def do_transform(self, x, is_y):
        if is_y: return scale_min(x, self.sz_y, cv2.INTER_NEAREST)
        else   : return scale_min(x, self.sz,   cv2.INTER_AREA   )
    
class NoCrop(CoordTransform):
    """  A transformation that resize to a square image without cropping.

    This transforms (optionally) resizes x,y at with the same parameters.
    Arguments:
        targ: int
            target size of the crop.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.sz_y = sz,sz_y

    def do_transform(self, x, is_y):
        if is_y: return no_crop(x, self.sz_y, cv2.INTER_NEAREST)
        else   : return no_crop(x, self.sz,   cv2.INTER_AREA   )

        
imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
stats = imagenet_stats

tfm_norm = Normalize(*stats, TfmType.NO)
tfm_denorm = Denormalize(*stats)

def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None,
              tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT):
    """
    Generate a standard set of transformations

    Arguments
    ---------
     normalizer :
         image normalizing function
     denorm :
         image denormalizing function
     sz :
         size, sz_y = sz if not specified.
     tfms :
         iterable collection of transformation functions
     max_zoom : float,
         maximum zoom
     pad : int,
         padding on top, left, right and bottom
     crop_type :
         crop type
     tfm_y :
         y axis specific transformations
     sz_y :
         y size, height
     pad_mode :
         cv2 padding style: repeat, reflect, etc.

    Returns
    -------
     type : ``Transforms``
         transformer for specified image operations.

    See Also
    --------
     Transforms: the transformer object returned by this function
    """
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    if sz_y is None: sz_y = sz
    scale = [RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y) if max_zoom is not None
             else Scale(sz, tfm_y, sz_y=sz_y)]
    if pad: scale.append(AddPadding(pad, mode=pad_mode))
    if crop_type!=CropType.GOOGLENET: tfms=scale+tfms
    return Transforms(sz, tfms, normalizer, denorm, crop_type,
                      tfm_y=tfm_y, sz_y=sz_y)

crop_fn_lu = {CropType.NO: NoCrop}

def compose(im, y, fns):
    """ apply a collection of transformation functions fns to images
    """
    for fn in fns:
        #pdb.set_trace()
        im, y =fn(im, y)
    return im if y is None else (im, y)

def scale_min(im, targ, interpolation=cv2.INTER_AREA):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    r,c,*_ = im.shape
    ratio = targ/min(r,c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz, interpolation=interpolation)

def scale_to(x, ratio, targ): 
    '''
    no clue, does not work.
    '''
    return max(math.floor(x*ratio), targ)

def crop(im, r, c, sz): 
    '''
    crop image into a square of size sz, 
    '''
    return im[r:r+sz, c:c+sz]

def no_crop(im, min_sz=None, interpolation=cv2.INTER_AREA):
    """ Returns a squared resized image """
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    return cv2.resize(im, (min_sz, min_sz), interpolation=interpolation)


# -------- end val_tfms stuff

def preproc_img(img):
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=0, crop_type=CropType.NO, tfm_y=None, sz_y=None)
    trans_img = val_tfm(img)
    return Variable(torch.FloatTensor(trans_img)).unsqueeze_(0)


def gen_anchors(anc_grids, anc_zooms, anc_ratios):
    anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
    k = len(anchor_scales)
    anc_offsets = [1/(o*2) for o in anc_grids]
    anc_x = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag) for ao,ag in zip(anc_offsets,anc_grids)])
    anc_y = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag) for ao,ag in zip(anc_offsets,anc_grids)])
    anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
    anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales]) for ag in anc_grids])
    grid_sizes_np = np.concatenate([np.array([ 1/ag for i in range(ag*ag) for o,p in anchor_scales]) for ag in anc_grids])
    anchors_np = np.concatenate([anc_ctrs, anc_sizes], axis=1)
    anchors = Variable(torch.FloatTensor(anchors_np))
    grid_sizes = Variable(torch.FloatTensor(grid_sizes_np)).unsqueeze(1)
    return anchors, grid_sizes

#gen ancs
anc_grids = [28,14,7,4,2]
anc_zooms = [2**(0/3),2**(1/3),2**(2/3)]
anc_ratios = [(1.,1.), (.5,1.), (1.,.5)]
anchors, grid_sizes = gen_anchors(anc_grids, anc_zooms, anc_ratios)

def hw2corners(ctr, hw): return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)

def actn_to_bb(actn, anchors, grid_sizes):
    actn_bbs = torch.tanh(actn)
    actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
    actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
    return hw2corners(actn_centers, actn_hw)

def to_np(v):
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if isinstance(v, torch.cuda.HalfTensor): v=v.float()
    return v.cpu().numpy()

def pred2dict(bb_np,score,cat_str):
    # convert to top left x,y bottom right x,y
    return {"x1": bb_np[1],
            "x2": bb_np[3],
            "y1": bb_np[0],
            "y2": bb_np[2],
            "score": score,
            "category": cat_str}

# non max suppression
def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def nms_preds(a_ic, p_cl, cl):
    nms_bb, nms_pr, nms_id = [],[],[]
    
    conf_scores = p_cl.sigmoid()[0].t().data
    boxes = a_ic.view(-1, 4)
    scores = conf_scores[cl]
    
    if len(scores)>0:
        ids, count = nms(boxes.data, scores, 0.4, 50)
        ids = ids[:count]

        nms_pr.append(scores[ids])
        nms_bb.append(boxes.data[ids])
        nms_id.append([cl]*count)

    else: nms_bb, nms_pr, nms_id = [[-1.,-1.,-1.,-1.,]],[[-1]],[[-1]]
    
    # return in order of a_ic, clas id, clas_pr
    return Variable(torch.FloatTensor(nms_bb[0])), Variable(torch.FloatTensor(nms_pr[0])), np.asarray(nms_id[0])

def get_predictions(img, nms=True):
    img_t = preproc_img(img)

    model  = load_model()

    #make predictions
    p_cl, p_bb = model(img_t)

    #convert bb and clas
    a_ic = actn_to_bb(p_bb[0], anchors, grid_sizes)
    clas_pr, clas_ids = p_cl[0].max(1)
    clas_pr = clas_pr.sigmoid()
    clas_ids = to_np(clas_ids)

    #non max suppression (optional)
    #cl = 1 hardcoded for now, bug with cl=0 to be fixed
    if nms: a_ic, clas_pr, clas_ids = nms_preds(a_ic, p_cl, 1)

    preds = []
    for i,a in enumerate(a_ic):
        cat_str = 'bg' if clas_ids[i]==len(id2cat) else id2cat[clas_ids[i]]
        score = to_np(clas_pr[i])[0].astype('float64')*100
        bb_np = to_np(a).astype('float64')
        preds.append(pred2dict(bb_np,score,cat_str))

    return {
        "bboxes": preds     
        }
    
def get_predictions1():
    main('dataset')

def load_model():

    dst = app.config['MODEL_FILE']
    # model = torch.load(dst)
    if os.path.isfile(dst): 
        model = torch.load(dst)
    else:
        dl_url = 'https://www.dropbox.com/s/e1gnf7oj7qdctlw/cocomodel_0502.pt?dl=1'
        with urlopen(dl_url) as u, NamedTemporaryFile(delete=False) as f:
            f.write(u.read())
            shutil.move(f.name, dst)

        model = torch.load(dst)
    return model
