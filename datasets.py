# import y4m
import numpy as np
import glob
import imageio
# import codecs
import matplotlib.pyplot as plt
YCbCr2RGB = np.array( [[1.164, 0., 1.793],
                       [1.164, -0.213, -0.533],
                       [1.164, 2.112, 0.0]] )

def frameToNDArray(frame):
    data = np.frombuffer(frame.buffer, dtype='uint8').astype('float32')
    H = frame.headers['H']
    W = frame.headers['W']
    data = np.reshape(data, [6, H//2, W//2])
    Y = data[0:4,:,:].reshape([H, W])
    Cb = data[4,:,:].clip(16, 240)-128
    Cb = Cb.repeat(2, axis=0).repeat(2, axis=1)
    Cr = data[5,:,:].clip(16, 240)-128
    Cr = Cr.repeat(2, axis=0).repeat(2, axis=1)
    #import pdb; pdb.set_trace()
    frame = np.stack([Y, Cb, Cr], axis=-1)
    frameRGB = np.dot(frame, YCbCr2RGB.T)
    return frameRGB.clip(0,255)/255

class Y4MDecoder:
    def __init__(self):
        self.frames = []

    def incorporateFrame(self,frame):
        self.frames.append(frameToNDArray(frame))

    def getVideo(self):
        return np.stack(self.frames)


def loadVideoFromPath(path):
    decoder = Y4MDecoder()
    parser = y4m.Reader(decoder.incorporateFrame, verbose=False)
    with open(path, 'rb') as f:
        data = f.read()
        parser.decode(data.decode('latin-1', errors='replace').encode('latin-1'))

    return decoder.getVideo()

# def generateDataset(gt_path,file_path):
#     gt = sorted(glob.glob(gt_path+"/scan_*"))
#     files = sorted(glob.glob(filepath+"/scan_*"))
#     sample = imageio.imread(files[0])
#     ip_frames = np.empty((6,sample.shape[0],sample.shape[1],1),dtype = np.float32)
#     gt_frames = np.empty((5,sample.shape[0],sample.shape[1],1),dtype = np.float32)
#     for i in range(6):
#         ip_frames[i,:,:,0] = imageio.imread(files[i])
#     for i in range(5):
#         gt_frames[i,:,:,0] = imageio.imread(gt[i])
       
#     mean_img = np
    
def generateDataSet(file_path):
    files = np.load(file_path)
    file = []
    for i in range(0,files.shape[0]-1,5):   
        file.append(files[i,:,:])
        if len(file)==20:
            break
                    
    frames = np.empty((len(file),file[0].shape[0],file[0].shape[1],3),files.dtype)
    for i in range(len(file)):
        for j in range(3): 
            frames[i,:,:,j] = file[i]
    
    mean_img = np.mean(frames, axis=0)
    restore = lambda img: img + mean_img

    train_befores = frames[:-2,:,:,:]
    train_middles = frames[1:-1,:,:,:]
    train_afters = frames[2:,:,:,:]

    val_befores = frames[:-1,:,:,:]
    val_middles = frames[1::2,:,:,:]
    val_afters = frames[1:,:,:,:]

    train_doublets = np.concatenate((train_befores, train_afters), axis=3)
    train_triplets = np.concatenate((train_befores, train_middles, train_afters), axis=3)

    val_doublets = np.concatenate((val_befores, val_afters), axis=3)
    val_targets = val_middles

    data = {"train_doublets": train_doublets, "train_triplets": train_triplets,
            "val_doublets": val_doublets, "val_targets": val_targets,
            "mean_img": mean_img, "restore": restore}
    print(frames.shape)
    print(train_doublets.shape)
    print(train_triplets.shape)
    plt.imshow(train_doublets[0,:,:,0])
    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = generateDataSet("/home/ajay/work/Sparse CT/cancerimagingarchivedata/Walnut1/tubeV1/")
    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow( data["restore"]( 0.5*( data["train_triplets"][0,:,:,0:3] + data["train_triplets"][0,:,:,6:9] ) ) )
    axarr[1,0].imshow( data["restore"]( data["train_triplets"][0,:,:,3:6] ) )
    axarr[0,1].imshow( data["restore"]( 0.5*( data["val_doublets"][0,:,:,0:3] + data["val_doublets"][0,:,:,6:9] ) ) )
    axarr[1,1].imshow( data["restore"]( data["val_targets"][0,:,:,:] ) )

    imgplt = plt.imshow(video[5,:,:,:])
    plt.show()
