import numpy as np
import cv2

txtp = "splitimage.txt"
savef='splited/'
def name(full_path):
    return full_path.split('/')[-1].split('.')[0]
with open(txtp) as f:
    iml = list(f.readlines())
for imp in iml:
    imp=imp.strip()
    img = cv2.imread(imp.strip())
    size = 100
    v_size = img.shape[0] // size * size
    h_size = img.shape[1] // size * size
    img = img[:v_size, :h_size]

    v_split = img.shape[0] // size
    h_split = img.shape[1] // size
    out_img = []
    [out_img.extend(np.hsplit(h_img, h_split)) for h_img in np.vsplit(img, v_split)]
    t=savef+str(0) + name(imp)+'.jpg'
    for i in range(len(out_img)):
        from PIL import Image
        #Image.fromarray(out_img[i]).save(savef+str(i) + name(imp)+'.jpg')
        cv2.imwrite(savef+str(i) + name(imp)+'.jpg', out_img[i])
