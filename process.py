import cv2


folder = './data/data/data'
target_fold = './data/train/imgs'

for i in range(1, 101):
    path = '%s/image_%d.jpg' % (folder, i)

    img = cv2.imread(path)
    print(img)

    tar_path = '%s/%d.png'% (target_fold, i+40)
    print(tar_path)
    cv2.imwrite(tar_path, img)