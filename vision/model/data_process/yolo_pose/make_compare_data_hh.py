import os
import shutil
import random

def createFolder(dataset_root, target_path):
    Ir_or_RGB = ["IR", "RGB"]
    paths = []
    for i in Ir_or_RGB:
        p = os.path.join(target_path, i)
        os.mkdir(p)
        paths.append(p)
    # print(paths)

    user_id = os.listdir(dataset_root)
    # print(user_id)
    train_val = ['train', 'val']

    path_train = []
    path_val = []
    for path in paths:
        p0 = os.path.join(path, train_val[0])
        os.mkdir(p0)
        path_train.append(p0)

        p1 = os.path.join(path, train_val[1])
        os.mkdir(p1)
        path_val.append(p1)
    # print(path_train)

    folder_list_train = []
    folder_list_val = []
    for path in path_train:
        for id in user_id:
            p = os.path.join(path, id)
            folder_list_train.append(p)
            os.mkdir(p)

    for path in path_val:
        for id in user_id:
            p = os.path.join(path, id)
            folder_list_val.append(p)
            os.mkdir(p)
    return folder_list_train, folder_list_val


if __name__ == '__main__':
    dataset_root = r'/home/hgh/source/datas/keypoints/PALM'
    target_path = r'/home/hgh/source/datas/keypoints/PALM_RGB_IR'
    folder_list_t, folder_list_v = createFolder(dataset_root, target_path)
    for i in range(len(folder_list_t)):
        print(folder_list_t[i])
        print(f"  {folder_list_v[i]}")

        id = folder_list_t[i].split(os.sep)[-1]
        model = folder_list_t[i].split(os.sep)[-3]

        root_imgs_path = os.path.join(dataset_root,id,model)
        img_list =os.listdir(root_imgs_path)

        for img in img_list:
            img_path = os.path.join(root_imgs_path,img)
            if random.randint(1,10) != 1:
                shutil.copy(img_path, folder_list_t[i])
                print(f"{img} save {folder_list_t[i]}")
            else:
                shutil.copy(img_path, folder_list_v[i])
                print(f"{img} save {folder_list_v[i]}")

