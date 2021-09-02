import os
import shutil
import Butterfly_identification.preprocessbutterfly as preproc
if __name__ == "__main__" :
    df_train, df_val, df_test = preproc.get_data(data=["train", "val", "test"])
    print(">>>>>>>>>>>>>>>>>>>>>>>>        Data loaded        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df_train, df_val, df_test = preproc.feature_engineering(df_train, df_val, df_test)
    print(">>>>>>>>>>>>>>>>>>>>>>>>      Preprocess done      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    parts =  os.getcwd().split("/")
    ABS_PATH = "/" + parts[1] + "/" + parts[2] + "/"
    os.chdir(ABS_PATH + "code/Em3line/Butterfly_identification/raw_data/")
    data_dir = ABS_PATH + "code/Em3line/Butterfly_identification/raw_data/"
    folder_name_docker = 'Docker/'
    os.mkdir(f'{data_dir + folder_name_docker}')
    print(f">>>>>>>>>>>>>>>>>>>>>>>>    {folder_name_docker} created   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    folder_name = 'Photos/'
    os.mkdir(f'{data_dir + folder_name_docker + folder_name}')
    print(f">>>>>>>>>>>>>>>>>>>>>>>>    {folder_name} created   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    path = data_dir + folder_name_docker + folder_name
    for folder in list(df_train["species"].unique()):
        os.mkdir(f"{path + folder}")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>    Species folders created   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    images_dir = ABS_PATH + "code/Em3line/Butterfly_identification/raw_data/IMG_labels/"
    for specie in list(df_train["species"].unique()):
        path_to_image = df_train.loc[df_train['species'] == specie, 'image_path'].sample(3)
        for image in path_to_image :
            old_path = images_dir + image
            new_path = path + image
            shutil.copyfile(old_path,new_path)
    print(f">>>>>>>>>>>>>>>>>>>>>>>>   3 images by species copied   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>   All done. Go check !   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
