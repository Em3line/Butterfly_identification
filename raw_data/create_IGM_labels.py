import os
import shutil
import Butterfly_identification.preprocessbutterflygcp as preproc

if __name__=="__main__" :
    parts =  os.getcwd().split("/")
    ABS_PATH ="/"+parts[1]+"/"+parts[2]+"/"

    df_train,df_val,df_test = preproc.get_data_local(data=["train","val","test"])
    print(">>>>>>>>>>>>>>>>>>>>>>>>        Data loaded        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


    df_train,df_val,df_test = preproc.feature_engineering(df_train,df_val,df_test)
    print(">>>>>>>>>>>>>>>>>>>>>>>>      Preprocess done      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    data_dir = ABS_PATH + "code/Em3line/Butterfly_identification/raw_data/"
    images_dir = ABS_PATH + "code/Em3line/Butterfly_identification/raw_data/IMG/"

    folder_name = "IGM_labels"

    #os.chdir(ABS_PATH + "/code/Em3line/Butterfly_identification/raw_data/")
    os.mkdir (f'{data_dir+folder_name}')
    #os.chdir(ABS_PATH + "/code/Em3line/Butterfly_identification/raw_data/IGM_labels")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>    {folder_name} created   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    IGM_dir = ABS_PATH + f"code/Em3line/Butterfly_identification/raw_data/{folder_name}/"

    os.mkdir (IGM_dir + "Train")
    os.mkdir (IGM_dir + "Val")
    os.mkdir (IGM_dir + "Test")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>   Train,Val,Test created   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


    Train_path = ABS_PATH + f"code/Em3line/Butterfly_identification/raw_data/{folder_name}/Train/"
    Val_path = ABS_PATH + f"code/Em3line/Butterfly_identification/raw_data/{folder_name}/Val/"
    Test_path = ABS_PATH + f"code/Em3line/Butterfly_identification/raw_data/{folder_name}/Test/"


    for folder in list(df_train["species"].unique()):
        os.mkdir(f"{Train_path+folder}")

    print(f">>>>>>>>>>>>>>>>>>>>>>>>   Train folders created   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    for folder in list(df_val["species"].unique()):
        os.mkdir(f"{Val_path+folder}")

    print(f">>>>>>>>>>>>>>>>>>>>>>>>    Val folders created    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    for folder in list(df_test["species"].unique()):
        os.mkdir(f"{Test_path+folder}")

    print(f">>>>>>>>>>>>>>>>>>>>>>>>   Test folders created   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    for i in range(len(df_train["species"])):
        old_path=images_dir+df_train["image_name"][i]
        new_path=Train_path+df_train["species"][i]+"/"+df_train["image_name"][i]
        shutil.copyfile(old_path,new_path)

    print(f">>>>>>>>>>>>>>>>>>>>>>>>   Train images copied   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    for i in range(len(df_val["species"])):
        old_path=images_dir+df_val["image_name"][i]
        new_path=Val_path+df_val["species"][i]+"/"+df_val["image_name"][i]
        shutil.copyfile(old_path,new_path)

    print(f">>>>>>>>>>>>>>>>>>>>>>>>    Val images copied    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    for i in range(len(df_test["species"])):
        old_path=images_dir+df_test["image_name"][i]
        new_path=Test_path+df_test["species"][i]+"/"+df_test["image_name"][i]
        shutil.copyfile(old_path,new_path)

    print(f">>>>>>>>>>>>>>>>>>>>>>>>   Test images copied   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")