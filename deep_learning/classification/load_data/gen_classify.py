import os
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator

def get_generator(hparams,dataset='train'):
    """[summary]

    Args:
        hparams ([type]): [description]
        dataset (str, optional): [description]. Defaults to 'train'.

    Returns:
        [type]: [description]
    """

    dataset_partition_folder = 'data/'    
    target_size = (hparams.img_height,hparams.img_weight)

    if(dataset=='train'):
        batch_size = hparams.batch_size
        datagen = ImageDataGenerator()
        dataset_csv = os.path.join(dataset_partition_folder,'train_partition.csv')
        img_directory = 'data/pngs/train'
        shuffle = True
    elif(dataset=='val'):
        batch_size = 1
        datagen = ImageDataGenerator()
        dataset_csv = os.path.join(dataset_partition_folder,'val_partition.csv')
        img_directory = 'data/pngs/train'
        shuffle = False

    df = pd.read_csv(dataset_csv)
    # update df for flow_from_dataframe method
    df['ptx_binary'] = df['ptx_binary'].astype('str')
    df['ImageId'] = df['ImageId'] + '.png' 

    img_directory = 'data/pngs/train'
    generator = datagen.flow_from_dataframe(dataframe=df, 
                                            directory=img_directory,
                                            x_col='ImageId',
                                            y_col='ptx_binary',
                                            class_mode='categorical',# categorical or binary
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            seed=21,
                                            shuffle=shuffle
                                            )

    return generator
