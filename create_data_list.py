from models.utils import create_data_lists_2

if __name__ == '__main__':
    # create_data_lists(voc07_path='VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007',
    #                   voc12_path='VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007',
    #                   output_folder='./')

    create_data_lists_2(voc_path='datasets', output_folder='./datasets/split')