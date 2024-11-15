import argparse

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing.
    #It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--dataroot', default='/data2/datasets/OCTA-500/6M', help='path to data') # 3M  6M
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids')
        parser.add_argument('--train_ids',type=list,default=[0,240],help='train id number')  # 3M:[0,140]  6M:[0,240]
        parser.add_argument('--val_ids',type=list,default=[240,250],help='val id number')  #3M:[140,150]  6M:[240,250]
        parser.add_argument('--test_ids',type=list,default=[250,300],help='test id number')  #3M:[150,200]  6M:[250,300]
        parser.add_argument('--modality_filename', type=list, default=['OCTA_npy','OCT_OD_npy'], help='dataroot/dataset filename') #  'OCTA_npy','OCT_OD_npy'
        parser.add_argument('--label_filename', type=list, default=['GT_CAVF'], help='dataroot/dataset filename')
        parser.add_argument('--data_size', type=list, default=[128,400,400], help='input data size separated with comma')  #3M:[128,304,304]  6M:[128,400,400]
        #基于patch的OD-Net难学习loss不降，需要重新找学习率
        #3M:  IPNV2：[128,304,304]  IPN：patch[128,76,76]        #6M:  IPNV2：[128,400,400]  IPN：patch[128,100,100]
        parser.add_argument('--train_size', type=list, default=[128,400,400], help='input data size separated with comma')  #3M:[128,304,304]  6M:[128,400,400]
        parser.add_argument('--in_channels', type=int, default=2, help='input channels')
        parser.add_argument('--n_classes', type=int, default=5, help='class number')
        parser.add_argument('--saveroot', default='logs', help='path to save models and results')
        #-------------------------------------------------------------------------------------------------------
        parser.add_argument('--seg_pth',  type=str, 
                            default='logs/3M/seg/0.885157/290.pth', 
                            help='pth path of the trainded seg model for training cla model') #False
        parser.add_argument('--disease_npy',  type=str, 
                            default='/data2/datasets/OCTA-500/6M/6M-Text labels.npy',
                            help='numpy path of the disease label for training and testing cla model') #False
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        print('')

    def parse(self):
        """Parse our options"""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt



