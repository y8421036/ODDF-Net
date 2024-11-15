from .base_options import BaseOptions
import os

class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--load', type=str, default=False, help='whether restore or not') #False
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size') # seg 6M=2 3M=4   cla:6M=4 3M=4
        parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate for adam')
        parser.add_argument('--num_epochs', type=int, default=20, help='iterations for batch_size samples') #300  20
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--optimizer', type=str, default='Adam')
        return parser
