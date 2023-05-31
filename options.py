import argparse
import os


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_common_args(self):
        # train和test共有的参数
        self.parser.add_argument('--model_type', type=str, default='base_model', help='used in model_entry.py')
        self.parser.add_argument('--data_type', type=str, default='base_dataset', help='used in data_entry.py')
        self.parser.add_argument('--save_prefix', type=str, default='pref',
                                 help='some comment for model or test result dir')
        self.parser.add_argument('--load_model_path', type=str, default='checkpoints/base_model_pref/0.pth',
                                 help='model path for pretrain or test')
        self.parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
        self.parser.add_argument('--val_list', type=str, default='/data/dataset1/list/base/val.txt',
                                 help='val list in train, test list path in test')
        self.parser.add_argument('--gpus', nargs='+', type=int)
        self.parser.add_argument('--seed', type=int, default=1234)

    def parse_train_args(self):
        # train特有的参数
        self.parse_common_args()
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                 help='momentum for sgd, alpha parameter for adam')
        self.parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                                 help='beta parameters for adam')
        self.parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                                 metavar='W', help='weight decay')
        self.parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
        self.parser.add_argument('--train_list', type=str, default='/data/dataset1/list/base/train.txt')
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--epochs', type=int, default=100)

    def parse_test_args(self):
        # test特有的参数
        self.parse_common_args()
        self.parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
        self.parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')

    def get_train_args(self):
        # 返回train参数
        self.parse_train_args()
        args = self.parser.parse_args()
        return args

    def get_test_args(self):
        # 返回test参数
        self.parse_test_args()
        args = self.parser.parse_args()
        return args

    def get_train_model_dir(self, args):
        # 生成模型保存路径model_dir
        model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
        if not os.path.exists(model_dir):
            os.system('mkdir -p ' + model_dir)
        args.model_dir = model_dir

    def get_test_result_dir(self, args):
        # 生成结果保存的路径result_dir
        ext = os.path.basename(args.load_model_path).split('.')[-1]
        model_dir = args.load_model_path.replace(ext, '')
        val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(
            args.val_list.replace('.txt', ''))
        result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
        if not os.path.exists(result_dir):
            os.system('mkdir -p ' + result_dir)
        args.result_dir = result_dir

    def save_args(self, args, save_dir):
        # 保存参数至文件
        args_path = os.path.join(save_dir, 'args.txt')
        with open(args_path, 'w') as fd:
            fd.write(str(args).replace(', ', ',\n'))

    def prepare_train_args(self):
        args = self.get_train_args()
        self.get_train_model_dir(args)
        self.save_args(args, args.model_dir)
        return args

    def prepare_test_args(self):
        args = self.get_test_args()
        self.get_test_result_dir(args)
        self.save_args(args, args.result_dir)
        return args


if __name__ == '__main__':
    model_dir = os.path.join('checkpoints', 'cnn' + '_' + 'train')
    print(model_dir)
