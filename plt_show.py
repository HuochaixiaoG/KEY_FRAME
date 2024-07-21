import time
import matplotlib.pyplot as plt
import argparse
import numpy as np

# loss单曲线
def train_loss_plt(args):
    loss = list(map(float, args.train_loss))
    plt.plot(range(args.epoch), loss, color='orange', label='train_loss')
    plt.title('train loss Curve', fontsize=args.fontsize)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
# acc单曲线
def train_acc_plt(args):
    acc = list(map(float, args.train_acc))
    plt.plot(range(args.epoch), acc, color='red', label='train_acc')
    plt.title('train acc Curve', fontsize=args.fontsize)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
def valid_loss_plt(args):
    loss = list(map(float, args.valid_loss))
    plt.plot(range(args.epoch), loss, color='green', label='valid_loss')
    plt.title('valid loss Curve', fontsize=args.fontsize)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
# acc单曲线
def valid_acc_plt(args):
    acc = list(map(float, args.valid_acc))
    plt.plot(range(args.epoch), acc, color='blue', label='valid_acc')
    plt.title('valid acc Curve', fontsize=args.fontsize)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
# loss双图
def loss_plt(args):
    train_loss = list(map(float, args.train_loss))
    valid_loss = list(map(float, args.valid_loss))
    plt.plot(range(args.epoch), train_loss, color='orange', label='train_loss')
    plt.plot(range(args.epoch), valid_loss, color='green', label='valid_loss')
    plt.title('Loss Curve', fontsize=args.fontsize)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
# acc双图
def acc_plt(args):
    train_acc = list(map(float, args.train_acc))
    valid_acc = list(map(float, args.valid_acc))
    plt.plot(range(args.epoch), train_acc, color='red', label='train_acc')
    plt.plot(range(args.epoch), valid_acc, color='blue', label='valid_acc')
    plt.title('AUC', fontsize=args.fontsize)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
# acc-loss
def train_acc_loss(args):
    acc = list(map(float, args.train_acc))
    loss = list(map(float, args.train_loss))
    plt.plot(range(args.epoch), acc, color='red', label='train_acc')
    plt.plot(range(args.epoch), loss, color='orange', label='train_loss')
    plt.title('train acc-loss Curve', fontsize=args.fontsize)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
def valid_acc_loss(args):
    acc = list(map(float, args.valid_acc))
    loss = list(map(float, args.valid_loss))
    plt.plot(range(args.epoch), acc, color='blue', label='valid_acc')
    plt.plot(range(args.epoch), loss, color='green', label='valid_loss')
    plt.title('valid acc-loss Curve', fontsize=args.fontsize)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def four(args):
    plt.plot(np.arange(len(args.train_loss)), list(map(float, args.train_loss)), color="orange", label="train loss")
    plt.plot(np.arange(len(args.train_acc)), list(map(float, args.train_acc)), color="red", label="train acc")
    plt.plot(np.arange(len(args.valid_loss)), list(map(float, args.valid_loss)), color="green", label="valid loss")
    plt.plot(np.arange(len(args.valid_acc)), list(map(float, args.valid_acc)), color="blue", label="valid acc")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    plt.ylabel("{}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    plt.title('Accuracy & loss')
    plt.savefig("./loss_and_acc_{}.jpg".format(time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())))
    plt.show()
def main(args):
    train_loss_plt(args)   # train-loss
    train_acc_plt(args)    # train-acc
    valid_loss_plt(args)   # valid-loss
    valid_acc_plt(args)    # valid-acc
    loss_plt(args)         # loss双曲线
    acc_plt(args)          # acc双曲线
    train_acc_loss(args)   # train双曲线
    valid_acc_loss(args)   # valid双曲线
    four(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    train_acc = ['0.627', '0.681', '0.708', '0.715', '0.719', '0.725', '0.725', '0.734', '0.729', '0.732', '0.738',
                 '0.743', '0.736', '0.744', '0.745', '0.746', '0.751', '0.756', '0.754', '0.755', '0.759', '0.761',
                 '0.762', '0.766', '0.774', '0.767', '0.771', '0.773', '0.775', '0.778', '0.780', '0.780', '0.780',
                 '0.782', '0.783', '0.789', '0.790', '0.794', '0.793', '0.796', '0.798', '0.803', '0.802', '0.805',
                 '0.808', '0.809', '0.815', '0.811', '0.821', '0.817', '0.821', '0.824', '0.828', '0.830', '0.830',
                 '0.836', '0.840', '0.841', '0.845', '0.846', '0.849', '0.854', '0.852', '0.855', '0.860', '0.862',
                 '0.866', '0.865', '0.868', '0.872', '0.873', '0.875', '0.881', '0.882', '0.884', '0.886', '0.888',
                 '0.892', '0.893', '0.896', '0.897', '0.899', '0.898', '0.903', '0.906', '0.905', '0.908', '0.909',
                 '0.910', '0.910', '0.911', '0.912', '0.914', '0.913', '0.915', '0.915', '0.918', '0.916', '0.917',
                 '0.918']
    train_loss = ['0.650', '0.603', '0.575', '0.566', '0.557', '0.552', '0.548', '0.544', '0.541', '0.537', '0.531',
                  '0.525', '0.528', '0.519', '0.519', '0.515', '0.510', '0.505', '0.500', '0.502', '0.500', '0.492',
                  '0.492', '0.488', '0.484', '0.482', '0.479', '0.474', '0.472', '0.467', '0.467', '0.462', '0.464',
                  '0.456', '0.457', '0.448', '0.446', '0.444', '0.441', '0.436', '0.434', '0.429', '0.424', '0.421',
                  '0.416', '0.412', '0.402', '0.406', '0.396', '0.395', '0.390', '0.385', '0.382', '0.374', '0.372',
                  '0.364', '0.358', '0.352', '0.350', '0.348', '0.341', '0.335', '0.333', '0.328', '0.320', '0.317',
                  '0.312', '0.311', '0.304', '0.300', '0.293', '0.291', '0.287', '0.279', '0.275', '0.273', '0.269',
                  '0.263', '0.259', '0.256', '0.253', '0.249', '0.247', '0.241', '0.240', '0.239', '0.236', '0.233',
                  '0.229', '0.230', '0.227', '0.225', '0.224', '0.223', '0.222', '0.220', '0.221', '0.219', '0.219',
                  '0.219']
    valid_acc = ['0.668', '0.722', '0.719', '0.704', '0.719', '0.718', '0.729', '0.731', '0.727', '0.734', '0.744',
                 '0.713', '0.734', '0.682', '0.733', '0.731', '0.744', '0.747', '0.730', '0.718', '0.753', '0.734',
                 '0.704', '0.745', '0.749', '0.742', '0.748', '0.694', '0.750', '0.761', '0.735', '0.759', '0.757',
                 '0.758', '0.759', '0.749', '0.751', '0.764', '0.760', '0.761', '0.761', '0.759', '0.748', '0.750',
                 '0.756', '0.764', '0.765', '0.759', '0.761', '0.759', '0.766', '0.759', '0.752', '0.749', '0.767',
                 '0.760', '0.760', '0.759', '0.761', '0.767', '0.772', '0.760', '0.771', '0.778', '0.761', '0.771',
                 '0.771', '0.761', '0.769', '0.751', '0.767', '0.772', '0.768', '0.764', '0.763', '0.763', '0.765',
                 '0.767', '0.765', '0.771', '0.772', '0.769', '0.769', '0.771', '0.769', '0.768', '0.770', '0.770',
                 '0.763', '0.767', '0.768', '0.765', '0.771', '0.768', '0.765', '0.770', '0.766', '0.770', '0.768',
                 '0.767']
    valid_loss = ['0.633', '0.569', '0.567', '0.579', '0.549', '0.568', '0.545', '0.547', '0.553', '0.536', '0.532',
                  '0.554', '0.532', '0.603', '0.542', '0.549', '0.530', '0.530', '0.545', '0.558', '0.521', '0.531',
                  '0.560', '0.527', '0.513', '0.530', '0.522', '0.582', '0.513', '0.504', '0.523', '0.510', '0.513',
                  '0.508', '0.515', '0.517', '0.515', '0.503', '0.514', '0.507', '0.515', '0.515', '0.518', '0.551',
                  '0.524', '0.509', '0.530', '0.503', '0.525', '0.527', '0.508', '0.518', '0.528', '0.532', '0.515',
                  '0.531', '0.535', '0.547', '0.555', '0.532', '0.547', '0.519', '0.521', '0.555', '0.537', '0.547',
                  '0.570', '0.552', '0.591', '0.567', '0.586', '0.584', '0.590', '0.591', '0.577', '0.625', '0.580',
                  '0.584', '0.585', '0.618', '0.608', '0.613', '0.632', '0.647', '0.658', '0.623', '0.634', '0.652',
                  '0.670', '0.646', '0.654', '0.647', '0.650', '0.660', '0.644', '0.659', '0.651', '0.659', '0.660',
                  '0.666']

    parser.add_argument('--train_acc', type=list, default=train_acc)  # 通过调用add_argument()方法给ArgumentParser添加程序参数信息
    parser.add_argument('--train_loss', type=list, default=train_loss)  # default是不指定参数时的默认值
    parser.add_argument('--valid_acc', type=list, default=valid_acc)  # type是命令行参数应该被转换成的类型
    parser.add_argument('--valid_loss', type=list, default=valid_loss)
    parser.add_argument('--epoch', type=int, default=len(train_acc))
    parser.add_argument('--fontsize', type=int, default=16)
    opt = parser.parse_args()  # 使用 parse_args() 解析添加的参数
    main(opt)
