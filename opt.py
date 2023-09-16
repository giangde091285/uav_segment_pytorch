import argparse

def get_args():

    parser = argparse.ArgumentParser()
    
    # dataset
   
    parser.add_argument('--if_spilt', type=bool, default=False)
    parser.add_argument('--if_crop', type=bool, default=False)
    parser.add_argument('--tar_size', type=int, default=256)
    parser.add_argument('--tar_num', type=int, default=4)
    parser.add_argument('--train_scale', type=float, default=0.8)
    parser.add_argument('--val_scale', type=float, default=0.1)

    # train
    parser.add_argument('--model', type=str, default='U_net')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='Cosine')
    parser.add_argument('--loss', type=str, default='DiceLoss')
    parser.add_argument('--if_pre_ckpt', type=bool, default=True)
    parser.add_argument('--pre_ckpt_path', type=str, default='./ckpt_history/new4_checkpoint_epoch_20.0.pth')

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--classes', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=3)

    # predict
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/checkpoint_epoch_18.0.pth')
    parser.add_argument('--ifsave', type=str, default=False)
    parser.add_argument('--output_dir', type=str, default='./output/')


    return parser.parse_args()
