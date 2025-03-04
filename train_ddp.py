import argparse
import datetime
import os
import os.path as osp
import time

import torch
import torch.utils.data
from torch.cuda import amp

from datasets import build_test_loader, build_train_loader_ddp
from defaults import get_default_cfg
from engines.engine import evaluate_performance, train_one_epoch
# from models.coam import COAT
# from models.coat import COAT
# from models.w1t2t3 import COAT
# from models.w1w2t3 import COAT
# from models.t1t2w1 import COAT
# from models.t1w2w1 import COAT
# from models.coam_att_two import COAT
import torch.distributed as dist
from models.coam_att_two import COAT # two_head
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed, write_text

from loss.softmax import SoftmaxLoss


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.NVIDIA_DEVICE)
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group('nccl',world_size=world_size,rank=rank)
    device = torch.device(cfg.DEVICE,rank)

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    mkdir(osp.join(output_dir, 'checkpoints'))

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    write_text(sentence="Creating model", fpath=os.path.join(output_dir, 'os.txt'))
    model = COAT(cfg)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],find_unused_parameters=True)

    write_text(sentence="Loading data", fpath=os.path.join(output_dir, 'os.txt'))
    train_loader,sampler = build_train_loader_ddp(cfg)
    gallery_loader, query_loader = build_test_loader(cfg)

    softmax_criterion_s2 = None
    softmax_criterion_s3 = None
    if cfg.MODEL.LOSS.USE_SOFTMAX:
        softmax_criterion_s2 = SoftmaxLoss(cfg)
        softmax_criterion_s3 = SoftmaxLoss(cfg)
        softmax_criterion_s2.to(device)
        softmax_criterion_s3.to(device)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        eval_path = osp.join(output_dir, 'pure_eval')
        mkdir(eval_path)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
            outsys_dir=eval_path,
            gallery_size=cfg.EVAL_GALLERY_SIZE,
        )
        exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.MODEL.LOSS.USE_SOFTMAX:
        params_softmax_s2 = [p for p in softmax_criterion_s2.parameters() if p.requires_grad]
        params_softmax_s3 = [p for p in softmax_criterion_s3.parameters() if p.requires_grad]
        params.extend(params_softmax_s2)
        params.extend(params_softmax_s3)

    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=cfg.SOLVER.GAMMA
    )
    scaler = amp.GradScaler()

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    write_text(sentence="Creating output folder", fpath=os.path.join(output_dir, 'os.txt'))
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    write_text(sentence="Full config is saved to {}".format(path), fpath=os.path.join(output_dir, 'os.txt'))
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        write_text("TensorBoard files are saved to {}".format(tf_log_path), fpath=osp.join(output_dir, 'os.txt'))

    print("Start training...")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        sampler.set_epoch(epoch)
        train_one_epoch(cfg=cfg,
                        model=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        device=device,
                        epoch=epoch,
                        scaler=scaler,
                        tfboard=tfboard,
                        softmax_criterion_s2=softmax_criterion_s2,
                        softmax_criterion_s3=softmax_criterion_s3,
                        outsys_dir=output_dir
                        )
        lr_scheduler.step()

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            n_iter = (epoch + 1) * len(train_loader)
            ret = evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=False,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
                gallery_size=cfg.EVAL_GALLERY_SIZE,
                outsys_dir=output_dir,
            )

            if epoch == cfg.SOLVER.MAX_EPOCHS - 1:
                write_text(sentence='using GT boxes', fpath=osp.join(output_dir, 'os.txt'))
                ret_gt = evaluate_performance(
                    model,
                    gallery_loader,
                    query_loader,
                    device,
                    use_gt=True,
                    use_cache=cfg.EVAL_USE_CACHE,
                    use_cbgm=cfg.EVAL_USE_CBGM,
                    gallery_size=cfg.EVAL_GALLERY_SIZE,
                    outsys_dir=output_dir,
                )
                if tfboard:
                    tfboard.add_scalar("test_gt/mAP", ret_gt['mAP'], n_iter)
                    tfboard.add_scalar("test_gt/r1", ret_gt['accs'][0], n_iter)
                    tfboard.add_scalar("test_gt/r10", ret_gt['accs'][2], n_iter)

            write_text(sentence=' ', fpath=osp.join(output_dir, 'os.txt'))

            if tfboard:
                tfboard.add_scalar("test/mAP", ret['mAP'], n_iter)
                tfboard.add_scalar("test/r1", ret['accs'][0], n_iter)
                tfboard.add_scalar("test/r10", ret['accs'][2], n_iter)

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            ckpt_dir = osp.join(output_dir, 'checkpoints')
            if rank == 0:
                save_on_master(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    osp.join(ckpt_dir, f"epoch_{epoch}.pth"),
                )
    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    write_text("Total training time {}".format(total_time_str), fpath=osp.join(output_dir, 'os.txt'))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    parser.add_argument('--local_rank',type=int)
    args = parser.parse_args()
    main(args)
