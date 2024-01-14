"""
Author: Eckert ZHANG
Date: 2021-07-12 17:54:18
LastEditTime: 2022-03-31 22:48:03
LastEditors: Eckert ZHANG
Description: 
"""
import os.path
import torch, pdb
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm
import warnings
import imageio


class Trainer:
    def __init__(self,
                 net,
                 train_dataset,
                 test_dataset,
                 args,
                 conf,
                 device=None):
        self.args = args
        self.conf = conf
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        const = 1 if args.only_test else 16
        self.test_data_batch = min(args.batch_size, const)

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,  #8
            pin_memory=False,
        )
        if args.only_test or args.only_video:
            self.test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.test_data_batch,
                shuffle=False,
                num_workers=4,  #4
                pin_memory=False,
            )
        else:
            self.test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.test_data_batch,
                shuffle=True,
                num_workers=4,  #4
                pin_memory=False,
            )

        self.num_total_batches = len(self.train_dataset)
        self.exp_name = args.name
        self.save_interval = conf.get_int("save_interval")
        self.print_interval = conf.get_int("print_interval")
        self.vis_interval = conf.get_int("vis_interval")
        self.eval_interval = conf.get_int("eval_interval")
        self.num_epoch_repeats = conf.get_int("num_epoch_repeats", 1)
        self.num_epochs = args.epochs
        self.accu_grad = conf.get_int("accu_grad", 1)
        self.fixed_test = args.fixed_test

        ### Set associated paths
        self.summary_path = os.path.join(args.resultdir, args.name,
                                         args.logs_path)
        os.makedirs(self.summary_path, exist_ok=True)
        self.writer = SummaryWriter(self.summary_path)
        if self.args.warp_pretrain:
            self.visual_path = os.path.join(self.args.resultdir,
                                            self.args.name, 'vis_pretrain')
            os.makedirs(self.visual_path, exist_ok=True)
        else:
            self.visual_path = os.path.join(self.args.resultdir,
                                            self.args.name,
                                            self.args.visual_path)
        if self.args.only_test:
            self.visual_path_com = os.path.join(self.visual_path, 'components')
            os.makedirs(self.visual_path_com, exist_ok=True)

        if self.args.warp_pretrain:
            self.optim = None
            self.lr_scheduler = None
            self.stage_net_G = 'warp'
        else:
            ### Set Optimizer & Scheduler (Currently only Adam supported)
            self.optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            if args.gamma != 1.0:
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optim, gamma=args.gamma)
            else:
                self.lr_scheduler = None

        ### Load weights
        # self.managed_weight_saving = hasattr(net, "load_weights")
        # if self.managed_weight_saving:
        #     net.load_weights(self.args)
        self.iter_state_path = "%s/%s/%s/_iter" % (
            self.args.resultdir,
            self.args.name,
            self.args.checkpoints_path,
        )
        self.optim_state_path = "%s/%s/%s/_optim" % (
            self.args.resultdir,
            self.args.name,
            self.args.checkpoints_path,
        )
        self.lrsched_state_path = "%s/%s/%s/_lrsched" % (
            self.args.resultdir,
            self.args.name,
            self.args.checkpoints_path,
        )
        self.default_net_state_path = "%s/%s/%s/net" % (
            self.args.resultdir,
            self.args.name,
            self.args.checkpoints_path,
        )
        self.renderer_state_path = "%s/%s/%s/_renderer" % (
            self.args.resultdir,
            self.args.name,
            self.args.checkpoints_path,
        )
        self.start_iter_id = 0
        if args.resume:
            if not self.args.warp_pretrain:
                net.load_weights(self.args,
                                 opt_init=self.args.resume_init,
                                 ckpt_path_init=self.args.resume_ckpt_init,
                                 strict=False,
                                 device=device)
            if os.path.exists(self.optim_state_path):
                try:
                    self.optim.load_state_dict(
                        torch.load(self.optim_state_path, map_location=device))
                except:
                    warnings.warn("Failed to load optimizer state at " +
                                  self.optim_state_path)
            if self.lr_scheduler is not None and os.path.exists(
                    self.lrsched_state_path):
                self.lr_scheduler.load_state_dict(
                    torch.load(self.lrsched_state_path, map_location=device))
            if os.path.exists(self.iter_state_path):
                self.start_iter_id = torch.load(self.iter_state_path,
                                                map_location=device)["iter"]
            # if not self.managed_weight_saving and os.path.exists(
            #         self.default_net_state_path):
            #     net.load_state_dict(
            #         torch.load(self.default_net_state_path,
            #                    map_location=device))

    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self):
        """
        Ran at each save step for saving extra state
        """
        pass

    def warp_pretrain_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def train_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, data, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None

    def vis_video_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None

    def warp_vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None

    def start(self):
        def fmt_loss_str(losses):
            return "loss " + (" ".join(k + ":" + str(losses[k])
                                       for k in losses))

        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x

        test_data_iter = data_loop(self.test_data_loader)
        step_id = self.start_iter_id

        if self.args.only_test:
            self.net.eval()
            for epoch in range(1):
                batch = 0
                for test_data in self.test_data_loader:
                    # pdb.set_trace()
                    print("generating visualization")
                    if self.fixed_test:
                        test_data = next(iter(self.test_data_loader))
                    else:
                        test_data = next(test_data_iter)
                    with torch.no_grad():
                        vis, vis_vals, outimg = self.vis_step(
                            test_data, global_step=step_id)
                    if vis_vals is not None:
                        self.writer.add_scalars("vis",
                                                vis_vals,
                                                global_step=step_id)
                    if vis is not None:
                        vis_u8 = (vis * 255).astype(np.uint8)
                        imageio.imwrite(
                            os.path.join(
                                self.visual_path,
                                "{:04}_{:04}_vis.png".format(epoch, batch),
                            ),
                            vis_u8,
                        )
                    if outimg is not None:
                        if isinstance(outimg, list):
                            if len(outimg) == 4:
                                vis_src, gt, rgb, depth_np = outimg
                                vis_warp = None
                            elif len(outimg) == 5:
                                vis_src, gt, rgb, depth_np, vis_warp = outimg
                            imageio.imwrite(
                                os.path.join(
                                    self.visual_path_com,
                                    "{:04}_{:04}_vis_0_srcs.png".format(epoch, batch),
                                ),
                                (vis_src * 255).astype(np.uint8),
                            )
                            imageio.imwrite(
                                os.path.join(
                                    self.visual_path_com,
                                    "{:04}_{:04}_vis_1_tar.png".format(epoch, batch),
                                ),
                                (gt * 255).astype(np.uint8),
                            )
                            imageio.imwrite(
                                os.path.join(
                                    self.visual_path_com,
                                    "{:04}_{:04}_vis_2_out.png".format(epoch, batch),
                                ),
                                (rgb * 255).astype(np.uint8),
                            )
                            np.save(
                                os.path.join(
                                    self.visual_path_com,
                                    "{:04}_{:04}_vis_3_depth.npy".format(epoch, batch),
                                ), depth_np)
                            if vis_warp is not None:
                                imageio.imwrite(
                                    os.path.join(
                                        self.visual_path_com,
                                        "{:04}_{:04}_vis_4_warps.png".format(epoch, batch),
                                    ),
                                    (vis_warp * 255).astype(np.uint8),
                                )
                        else:
                            vis_u8 = (outimg * 255).astype(np.uint8)
                            imageio.imwrite(
                                os.path.join(
                                    self.visual_path_com,
                                    "{:04}_{:04}_vis_out.png".format(epoch, batch),
                                ),
                                vis_u8,
                            )
                    batch += 1
        elif self.args.only_video:
            save_video_path = os.path.join(self.args.resultdir, self.args.name,
                                           self.args.visual_path)
            os.makedirs(save_video_path, exist_ok=True)
            self.net.eval()
            for epoch in range(1):
                batch = 0
                for test_data in self.test_data_loader:
                    print("generating video")
                    if self.fixed_test:
                        test_data = next(iter(self.test_data_loader))
                    else:
                        test_data = next(test_data_iter)
                    with torch.no_grad():
                        frames, vid_name, others = self.vis_video_step(
                            test_data, global_step=step_id)
                    if frames is not None:
                        vid_path = os.path.join(save_video_path,
                                                "video_" + vid_name + ".mp4")
                        imageio.mimwrite(vid_path,
                                         (frames * 255).astype(np.uint8),
                                         fps=30,
                                         quality=8)
                    if others is not None:
                        if isinstance(others, list):
                            if len(others) == 3:
                                frames_in, gt, depths = others
                                frames_warped = None
                            elif len(others) == 4:
                                frames_in, frames_warped, gt, depths = others
                            
                        if frames_in is not None:
                            vid_path = os.path.join(save_video_path,
                                                    "video_" + vid_name + "_src.png")
                            frames_in = (frames_in * 255).astype(np.uint8)
                            imageio.imwrite(
                                vid_path,
                                frames_in,
                            )
                        if frames_warped is not None:
                            vid_path = os.path.join(save_video_path,
                                                    "video_" + vid_name + "_warped.png")
                            frames_warped = (frames_warped * 255).astype(np.uint8)
                            imageio.imwrite(
                                vid_path,
                                frames_warped,
                            )
                        if gt is not None:
                            vid_path = os.path.join(save_video_path,
                                                    "video_" + vid_name + "_tar.png")
                            gt = (gt * 255).astype(np.uint8)
                            imageio.imwrite(
                                vid_path,
                                gt,
                            )
                        if depths is not None:
                            vid_path = os.path.join(save_video_path,
                                                    "video_" + vid_name + "_depths.npy")
                            np.save(vid_path, depths)
                        batch += 1
        else:
            if self.args.warp_pretrain:
                training_step = self.warp_pretrain_step
            else:
                training_step = self.train_step
            progress = tqdm.tqdm(bar_format="[{rate_fmt}] ")
            for epoch in range(self.num_epochs):
                self.writer.add_scalar("lr",
                                       self.optim.param_groups[0]["lr"],
                                       global_step=step_id)
                batch = 0
                for _ in range(self.num_epoch_repeats):
                    for data in self.train_data_loader:
                        losses = training_step(data, global_step=step_id)
                        loss_str = fmt_loss_str(losses)
                        if batch % self.print_interval == 0:
                            print(
                                "E",
                                epoch,
                                "B",
                                batch,
                                loss_str,
                                " lr",
                                self.optim.param_groups[0]["lr"],
                            )

                        if self.args.warp_pretrain:
                            if batch % self.save_interval == 0 and (
                                    epoch > 0 or batch > 0):
                                print("saving")
                                self.extra_save_state()
                            if batch % self.vis_interval == 0:
                                print("generating visualization")
                                if self.fixed_test:
                                    test_data = next(
                                        iter(self.test_data_loader))
                                else:
                                    test_data = next(test_data_iter)
                                with torch.no_grad():
                                    vis = self.warp_vis_step(
                                        test_data, global_step=step_id)
                                if vis is not None:
                                    vis_u8 = (vis * 255).astype(np.uint8)
                                    imageio.imwrite(
                                        os.path.join(
                                            self.visual_path,
                                            "{:04}_{:04}_vis.png".format(
                                                epoch, batch),
                                        ),
                                        vis_u8,
                                    )
                        else:
                            if batch % self.eval_interval == 0:
                                test_data = next(test_data_iter)
                                self.net.eval()
                                with torch.no_grad():
                                    test_losses = self.eval_step(
                                        test_data, global_step=step_id)
                                self.net.train()
                                test_loss_str = fmt_loss_str(test_losses)
                                self.writer.add_scalars("train",
                                                        losses,
                                                        global_step=step_id)
                                self.writer.add_scalars("test",
                                                        test_losses,
                                                        global_step=step_id)
                                print("*** Eval:", "E", epoch, "B", batch,
                                      test_loss_str, " lr")

                            if batch % self.save_interval == 0 and (
                                    epoch > 0 or batch > 0):
                                print("saving")
                                self.net.save_weights(self.args)
                                # if self.managed_weight_saving:
                                #     self.net.save_weights(self.args)
                                # else:
                                #     torch.save(self.net.state_dict(),
                                #                self.default_net_state_path)
                                torch.save(self.optim.state_dict(),
                                           self.optim_state_path)
                                if self.lr_scheduler is not None:
                                    torch.save(self.lr_scheduler.state_dict(),
                                               self.lrsched_state_path)
                                torch.save({"iter": step_id + 1},
                                           self.iter_state_path)
                                self.extra_save_state()

                            if batch % self.vis_interval == 0:
                                print("generating visualization")
                                if self.fixed_test:
                                    test_data = next(
                                        iter(self.test_data_loader))
                                else:
                                    test_data = next(test_data_iter)
                                self.net.eval()
                                with torch.no_grad():
                                    vis, vis_vals = self.vis_step(
                                        test_data, global_step=step_id)
                                if vis_vals is not None:
                                    self.writer.add_scalars(
                                        "vis", vis_vals, global_step=step_id)
                                self.net.train()
                                if vis is not None:
                                    vis_u8 = (vis * 255).astype(np.uint8)
                                    imageio.imwrite(
                                        os.path.join(
                                            self.visual_path,
                                            "{:04}_{:04}_vis.png".format(
                                                epoch, batch),
                                        ),
                                        vis_u8,
                                    )

                        if (batch == self.num_total_batches - 1 or
                                batch % self.accu_grad == self.accu_grad - 1):
                            self.optim.step()
                            self.optim.zero_grad()

                        if not self.args.warp_pretrain:
                            self.post_batch(epoch, batch)
                        step_id += 1
                        batch += 1
                        progress.update(1)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
