import os
import pdb
import time
from absl import app
from absl import flags
import torch
import gc
import tqdm

from diffphys.dp_model import phys_model
from diffphys.vis import PhysVisualizer
from diffphys.dataloader import DataLoader

# distributed data parallel
flags.DEFINE_integer("local_rank", 0, "for distributed training")
flags.DEFINE_integer("ngpu", 1, "number of gpus to use")

flags.DEFINE_integer("accu_steps", 1, "how many steps to do gradient accumulation")
flags.DEFINE_string("seqname", "shiba-haru-1002", "name of the sequence")
flags.DEFINE_string("logroot", "logdir/", "Root directory for output files")
flags.DEFINE_string("logname", "dynamics", "Experiment Name")
flags.DEFINE_float("phys_learning_rate", 1e-4, "learning rate")
flags.DEFINE_integer("num_rounds", 5, "total update iterations")
flags.DEFINE_integer("warmup_iters", 0, "warmup iterations, only >0 for DR+DP")
flags.DEFINE_string("urdf_template", "a1", "whether to use predefined skeleton")
flags.DEFINE_integer("num_freq", 10, "number of freqs in fourier encoding")
flags.DEFINE_integer("t_embed_dim", 128, "dimension of the pose code")
flags.DEFINE_integer("iters_per_round", 20, "iters per epoch")
flags.DEFINE_float("ratio_phys_cycle", 1.0, "iters per epoch")
flags.DEFINE_float("noise_std", 2e-3, "noise std added to initial states")

flags.DEFINE_float("traj_wt", 0.01, "weight for traj matching loss")
flags.DEFINE_float("pos_state_wt", 0.01, "weight for position matching reg")
flags.DEFINE_float("vel_state_wt", 1e-4, "weight for velocity matching reg")
flags.DEFINE_float("pos_distill_wt", 0.0, "weight for distilling proxy kienmatics")

# regs
flags.DEFINE_float("reg_torque_wt", 0.0, "weight for torque regularization")
flags.DEFINE_float("reg_res_f_wt", 0.0, "weight for residual force regularization")
flags.DEFINE_float("reg_foot_wt", 0.0, "weight for foot contact regularization")
flags.DEFINE_float("reg_root_wt", 0.0, "weight for root pose regularization")

# flags.DEFINE_float("reg_pose_state_wt", 0.01, "weight for position matching reg")
# flags.DEFINE_float("reg_vel_state_wt", 1e-6, "weight for velocity matching reg")
# flags.DEFINE_float("reg_torque_wt", 1e-5, "weight for torque regularization")
# flags.DEFINE_float("reg_res_f_wt", 5e-5, "weight for residual force regularization")
# flags.DEFINE_float("reg_foot_wt", 1e-4, "weight for foot contact regularization")


def main(_):
    opts = flags.FLAGS
    opts = opts.flag_values_dict()

    logname = "%s-%s" % (opts["seqname"], opts["logname"])
    save_dir = os.path.join(opts["logroot"], logname)
    vis = PhysVisualizer(save_dir)
    dataloader = DataLoader(opts)

    # model
    model = phys_model(opts, dataloader)
    model.cuda()

    # opt
    for it in tqdm.tqdm(range(model.total_iters)):
        model.progress = it / (opts["num_rounds"] * opts["iters_per_round"])

        # gc.collect()  # need to be used together with empty_cache()
        # torch.cuda.empty_cache()
        # print("allocated: %.2f M" % (torch.cuda.memory_allocated() / (1024**2)))
        # print("cached: %.2f M" % (torch.cuda.memory_cached() / (1024**2)))

        # eval
        if it % opts["iters_per_round"] == 0:
            # save net
            model.save_checkpoint(it)

            # inference
            model.reinit_envs(1, frames_per_wdw=model.total_frames, is_eval=True)
            model.forward()
            data = model.query()
            vis.show(it, data, fps=1.0 / model.frame_interval)

            # training
            # model.reinit_envs(100, frames_per_wdw=2, is_eval=False)
            # model.reinit_envs(10, frames_per_wdw=8, is_eval=False)
            model.reinit_envs(10, frames_per_wdw=24, is_eval=False)
            ##TODO schedule window length
            # frames_per_wdw = int(0.5*(model.total_frames - 1)/["total_iters"]*it + 1)
            # num_envs = max(1,int(100 / frames_per_wdw))
            # print('wdw/envs: %d/%d'%(frames_per_wdw, num_envs))
            # model.reinit_envs(num_envs, frames_per_wdw=frames_per_wdw,is_eval=False)

        # train
        t = time.time()
        loss = 0
        for accu_it in range(opts["accu_steps"]):
            loss_dict = model.forward()
            loss += loss_dict["total_loss"]
        loss = loss / float(opts["accu_steps"])
        model.backward(loss)
        grad_dict = model.update()
        loss_dict.update(grad_dict)
        loss_dict["iter_time"] = time.time() - t
        loss_dict["loss"] = loss
        vis.write_log(loss_dict, it)


if __name__ == "__main__":
    app.run(main)
