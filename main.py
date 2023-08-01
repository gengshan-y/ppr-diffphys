import pdb
import time
from absl import app
from absl import flags

from diffphys.warp_env import phys_model
from diffphys.vis import Logger
from diffphys.dataloader import DataLoader

# distributed data parallel
flags.DEFINE_integer("local_rank", 0, "for distributed training")
flags.DEFINE_integer("ngpu", 1, "number of gpus to use")

flags.DEFINE_integer("accu_steps", 1, "how many steps to do gradient accumulation")
flags.DEFINE_string("seqname", "shiba-haru-1002", "name of the sequence")
flags.DEFINE_string("logroot", "logdir/", "Root directory for output files")
flags.DEFINE_string("logname", "dynamics", "Experiment Name")
flags.DEFINE_float("learning_rate", 2e-4, "learning rate")
flags.DEFINE_integer("num_epochs", 5, "total update iterations")
flags.DEFINE_string("urdf_template", "a1", "whether to use predefined skeleton")
flags.DEFINE_integer("num_freq", 10, "number of freqs in fourier encoding")
flags.DEFINE_integer("t_embed_dim", 128, "dimension of the pose code")
flags.DEFINE_integer("iters_per_epoch", 100, "iters per epoch")


def main(_):
    opts = flags.FLAGS
    opts = opts.flag_values_dict()

    vis = Logger(opts)
    dataloader = DataLoader(opts)

    # model
    model = phys_model(opts, dataloader)
    model.cuda()

    # opt
    for it in range(opts["num_epochs"] * opts["iters_per_epoch"] + 1):
        model.progress = it / (opts["num_epochs"] * opts["iters_per_epoch"])

        # eval
        if it % opts["iters_per_epoch"] == 0:
            # save net
            model.save_network(epoch_label=it)

            # inference
            model.reinit_envs(1, wdw_length=model.gt_steps, is_eval=True)
            model.forward()
            data = model.query()
            vis.show(it, data)

            # training
            # model.reinit_envs(100, wdw_length=1,is_eval=False)
            model.reinit_envs(10, wdw_length=8, is_eval=False)
            ##TODO schedule window length
            # wdw_length = int(0.5*(model.gt_steps - 1)/["total_iters"]*it + 1)
            # num_envs = max(1,int(100 / wdw_length))
            # print('wdw/envs: %d/%d'%(wdw_length, num_envs))
            # model.reinit_envs(num_envs, wdw_length=wdw_length,is_eval=False)

        # train
        t = time.time()
        loss = 0
        for accu_it in range(opts["accu_steps"]):
            loss_dict = model.forward()
            loss += loss_dict["total_loss"]
        loss = loss / float(opts["accu_steps"])
        model.backward(loss)
        grad_list = model.update()
        print(it)
        log_data = loss_dict
        log_data["iter_time"] = time.time() - t
        log_data["loss"] = loss
        for k, v in grad_list.items():
            log_data[k] = v
        vis.write_log(log_data, it)


if __name__ == "__main__":
    app.run(main)
