import argparse
import shutil

import torch.distributed as dist
import torch.utils.tensorboard
import yaml
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm
import sys 
from models.data.config import get_dataset_info
from models.net.MGD_model import MGD
from models.utils import prepare_context, compute_mean_mad
from utils_.common import get_optimizer, get_scheduler
from utils_.datasets import CustomDataset
from utils_.misc import *
from utils_.transforms import *

def train(it, model, train_loader, optimizer_global, optimizer_local, config, writer, logger, device, property_norms):
    model.train()
    sum_loss, sum_n = 0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0, 0
    sum_loss_node_global, sum_loss_node_local = 0, 0
    with tqdm(total=len(train_loader), desc='Training',leave=True) as pbar:
        for batch in train_loader:
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            batch = batch.to(device)
            # Prepare context if needed
            context = prepare_context(config.context, batch, property_norms)

            loss_vae_kl = 0.00

            loss = model(
                batch,
                context=context,
                return_unreduced_loss=True
            )

            if config.model.vae_context:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, loss_vae_kl = loss
                loss_vae_kl = loss_vae_kl.mean().item()
            else:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local = loss
            # print(f"loss:{loss},loss_pos_global:{loss_pos_global},loss_pos_local:{loss_pos_local},
            # loss_node_global:{loss_node_global},loss_node_local:{loss_node_local}")
            loss = loss.mean()
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer_global.step()
            optimizer_local.step()
            sum_loss += loss.item()
            sum_n += 1
            sum_loss += loss.mean().item()
            sum_loss_pos_global += loss_pos_global.mean().item()
            sum_loss_node_global += loss_node_global.mean().item()
            sum_loss_pos_local += loss_pos_local.mean().item()
            sum_loss_node_local += loss_node_local.mean().item()
            pbar.set_postfix({'loss': '%.2f' % (loss.item())})
            pbar.update(1)

    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / sum_n
    avg_loss_node_global = sum_loss_node_global / sum_n
    avg_loss_pos_local = sum_loss_pos_local / sum_n
    avg_loss_node_local = sum_loss_node_local / sum_n


    logger.info(
        f'[Train] Iter {it:05d} | Loss {loss.item():.2f} | '
        f'Loss(pos_Global) {avg_loss_pos_global:.2f} | Loss(pos_Local) {avg_loss_pos_local:.2f} | '
        f'Loss(node_global) {avg_loss_node_global:.2f} | Loss(node_local) {avg_loss_node_local:.2f} | '
        f'Loss(vae_KL) {loss_vae_kl:.2f} |Grad {orig_grad_norm:.2f} | '
        f'LR {optimizer_global.param_groups[0]["lr"]:.6f}'
    )
    writer.add_scalar('train/loss', avg_loss, it)
    writer.add_scalar('train/loss_pos_global', avg_loss_pos_global, it)
    writer.add_scalar('train/loss_node_global', avg_loss_node_global, it)
    writer.add_scalar('train/loss_pos_local', avg_loss_pos_local, it)
    writer.add_scalar('train/loss_node_local', avg_loss_node_local, it)
    writer.add_scalar('train/loss_vae_KL', loss_vae_kl, it)
    writer.add_scalar('train/lr', optimizer_global.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

    return avg_loss


def validate(it, model, val_loader, config, writer, logger, device, property_norms):
    model.eval()
    sum_loss, sum_n = 0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0, 0
    sum_loss_node_global, sum_loss_node_local = 0, 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_loader, desc=f'Validation Iter {it}'):
            batch = batch.to(device)
            # Prepare context if needed
            context = prepare_context(config.context, batch, property_norms)

            loss = model(
                batch,
                context=context,
                return_unreduced_loss=True
            )

            if config.model.vae_context:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, loss_vae_kl = loss
                # loss_vae_kl = loss_vae_kl.mean().item()
            else:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local = loss

            # print(loss)
            sum_loss += loss.sum().item()
            sum_n += loss.size(0)
            sum_loss_pos_global += loss_pos_global.mean().item()
            sum_loss_node_global += loss_node_global.mean().item()
            sum_loss_pos_local += loss_pos_local.mean().item()
            sum_loss_node_local += loss_node_local.mean().item()

    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / sum_n
    avg_loss_node_global = sum_loss_node_global / sum_n
    avg_loss_pos_local = sum_loss_pos_local / sum_n
    avg_loss_node_local = sum_loss_node_local / sum_n

    if config.train.scheduler.type == 'plateau':
        scheduler_global.step(avg_loss_pos_global + avg_loss_node_global)
        scheduler_local.step(avg_loss_pos_local + avg_loss_node_local)
    else:
        scheduler_global.step()
        if 'global' not in config.model.network:
            scheduler_local.step()


    logger.info('[Validate] Iter %05d | Loss %.6f ' % (
        it, avg_loss
    ))

    writer.add_scalar('val/loss', avg_loss, it)
    writer.flush()
    return avg_loss


#train
if __name__ == '__main__':
    # Load configuration
    config_path = 'config.yaml'  # Update this path to your config file location
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    # Set device
    device = 'cuda' if config.cuda and torch.cuda.is_available() else 'cpu'
    seed_all(42)

    # Set up logging and directories
    log_dir = get_new_log_dir(config.logdir, prefix='model_training')
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    # Log config
    logger.info('Loaded configuration:')
    logger.info(config)

    # Dataset preparation
    dataset_info = get_dataset_info(config.dataset, remove_h=False)
    transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])

    train_set = CustomDataset('train', pre_transform=transforms)
    val_set = CustomDataset('valid', pre_transform=transforms)

    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False)

    # Compute property normalization if context is provided
    property_norms, property_norms_val = None, None
    if config.context:
        property_norms = compute_mean_mad(train_set, config.context, config.dataset)
        property_norms_val = compute_mean_mad(val_set, config.context, config.dataset)

    # Build model
    logger.info('Building model...')
    model = MDMFullDP(config.model).to(device)
    # print(f"Model context: {model.context}")  
    # print(f"config context: {config.context}")

    # Optimizers and schedulers
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)

    # Training loop
    best_train_loss = float('inf')
    for it in range(1, config.train.max_iters + 1):
        avg_train_loss = train(it, model, train_loader, optimizer_global, optimizer_local, config, writer, logger,
                               device,
                               property_norms)
        if avg_train_loss < best_train_loss:
            ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
            torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'scheduler_local': scheduler_local.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_train_loss,
                }, ckpt_path)
            logger.info(f'Model checkpoint saved at Iter {it}')
            best_train_loss = avg_train_loss