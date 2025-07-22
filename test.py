import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #,1

import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
import h5py


from models.uni3d import create_uni3d
import utils.open_clip as open_clip

import psutil
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from FGM import PGD
from CWPerturb import CWPerturb
from drop import SaliencyDrop
from KNN import CWKNN
from ADD import CWAdd
from models.point_encoder import fps6d
from DUP_Net.DUP_Net import DUPNet
from AOF import CWAOF
from AdvPC import CWAdvPC


parser = argparse.ArgumentParser(description="Uni3D Model Testing")
parser.add_argument("--pc_model", type=str, default="eva02_base_patch14_448", help="Point cloud model name")
parser.add_argument("--pc_feat_dim", type=int, default=768, help="Point cloud feature dimension")
parser.add_argument("--pretrained_pc", type=str, default=None, help="Path to pretrained point cloud model")
parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Drop path rate")
parser.add_argument("--pc_encoder_dim", type=int, default=512, help="Point cloud encoder dimension")
parser.add_argument("--embed_dim", type=int, default=1024, help="Embedding dimension")
parser.add_argument("--ckpt_path", type=str, default="XXX", help="Path to model checkpoint")
parser.add_argument("--num_group", type=int, default=512, help="Number of groups for point cloud processing")
parser.add_argument("--group_size", type=int, default=64, help="Group size for point cloud processing")
parser.add_argument("--prompt_ckpt", type=str, default="", help="Path to prompt checkpoint")
parser.add_argument("--task", type=str, default="Modelnet40", help="Task to perform: test or train")
args = parser.parse_args()


from sklearn.neighbors import NearestNeighbors

def sor_filter(batch_pc, k=16, alpha=1.0):
    """
    Statistical Outlier Removal on batched point clouds [B, K, 6].
    Only xyz used for outlier detection, rgb is preserved.
    
    Args:
        batch_pc: torch.Tensor, shape [B, K, 6]
        k: int, number of neighbors
        alpha: float, threshold scale factor

    Returns:
        filtered_pc: list of [N_i, 6] filtered tensors (varied number per batch)
    """
    B, K, _ = batch_pc.shape
    filtered_pcs = []

    for b in range(B):
        pc = batch_pc[b].cpu().numpy()  # shape [K, 6]
        xyz = pc[:, :3]

        # Use sklearn's NearestNeighbors for kNN
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xyz)
        distances, _ = nbrs.kneighbors(xyz)
        avg_dists = np.mean(distances[:, 1:], axis=1)  # skip self-distance

        # Thresholding
        mean = np.mean(avg_dists)
        std = np.std(avg_dists)
        threshold = mean + alpha * std
        mask = avg_dists < threshold

        filtered_pc = pc[mask]  # keep corresponding RGB
        filtered_pcs.append(torch.tensor(filtered_pc, dtype=batch_pc.dtype, device=batch_pc.device))

    return filtered_pcs  # List of tensors with shape [N_i, 6]

def srs_sample(batch_pc, num_samples=500):
    """
    Simple Random Sampling on batched point clouds [B, K, 6].
    Args:
        batch_pc: torch.Tensor, shape [B, K, 6]
        num_samples: int, number of points to sample from each batch

    Returns:
        sampled_pc: torch.Tensor, shape [B, num_samples, 6]
    """
    B, K, _ = batch_pc.shape
    logger.info(f"Number of points: {K}")
    num_samples = K - num_samples
    #assert num_samples <= K, "num_samples must be less than or equal to K"

    idx = torch.rand(B, K).argsort(dim=1)[:, :num_samples]
    batch_indices = torch.arange(B).unsqueeze(1).expand(-1, num_samples)

    sampled_pc = batch_pc[batch_indices, idx]  # shape [B, num_samples, 6]
    return sampled_pc

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []
        self.labels = []
        self.classnames = []

        classnames_path = os.path.join(root_dir, 'modelnet40_shape_names.txt')
        if os.path.exists(classnames_path):
            with open(classnames_path, 'r') as f:
                self.classnames = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Classnames file not found: {classnames_path}")

        split_file = os.path.join(root_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            file_list = [line.strip() for line in f.readlines()]

        for item in file_list:
            category = item.rsplit('_', 1)[0]
            file_path = os.path.join(root_dir, category, f'{item}.txt')
            if not os.path.exists(file_path):
                print(f'[Warning] File not found: {file_path}')
                continue
            self.data.append(file_path)
            self.labels.append(self.classnames.index(category))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pc_path = self.data[idx]
        label = self.labels[idx]
        points = np.loadtxt(pc_path, delimiter=',').astype(np.float32)[:, 0:3]
        if self.transform:
            points = self.transform(points)
        return points, label

def print_memory(tag):
    cpu_mem = psutil.virtual_memory()
    print(f"[{tag}] CPU Memory Used: {(cpu_mem.total - cpu_mem.available) / (1024**2):.2f} MB / {cpu_mem.total / (1024**2):.2f} MB")

transform = transforms.Compose([
    lambda pc: np.hstack((pc[:, :3], np.full((pc.shape[0], 3), 0.5))),
    lambda pc: torch.tensor(pc, dtype=torch.float32)
])


def cos_loss(point_features, text_features):
    point_features = F.normalize(point_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    cosine_sim = F.cosine_similarity(point_features, text_features, dim=-1)
    return -cosine_sim.mean()

class ClipPointsL2(nn.Module):

    def __init__(self, budget):
        """Clip point cloud with a given global l2 budget.

        Args:
            budget (float): perturbation budget
        """
        super(ClipPointsL2, self).__init__()

        self.budget = budget

    def forward(self, pc, ori_pc):
        """Clipping every point in a point cloud.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
        """
        with torch.no_grad():
            diff = pc - ori_pc  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=[1, 2]) ** 0.5  # [B]
            scale_factor = self.budget / (norm + 1e-9)  # [B]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B]
            diff = diff * scale_factor[:, None, None]
            pc = ori_pc + diff
        return pc
class ClipPointsLinf(nn.Module):

    def __init__(self, budget):
        """Clip point cloud with a given l_inf budget.

        Args:
            budget (float): perturbation budget
        """
        super(ClipPointsLinf, self).__init__()

        self.budget = budget

    def forward(self, pc, ori_pc):
        """Clipping every point in a point cloud.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
        """
        with torch.no_grad():
            diff = pc - ori_pc  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=1) ** 0.5  # [B, K]
            scale_factor = self.budget / (norm + 1e-9)  # [B, K]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B, K]
            diff = diff * scale_factor[:, None, :]
            pc = ori_pc + diff
        return pc
class ProjectInnerPoints(nn.Module):

    def __init__(self):
        """Eliminate points shifted inside an object.
        Introduced by AAAI'20 paper.
        """
        super(ProjectInnerPoints, self).__init__()

    def forward(self, pc, ori_pc, normal=None):
        """Clipping "inside" points to the surface of the object.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
            normal (torch.FloatTensor, optional): normals. Defaults to None.
        """
        with torch.no_grad():
            # in case we don't have normals
            if normal is None:
                return pc
            diff = pc - ori_pc
            inner_diff_normal = torch.sum(
                diff * normal, dim=1)  # [B, K]
            inner_mask = (inner_diff_normal < 0.)  # [B, K]

            # clip to surface!
            # 1) vng = Normal x Perturb
            vng = torch.cross(normal, diff, dim=1)  # [B, 3, K]
            vng_norm = torch.sum(vng ** 2, dim=1) ** 0.5  # [B, K]

            # 2) vref = vng x Normal
            vref = torch.cross(vng, normal)  # [B, 3, K]
            vref_norm = torch.sum(vref ** 2, dim=1) ** 0.5  # [B, K]

            # 3) Project Perturb onto vref
            diff_proj = diff * vref / \
                (vref_norm[:, None, :] + 1e-9)  # [B, 3, K]

            # some diff is completely opposite to normal
            # just set them to (0, 0, 0)
            opposite_mask = inner_mask & (vng_norm < 1e-6)
            opposite_mask = opposite_mask.\
                unsqueeze(1).expand_as(diff_proj)
            diff_proj[opposite_mask] = 0.

            # set inner points with projected perturbation
            inner_mask = inner_mask.\
                unsqueeze(1).expand_as(diff)
            diff[inner_mask] = diff_proj[inner_mask]
            pc = ori_pc + diff
        return pc
class ProjectInnerClipLinf(nn.Module):

    def __init__(self, budget):
        """Project inner points to the surface and
        clip the l_inf norm of perturbation.

        Args:
            budget (float): l_inf norm budget
        """
        super(ProjectInnerClipLinf, self).__init__()

        self.project_inner = ProjectInnerPoints()
        self.clip_linf = ClipPointsLinf(budget=budget)

    def forward(self, pc, ori_pc, normal=None):
        """Project to the surface and then clip.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
            normal (torch.FloatTensor, optional): normals. Defaults to None.
        """
        with torch.no_grad():
            # project
            pc = self.project_inner(pc, ori_pc, normal)
            # clip
            pc = self.clip_linf(pc, ori_pc)
        return pc
class HausdorffDist(nn.Module):

    def __init__(self, method='adv2ori'):
        """Compute hausdorff distance between two point clouds.

        Args:
            method (str, optional): type of hausdorff. Defaults to 'adv2ori'.
        """
        super(HausdorffDist, self).__init__()

        self.method = method

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute hausdorff distance between two point clouds.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        loss1, loss2 = hausdorff(adv_pc, ori_pc)  # [B], adv2ori, ori2adv
        if self.method == 'adv2ori':
            loss = loss1
        elif self.method == 'ori2adv':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss


def contrastive_loss(point_features, text_features, tau=0.07):
    B = point_features.size(0)
    logits = torch.matmul(point_features, text_features.t()) / tau
    labels = torch.arange(B, device=point_features.device, dtype=torch.long)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.t(), labels)
    return (loss_p2t + loss_t2p) * 0.5
class UntargetedLogitsAdvLoss(nn.Module):

    def __init__(self, kappa=0.):
        """Adversarial function on logits.

        Args:
            kappa (float, optional): min margin. Defaults to 0..
        """
        super(UntargetedLogitsAdvLoss, self).__init__()

        self.kappa = kappa

    def forward(self, logits, targets):
        """Adversarial loss function using logits.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        B, K = logits.shape
        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()
        one_hot_targets = torch.zeros(B, K).cuda().scatter_(
            1, targets, 1).float()  # to one-hot
        real_logits = torch.sum(one_hot_targets * logits, dim=1)
        other_logits = torch.max((1. - one_hot_targets) * logits -
                                 one_hot_targets * 10000., dim=1)[0]
        loss = torch.clamp(real_logits - other_logits + self.kappa, min=0.)
        return loss.mean()



class _Distance(nn.Module):

    def __init__(self):
        super(_Distance, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        pass

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))  # [B, K, K]
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(
            1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
class ChamferDistance(_Distance):

    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, preds, gts):
        """
        preds: [B, N1, 3]
        gts: [B, N2, 3]
        """
        P = self.batch_pairwise_dist(gts, preds)  # [B, N2, N1]
        mins, _ = torch.min(P, 1)  # [B, N1], find preds' nearest points in gts
        loss1 = torch.mean(mins, dim=1)  # [B]
        mins, _ = torch.min(P, 2)  # [B, N2], find gts' nearest points in preds
        loss2 = torch.mean(mins, dim=1)  # [B]
        return loss1, loss2
class HausdorffDistance(_Distance):

    def __init__(self):
        super(HausdorffDistance, self).__init__()

    def forward(self, preds, gts):
        """
        preds: [B, N1, 3]
        gts: [B, N2, 3]
        """
        P = self.batch_pairwise_dist(gts, preds)  # [B, N2, N1]
        # max_{y \in pred} min_{x \in gt}
        mins, _ = torch.min(P, 1)  # [B, N1]
        loss1 = torch.max(mins, dim=1)[0]  # [B]
        # max_{y \in gt} min_{x \in pred}
        mins, _ = torch.min(P, 2)  # [B, N2]
        loss2 = torch.max(mins, dim=1)[0]  # [B]
        return loss1, loss2
chamfer = ChamferDistance()
hausdorff = HausdorffDistance()
class ClipPointsL2(nn.Module):

    def __init__(self, budget):
        """Clip point cloud with a given global l2 budget.

        Args:
            budget (float): perturbation budget
        """
        super(ClipPointsL2, self).__init__()

        self.budget = budget

    def forward(self, pc, ori_pc):
        """Clipping every point in a point cloud.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
        """
        with torch.no_grad():
            diff = pc - ori_pc  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=[1, 2]) ** 0.5  # [B]
            scale_factor = self.budget / (norm + 1e-9)  # [B]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B]
            diff = diff * scale_factor[:, None, None]
            pc = ori_pc + diff
        return pc
class CrossEntropyAdvLoss(nn.Module):

    def __init__(self):
        """Adversarial function on output probabilities.
        """
        super(CrossEntropyAdvLoss, self).__init__()

    def forward(self, logits, targets):
        """Adversarial loss function using cross entropy.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        loss = F.cross_entropy(logits, targets)
        return loss
class L2Dist(nn.Module):

    def __init__(self):
        """Compute global L2 distance between two point clouds.
        """
        super(L2Dist, self).__init__()

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute L2 distance between two point clouds.
        Apply different weights for batch input for CW attack.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3] or [B, 3, K]
            ori_pc (torch.FloatTensor): [B, K, 3] or [B, 3, k]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        dist = torch.sqrt(torch.sum(
            (adv_pc - ori_pc) ** 2, dim=[1, 2]))  # [B]
        dist = dist * weights
        if batch_avg:
            return dist.mean()
        return dist
class KNNDist(nn.Module):

    def __init__(self, k=5, alpha=1.05):
        """Compute kNN distance punishment within a point cloud.

        Args:
            k (int, optional): kNN neighbor num. Defaults to 5.
            alpha (float, optional): threshold = mean + alpha * std. Defaults to 1.05.
        """
        super(KNNDist, self).__init__()

        self.k = k
        self.alpha = alpha

    def forward(self, pc, weights=None, batch_avg=True):
        """KNN distance loss described in AAAI'20 paper.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B]. Defaults to None.
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        # build kNN graph
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K], l2^2

        dist = torch.clamp(dist, min=-1e-6)
        #assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
        # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # d_p, [B, K]
        with torch.no_grad():
            mean = torch.mean(value, dim=-1)  # [B]
            std = torch.std(value, dim=-1)  # [B]
            # [B], penalty threshold for batch
            threshold = mean + self.alpha * std
            weight_mask = (value > threshold[:, None]).\
                float().detach()  # [B, K]
        loss = torch.mean(value * weight_mask, dim=1)  # [B]
        # accumulate loss
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss
class ChamferDist(nn.Module):

    def __init__(self, method='adv2ori'):
        """Compute chamfer distance between two point clouds.

        Args:
            method (str, optional): type of chamfer. Defaults to 'adv2ori'.
        """
        super(ChamferDist, self).__init__()

        self.method = method

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute chamfer distance between two point clouds.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        loss1, loss2 = chamfer(adv_pc, ori_pc)  # [B], adv2ori, ori2adv
        if self.method == 'adv2ori':
            loss = loss1
        elif self.method == 'ori2adv':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss
class ChamferkNNDist(nn.Module):

    def __init__(self, chamfer_method='adv2ori',
                 knn_k=5, knn_alpha=1.05,
                 chamfer_weight=5., knn_weight=3.):
        """Geometry-aware distance function of AAAI'20 paper.

        Args:
            chamfer_method (str, optional): chamfer. Defaults to 'adv2ori'.
            knn_k (int, optional): k in kNN. Defaults to 5.
            knn_alpha (float, optional): alpha in kNN. Defaults to 1.1.
            chamfer_weight (float, optional): weight factor. Defaults to 5..
            knn_weight (float, optional): weight factor. Defaults to 3..
        """
        super(ChamferkNNDist, self).__init__()

        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.knn_dist = KNNDist(k=knn_k, alpha=knn_alpha)
        self.w1 = chamfer_weight
        self.w2 = knn_weight

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Adversarial constraint function of AAAI'20 paper.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        knn_loss = self.knn_dist(
            adv_pc, weights=weights, batch_avg=batch_avg)
        loss = chamfer_loss * self.w1 + knn_loss * self.w2
        return loss


def random_target(labels, num_classes):
    targets = torch.randint_like(labels, low=0, high=num_classes)
    targets = torch.where(targets == labels, (targets + 1) % num_classes, targets)
    return targets

def test_clean_with_list(model, test_loader, text_features=None, batch_text_features=None, prompt=None):
    if batch_text_features is None:
        batch_text_features = text_features

    logger.info(f"text_features shape: {text_features.shape}")
    correct = 0
    total = 0
    model.eval()
    save = True

    for data in test_loader:
        original_data = data[0].cuda()  # [B, K, 6]
        labels = data[1].to('cuda')
        
        filtered_list = sor_filter(original_data)  # List of [N_i, 6] tensors

        logger.info("begin eva")
        for i, pc in enumerate(filtered_list):
            with torch.no_grad():
                pc = pc.unsqueeze(0)  # [1, N_i, 6]
                if prompt is not None:
                    prompt_now = prompt.expand(1, -1, -1)
                    point_feature = model.encode_pc(pc, prompt_now)
                else:
                    point_feature = model.encode_pc(pc)

                point_feature = point_feature / point_feature.norm(dim=-1, keepdim=True)
                similarity_point = torch.matmul(point_feature, batch_text_features.T)
                prediction = similarity_point.argmax(dim=-1)  # [1]
                correct += (prediction == labels[i:i+1]).sum().item()
                total += 1

                # 保存第一个点云
                if save and i == 0:
                    np.save('clean_pointcloud.npy', pc.squeeze(0).cpu().numpy())
                    print("Saved batch 0's first point cloud to clean_pointcloud.npy!")
                    save = False

    accuracy = correct / total
    print(f"ACC: {accuracy:.4f}")
    return accuracy
def test_clean(model, test_loader,text_features=None,batch_text_features= None, prompt=None,defense = None):
    all_points = []
    all_labels = []
    if(batch_text_features is not None):
        logger.info(f"batch_text_features shape: {batch_text_features.shape}")
        logger.info(f"text_features shape: {text_features.shape}")
    else:
        batch_text_features = text_features
    correct = 0
    total = 0
    model.eval()
    save = False
    for data in test_loader:
        #perturbed_data = fps6d(data[0].cuda(),1024) #data[0].cuda() #
        perturbed_data = data[0].cuda()
        if defense == 'SRS':
            perturbed_data = srs_sample(perturbed_data)
        elif defense == 'DUP':
            #perturbed_data = fps6d(data[0].cuda(),2048) #
            perturbed_data = data[0].cuda()
            xyz = perturbed_data[..., :3]                   # [B, K, 3]
            feat = perturbed_data[..., 3:]                  # [B, K, 3]
            defended_xyz = dupnet(xyz)               # [B, K', 3] 或 list of [K', 3]
            B, K_new, _ = defended_xyz.shape
            repeated_feat = feat[:, :1, :].repeat(1, K_new, 1)  # [B, K', 3]
            perturbed_data = torch.cat([defended_xyz, repeated_feat], dim=-1)  # [B, K', 6]'''

        #perturbed_data = srs_sample(perturbed_data)
        if(save):
            first_point_cloud = perturbed_data[0].cpu().detach().numpy()  # 取 batch 0，转 numpy
            #np.save('clean_pointcloud.txt', first_point_cloud)
            np.savetxt('perturb_pointcloud.txt', first_point_cloud, fmt='%.6f', delimiter=',')
            print("Saved batch 0's first point cloud to first_perturbed_pointcloud.npy!")
            save = False
        logger.info("begin eva")
        with torch.no_grad():
            point_clouds = perturbed_data
            if not prompt == None:
                B = point_clouds.shape[0]
                prompt_now = prompt.expand(B, -1, -1)
                point_features = model.encode_pc(point_clouds, prompt_now) #, text_features
            else:
                point_features = model.encode_pc(point_clouds)
            point_features /= point_features.norm(dim=-1, keepdim=True)
            #batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)
            similarity_point = torch.matmul(point_features, batch_text_features.T)
            predictions = similarity_point.argmax(dim=-1)
            logger.info(f"Predictions: {predictions}")
            correct += (predictions == data[1].cuda()).sum().item() ##.to('cuda:0')
            total += data[1].size(0)

        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    np.savez('test_data_clean.npz', points=all_points, labels=all_labels)

    accuracy = correct / total
    print(f"ACC: {accuracy:.4f}")
    return accuracy

def pgd_attack(model, data, text_features=None, budget=0.05, step_size=0.01, num_iter=10, dist_metric='l2'):
    """
    PGD attack bridge function for test_attack.

    Args:
        model (torch.nn.Module): The victim model
        data (tuple): (point_clouds, labels), where
                      - point_clouds: torch.FloatTensor [B, N, 3]
                      - labels: torch.LongTensor [B]
        text_features (torch.FloatTensor, optional): not used here.
        budget (float): epsilon budget
        step_size (float): step size per iteration
        num_iter (int): number of iterations
        dist_metric (str): distance metric ('l2', 'linf', etc.)

    Returns:
        perturbed point clouds (torch.FloatTensor) [B, N, 3]
    """
    
    point_clouds, labels = data  # unpack
    point_clouds = point_clouds.cuda()
    labels = labels.cuda()


    num_classes = NumsClass
    # 如果没有target，需要自己随机生成一个
    target = random_target(labels, num_classes)

    # Define a dummy adversarial loss function
    adv_func = CrossEntropyAdvLoss()
    
    clip_func = ClipPointsL2(budget=budget)

    # Instantiate PGD attacker
    attacker = PGD(model, adv_func, clip_func, budget, step_size, num_iter, dist_metric,text_features)

    # Run attack
    perturbed_point_clouds, _ = attacker.attack(point_clouds, target)

    # Convert to torch tensor
    perturbed_point_clouds = torch.from_numpy(perturbed_point_clouds).cuda().float()

    return perturbed_point_clouds
def perturb_attack(model, data,text_features):
    point_clouds, labels = data  # unpack
    point_clouds = point_clouds.cuda()
    labels = labels.cuda()
    num_classes = NumsClass
    target = random_target(labels, num_classes)
    adv_func = CrossEntropyAdvLoss()
    dist_func = L2Dist()

    attacker = CWPerturb(model, adv_func, dist_func,
                         attack_lr=1e-2,
                         init_weight=10., max_weight=80.,
                         binary_step=10,
                         num_iter=50,
                         text_features=text_features)
    _, perturbed_point_clouds, _ = attacker.attack(point_clouds, target)
    perturbed_point_clouds = torch.from_numpy(perturbed_point_clouds).cuda().float()
    return perturbed_point_clouds
def drop_attack(model, data,text_features,dropnum = 100):
    point_clouds, labels = data  # unpack
    point_clouds = point_clouds.cuda()
    labels = labels.cuda()
    num_classes = NumsClass
    target = random_target(labels, num_classes)

    attacker =  SaliencyDrop(model, num_drop=dropnum,
                            alpha=1, k=5,text_features=text_features)
    perturbed_point_clouds, _ = attacker.attack(point_clouds, target)
    perturbed_point_clouds = torch.from_numpy(perturbed_point_clouds).cuda().float()
    return perturbed_point_clouds
def KNN_attack(model, data,text_features):
    point_clouds, labels = data  # unpack
    point_clouds = point_clouds.cuda()
    #point_clouds = fps6d(point_clouds, 1024)
    labels = labels.cuda()
    num_classes = NumsClass
    target = random_target(labels, num_classes)
    adv_func = CrossEntropyAdvLoss()
    dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=5., knn_weight=3.)
    
    clip_func = ProjectInnerClipLinf(budget=0.1)
    attacker = CWKNN(model, adv_func, dist_func, clip_func,
                     attack_lr = 1e-3,
                     num_iter=2500,
                     text_features=text_features)

    perturbed_point_clouds, _ = attacker.attack(point_clouds, target)
    perturbed_point_clouds = torch.from_numpy(perturbed_point_clouds).cuda().float()
    return perturbed_point_clouds
def ADD_attack(model, data,type = "CD",text_features = None):
    point_clouds, labels = data  # unpack
    point_clouds = point_clouds.cuda()
    #point_clouds = fps6d(point_clouds, 1024)
    labels = labels.cuda()
    num_classes = NumsClass
    target = random_target(labels, num_classes)

    adv_func = CrossEntropyAdvLoss()
    
    if type == 'CD':
        dist_func = ChamferDist(method='adv2ori')
        init_w = 5e3
        upper_w = 4e4
    else:
        dist_func = HausdorffDist(method='adv2ori')
        init_w = 2e2
        upper_w = 9e2
    attacker = CWAdd(model, adv_func, dist_func,
                     attack_lr=1e-2,
                     init_weight=init_w, max_weight=upper_w,
                     binary_step=10,
                     num_iter=500,
                     num_add=512,
                     text_features=text_features)

    _, perturbed_point_clouds, _ = attacker.attack(point_clouds, target)
    perturbed_point_clouds = torch.from_numpy(perturbed_point_clouds).cuda().float()
    return perturbed_point_clouds


def AOF_attack(model, data, text_features=None):
    point_clouds, labels = data  # unpack
    point_clouds = point_clouds.cuda()
    #point_clouds = fps6d(point_clouds, 1024)
    labels = labels.cuda()
    #num_classes = NumsClass
    #target = random_target(labels, num_classes)

    clip_func = ClipPointsLinf(budget=0.18)
    adv_func = UntargetedLogitsAdvLoss(kappa=30.)
    dist_func = L2Dist()
    attacker = CWAOF(model, adv_func, dist_func,
                         attack_lr=1e-2,
                         binary_step=2,
                         num_iter=200, GAMMA=0.25,
                         low_pass = 100,
                         clip_func=clip_func,
                         text_features=text_features)

    _, perturbed_point_clouds, _ = attacker.attack(point_clouds, labels)
    perturbed_point_clouds = torch.from_numpy(perturbed_point_clouds).cuda().float()
    return perturbed_point_clouds


def AdvPC_attack(model, data, text_features=None,ae_model_path=None):
    point_clouds, labels = data  # unpack
    point_clouds = point_clouds.cuda()
    #point_clouds = fps6d(point_clouds, 1024)
    labels = labels.cuda()
    num_classes = NumsClass
    target = random_target(labels, num_classes)
    import encoders_decoders
    ae_model = encoders_decoders.AutoEncoder(3)
    ae_state_dict = torch.load(ae_model_path)
    print('Loading ae weight {}'.format(ae_model_path))
    try:
        ae_model.load_state_dict(ae_state_dict)
    except RuntimeError:
        ae_state_dict = {k[7:]: v for k, v in ae_state_dict.items()}
        ae_model.load_state_dict(ae_state_dict)

    ae_model = ae_model.cuda()

    clip_func = ClipPointsLinf(budget=0.18)
    dist_func = ChamferDist()
    adv_func = CrossEntropyAdvLoss()
    attacker =  CWAdvPC(model, ae_model, adv_func, dist_func,
                         attack_lr=1e-2,
                         binary_step=2,
                         num_iter=200, GAMMA=0.25,
                         clip_func=clip_func,
                         text_features=text_features)
    _, perturbed_point_clouds, _ = attacker.attack(point_clouds, target,labels)
    perturbed_point_clouds = torch.from_numpy(perturbed_point_clouds).cuda().float()
    return perturbed_point_clouds



def test_attack_pgd(model, text_features=None, batch_text_features=None, prompt=None):
    if batch_text_features is None:
        batch_text_features = text_features
    
    model.eval()
    all_points = []
    orginal_points = []
    all_labels = []

    for data in test_loader:
        #points = fps6d(data[0].cuda(),1024)
        points =data[0].cuda()
        perturbed_data = pgd_attack(model, (points,data[1]), text_features=text_features, budget=0.08, step_size=0.08/20, num_iter=20)
        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])
        orginal_points.append(points.cpu())

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    orginal_points = torch.cat(orginal_points,dim=0).numpy()
    np.savez('test_pgd_attack_data.npz', points=all_points, labels=all_labels,orginal_points = orginal_points) #

def test_attack_pgd_train(model, text_features=None, batch_text_features=None, prompt=None):
    if batch_text_features is None:
        batch_text_features = text_features
    
    model.eval()
    all_points = []
    orginal_points = []
    all_labels = []

    for data in train_loader:
        #points = fps6d(data[0].cuda(),1024)
        points =data[0].cuda()
        perturbed_data = pgd_attack(model, (points,data[1]), text_features=text_features, budget=0.08, step_size=0.08/20, num_iter=20)
        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])
        orginal_points.append(points.cpu())

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    orginal_points = torch.cat(orginal_points,dim=0).numpy()
    np.savez('2048_train_attack_data.npz', points=all_points, labels=all_labels,orginal_points = orginal_points)


def test_attack_perturb(model, text_features=None, batch_text_features=None, prompt=None):
    if batch_text_features is None:
        batch_text_features = text_features

    model.eval()
    all_points = []
    all_labels = []

    for data in test_loader:
        perturbed_data = perturb_attack(model, data, text_features=text_features)
        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    np.savez('perturb_attack_data.npz', points=all_points, labels=all_labels)

def test_attack_drop(model, text_features=None, batch_text_features=None, prompt=None, dropnum=100):
    if batch_text_features is None:
        batch_text_features = text_features

    model.eval()
    all_points = []
    all_labels = []

    for data in test_loader:
        perturbed_data = drop_attack(model, data, text_features=text_features, dropnum=dropnum)
        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    np.savez(f'drop_{dropnum}_attack_data.npz', points=all_points, labels=all_labels)

def test_attack_KNN(model, text_features=None, batch_text_features=None, prompt=None):
    if batch_text_features is None:
        batch_text_features = text_features

    model.eval()
    all_points = []
    all_labels = []

    for data in test_loader:
        logger.info("begin one batch")
        perturbed_data = KNN_attack(model, data, text_features=text_features)
        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])
        logger.info("one batch done")

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    np.savez('KNN_attack_data.npz', points=all_points, labels=all_labels)

def test_attack_ADD(model, text_features=None, batch_text_features=None, prompt=None, type="CD"):
    if batch_text_features is None:
        batch_text_features = text_features

    model.eval()
    all_points = []
    all_labels = []

    for data in test_loader:
        logger.info("begin one batch")
        perturbed_data = ADD_attack(model, data, text_features=text_features, type=type)
        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])
        logger.info("one batch done")

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    np.savez(f'ADD_{type}_attack_data.npz', points=all_points, labels=all_labels)

def test_attack_AOF(model, text_features=None, batch_text_features=None, prompt=None):
    if batch_text_features is None:
        batch_text_features = text_features
    
    model.eval()
    all_points = []
    all_labels = []

    for data in test_loader:
        perturbed_data = AOF_attack(model, (data[0][:,:,:3],data[1]), text_features=text_features)
        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])
        logger.info("one batch done")

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    np.savez('SN_AOF_attack_data.npz', points=all_points, labels=all_labels) #

def test_attack_AdvPC(model, text_features=None, batch_text_features=None, prompt=None,ae_model_path = None):
    if batch_text_features is None:
        batch_text_features = text_features
    
    model.eval()
    all_points = []
    all_labels = []

    for data in test_loader:
        perturbed_data = AdvPC_attack(model, (data[0][:,:,:3],data[1]), text_features=text_features,ae_model_path = ae_model_path)
        all_points.append(perturbed_data.cpu())
        all_labels.append(data[1])
        logger.info("one batch done")

    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    np.savez('Scanobjectnn_AdvPC_attack_data.npz', points=all_points, labels=all_labels) #

class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.points = torch.tensor(data['points'], dtype=torch.float32)  # [N, K, 6] or [N, K, 3]
        self.labels = torch.tensor(data['labels'], dtype=torch.long)     # [N]
        if self.points.shape[2] == 3:
            # If points are [N, K, 3], we need to add a dummy feature dimension
            colors = torch.full(self.points.shape, 0.5)
            # 拼接在一起，沿着 dim=1（列方向）
            points_with_color = torch.cat([self.points, colors], dim=2)
            self.points = points_with_color  # [N, K, 6]
            #print(f"Loaded point clouds with shape {self.points.shape} and labels with shape {self.labels.shape}")
    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]
class IF_defense_NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.points = torch.tensor(data['test_pc'], dtype=torch.float32)  # [N, K, 6] or [N, K, 3]
        self.labels = torch.tensor(data['test_label'], dtype=torch.long)     # [N]
        if self.points.shape[2] == 3:
            # If points are [N, K, 3], we need to add a dummy feature dimension
            colors = torch.full(self.points.shape, 0.5)
            points_with_color = torch.cat([self.points, colors], dim=2)
            self.points = points_with_color  # [N, K, 6]
            #print(f"Loaded point clouds with shape {self.points.shape} and labels with shape {self.labels.shape}")
    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]
    
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

'''def tsne_visualize(model, test_loader, text_features=None, batch_text_features=None,
                   prompt=None, defense=None, num_batches=None, save_path='tsne_vis.png'):
    if batch_text_features is None:
        batch_text_features = text_features

    model.eval()
    all_feats = []
    all_labels = []
    batches_processed = 0

    for data in test_loader:
        if num_batches is not None:
            if batches_processed >= num_batches:
                break
        perturbed_data = data[0].cuda()

        if defense == 'SRS':
            perturbed_data = srs_sample(perturbed_data)
        elif defense == 'DUP':
            perturbed_data = fps6d(data[0].cuda(), 2048)
            xyz = perturbed_data[..., :3]
            feat = perturbed_data[..., 3:]
            defended_xyz = dupnet(xyz)
            B, K_new, _ = defended_xyz.shape
            repeated_feat = feat[:, :1, :].repeat(1, K_new, 1)
            perturbed_data = torch.cat([defended_xyz, repeated_feat], dim=-1)

        with torch.no_grad():
            B = perturbed_data.shape[0]
            if prompt is not None:
                prompt_now = prompt.expand(B, -1, -1)
                point_features = model.encode_pc(perturbed_data, prompt_now)
            else:
                point_features = model.encode_pc(perturbed_data)
            point_features = point_features / point_features.norm(dim=-1, keepdim=True)
            
            all_feats.append(point_features.cpu())
            all_labels.append(data[1])

        batches_processed += 1

    # 合并所有 batch 的特征和标签
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(all_feats)

    # 可视化
    plt.figure(figsize=(12, 10), dpi=300)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='tab20', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(40), label='Class Label')
    if prompt is not None:
        plt.title(f't-SNE Visualization MultiModal Robust Prompt')
    else:
        plt.title(f't-SNE Visualization Clean')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")
'''

def tsne_visualize(model, test_loader, text_features, prompt=None,
                          defense=None, num_batches=None, save_path='tsne_logits_vis.png'):
    model.eval()
    all_logits = []
    all_labels = []
    batches_processed = 0

    for data in test_loader:
        if num_batches is not None and batches_processed >= num_batches:
            break

        perturbed_data = data[0].cuda()

        if defense == 'SRS':
            perturbed_data = srs_sample(perturbed_data)
        elif defense == 'DUP':
            perturbed_data = fps6d(data[0].cuda(), 1024)
            perturbed_data = data[0].cuda()
            xyz = perturbed_data[..., :3]
            feat = perturbed_data[..., 3:]
            defended_xyz = dupnet(xyz)
            B, K_new, _ = defended_xyz.shape
            repeated_feat = feat[:, :1, :].repeat(1, K_new, 1)
            perturbed_data = torch.cat([defended_xyz, repeated_feat], dim=-1)

        with torch.no_grad():
            B = perturbed_data.shape[0]
            if prompt is not None:
                prompt_now = prompt.expand(B, -1, -1)
                point_features = model.encode_pc(perturbed_data, prompt_now)
            else:
                point_features = model.encode_pc(perturbed_data)

            point_features = point_features / point_features.norm(dim=-1, keepdim=True)
            logits = torch.matmul(point_features, text_features.t())  # 计算 logits

            all_logits.append(logits.cpu())
            all_labels.append(data[1])

        batches_processed += 1

    all_feats = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(all_feats)

    plt.figure(figsize=(12, 10), dpi=300)
    #scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='tab20', s=10, alpha=0.7)
    #scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='viridis', s=10, alpha=0.7)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='nipy_spectral', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(15), label='Class Label') #40
    if prompt is not None:
        plt.title('t-SNE Visualization w/ our defender')
    else:
        plt.title('t-SNE Visualization w/o defender')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE visualization (logits) saved to {save_path}")

class ScanObjectNNDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        # 加载数据
        with h5py.File(self.h5_path, 'r') as f:
            self.data = f['data'][:]           # shape: (N, 1024, 3)
            self.labels = f['label'][:]        # shape: (N, 1)

        self.labels = self.labels.squeeze()    # shape: (N,)
        self.classnames =  ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        points = self.data[idx]               # shape: (1024, 3)
        label = self.labels[idx]

        if self.transform:
            points = self.transform(points)

        return points, label
task = args.task#"ScanObjectNN"  # '''ScanObjectNN or 'Modelnet40
logger.info(f"Task: {task} ,Ifdefense Blackbox")
if __name__ == "__main__":
    logger.info("loading dataset")
    if task == 'Modelnet40':
        train_dataset = ModelNet40Dataset(root_dir='../data/modelnet40_normal_resampled', split='modelnet40_train', transform=transform)
        test_dataset = ModelNet40Dataset(root_dir='../data/modelnet40_normal_resampled', split='modelnet40_test', transform=transform)
        NumsClass = 40
    else:
        train_dataset = ScanObjectNNDataset(h5_path='xxxxx/3D/data/scanobjectnn/main_split_nobg/training_objectdataset_augmentedrot_scale75.h5', transform=transform)
        test_dataset = ScanObjectNNDataset(h5_path='xxxxx/3D/data/scanobjectnn/main_split_nobg/test_objectdataset_augmentedrot_scale75.h5', transform=transform)
        NumsClass = 15

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4) #32

    
    logger.info("dataset loaded")

    logger.info("Loading CLIP Text model and encoding classnames...")
    clip_model_text, _, preprocess = open_clip.create_model_and_transforms_text(
        'EVA02-E-14-plus', pretrained="xxxxx/3D/open_clip_pytorch_model.bin", device="cpu"
    )
    clip_model_text.cuda()#.to('cuda:0')
    clip_model_text.eval()

    with torch.no_grad():
        classnames_text = [f"X X X A depth picture of {name}." for name in train_dataset.classnames]
        text_inputs = open_clip.tokenize(classnames_text).to("cuda:0")
        text_features = clip_model_text.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print("Classnames encoded into text features.")
    text_features = text_features.cuda(0)
    #del clip_model_text
    #torch.cuda.empty_cache()


    logger.info("loading point cloud student model")
    args.pc_model = "eva02_tiny_patch14_224"
    args.pc_feat_dim = 192
    args.ckpt_path = "xxxxx/3D/model_ti.pt"

    model_student = create_uni3d(args)
    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path)
        sd = checkpoint['module']
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model_student.load_state_dict(sd)
        print(f"Model student loaded from {args.ckpt_path}")
    else:
        print(f"[Warning] Checkpoint not found at {args.ckpt_path}")
    #model_student = nn.DataParallel(model_student, device_ids=[0, 1])
    model_student.cuda()#.to('cuda:0')
    model_student.eval()
    
    
    if args.task == 'Modelnet40':
        #ckpt = 'xxxxx/3D/Uni3D-main/ckpts/hyber_training/MN40_mixadv_student_adv_multi_prompt_tau_loss_weight_10_5_best_prompt.pt'
        #ckpt = "xxxxx/3D/Uni3D-main/MN40_mixadv_student_adv_multi_prompt_tau_loss_weight_5_3_best_prompt.pt"
        #ckpt = 'xxxxx/3D/Uni3D-main/MN40_mixadv_student_adv_multi_prompt_tau_loss_weight_10_3_best_prompt.pt'
        #ckpt = 'xxxxx/3D/Uni3D-main/MN40_mixadv_student_adv_multi_prompt_tau_loss_weight_15_3_best_prompt.pt'
        #ckpt = 'xxxxx/3D/Uni3D-main/MN40_mixadv_student_adv_multi_prompt_tau_loss_weight_10_5_best_prompt.pt'
        #ckpt = 'xxxxx/3D/Uni3D-main/MN40_mixadv_student_adv_multi_prompt_tau_loss_weight_10_7_best_prompt.pt'
        #ckpt = 'xxxxx/3D/Uni3D-main/MN40_ablation_fixed_loss_10_3_best_prompt.pt'
        #ckpt = 'xxxxx/3D/Uni3D-main/MN40_ablation_no_img_10_3_best_prompt.pt'
        ckpt = args.prompt_ckpt
        n_ctx = 3
        logger.info(f"Loading point prompt from {ckpt}")
        print(f"Loading point prompt from {ckpt}")
        point_prompt = torch.load(ckpt)
        #ckpts/loss+weight+tau_5_10/multiAdaptiveLoss1024_best_prompt.pt
        ctx = torch.load(ckpt[:-3]+'_ctx.pt')
        #'ckpts/loss+weight+tau_5_10/multiAdaptiveLoss1024_best_prompt_ctx.pt'

        attack_datasets = {
            "clean": "xxxxx/3D/data/test_data_clean_1024.npz",

            "PGD whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/1024_test_pgd_attack_data.npz",
            "Perturb whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/1024_perturb_attack_data.npz",
            "KNN whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/1024_KNN_attack_data.npz",
            "ADD CD whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ADD_CD_attack_data.npz",
            "ADD HD whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ADD_HD_attack_data.npz",
            "Drop 200 whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/drop_200_attack_data.npz",
            "AdvPC whitebox":"xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/MN40_AdvPC_attack_data.npz",
            "AOF whitebox":"xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/MN40_AOF_attack_data.npz",

            "PGD blackbox":"xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/pgd_attack_data.npz",
            "Perturb blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/perturb_attack_data.npz",
            "KNN blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/KNN_attack_data.npz",
            "ADD CD blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ADD_CD_attack_data.npz",
            "ADD HD blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ADD_HD_attack_data.npz",
            "Drop 200 blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/drop_200_attack_data.npz",
            "AdvPC blackbox":"xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/MN40_AdvPC_attack_data.npz",
            "AOF blackbox":"xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/MN40_AOF_attack_data.npz",
        }
    else:
        #point_prompt = torch.load('xxxxx/3D/Uni3D-main/ckpts/scanobjectnn/multiAdaptiveLoss1024_best_prompt.pt')
        #'xxxxx/3D/Uni3D-main/ckpts/scanobjectnn/multiAdaptiveLoss1024_best_prompt.pt'
        #ctx = torch.load('xxxxx/3D/Uni3D-main/ckpts/scanobjectnn/multiAdaptiveLoss1024_best_prompt_ctx.pt')
        #'xxxxx/3D/Uni3D-main/ckpts/scanobjectnn/multiAdaptiveLoss1024_best_prompt_ctx.pt'
        #ckpt = 'xxxxx/3D/Uni3D-main/ckpts/hyber_training/ScanObject_mixadv_student_adv_multi_prompt_tau_loss_weight_10_5_best_prompt.pt'
        #'xxxxx/3D/Uni3D-main/SN_10_3_best_prompt.pt'
        ckpt = args.prompt_ckpt
        n_ctx = 3
        logger.info(f"Loading point prompt from {ckpt}")
        print(f"Loading point prompt from {ckpt}")
        point_prompt = torch.load(ckpt)
        ctx = torch.load(ckpt[:-3]+'_ctx.pt')
        attack_datasets = {
            "clean":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_2048_test_clean.npz",

            "PGD whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_2048_test_pgd_attack_data.npz",
            "Perturb whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_perturb_attack_data.npz",
            "KNN whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_KNN_attack_data.npz",
            "ADD CD whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_ADD_CD_attack_data.npz",
            "ADD HD whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_ADD_HD_attack_data.npz",
            "Drop 200 whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_drop_200_attack_data.npz",
            "AdvPC whitebox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/Scanobjectnn_AdvPC_attack_data.npz",
            "AOF whitebox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/SN_AOF_attack_data.npz",

            "PGD blackbox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_pgd_attack_data.npz",
            "Perturb blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_perturb_attack_data.npz",
            "KNN blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_KNN_attack_data.npz",
            "ADD CD blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_ADD_CD_attack_data.npz",
            "ADD HD blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_ADD_HD_attack_data.npz",
            "Drop 200 blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_drop_200_attack_data.npz",
            "AdvPC blackbox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_AdvPC_attack_data.npz",
            "AOF blackbox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/SN_AOF_attack_data.npz",
  
        }
    ctx = ctx.cuda()

    from utils.tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()

    classnames = train_dataset.classnames
    classnames = [name.replace("_", " ") for name in classnames]
    classnames_text = [f"A depth picture of {name}." for name in classnames]
    text = tokenizer(classnames_text).cuda()

    prompt_prefix = " ".join(["X"] * n_ctx)
    text_prompts = [prompt_prefix + " A depth picture of " + name + "." for name in classnames]
    tokenized_prompts = torch.stack([tokenizer(p) for p in text_prompts])
    tokenized_prompts = tokenized_prompts.cuda()
    prompted_text_features = clip_model_text.encode_text(text,ctx,tokenized_prompts)

    dupnet = DUPNet(sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4)
    dupnet.pu_net.load_state_dict(torch.load("DUP_Net/pu-in_1024-up_4.pth"))
    dupnet = dupnet.cuda()
    dupnet.eval()

    
    
    print("===========================tSNE==========================")
    #dataset = NpzPointCloudDataset('xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/1024_perturb_attack_data.npz')
    dataset = NpzPointCloudDataset("xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_perturb_attack_data.npz")#'adv_data/whitebox/perturb_attack_data.npz')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    print("test clean model")
    tsne_visualize(model_student, loader, text_features=text_features,prompt=None, num_batches=None, save_path='SN_tsne_vis.png')
    
    logger.info("test perturb done")
    tsne_visualize(model_student, loader, text_features=prompted_text_features,prompt=point_prompt, num_batches=None, save_path='SN_tsne_vis_prompt.png')
    raise RuntimeError("test done")

    def group_test(model_student, dataset, text_features, prompted_text_features, point_prompt,type):
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        print("==========================="+type+"==========================")
        print("test clean model")
        test_clean(model_student, loader, text_features,prompt=None,defense = None)
        '''print("test SRS")
        test_clean(model_student, loader, text_features,prompt=None,defense = 'SRS')
        print("test SOR")
        test_clean_with_list(model_student, loader, text_features,prompt=None)
        print("test DUP")
        test_clean(model_student, loader, text_features,prompt=None,defense = 'DUP')'''
        print("test Multi-Modal Prompt")
        test_clean(model_student, loader, text_features=prompted_text_features,prompt=point_prompt)
        print("test Multi-Modal Prompt with prompted text features")
        test_clean(model_student, loader, text_features=prompted_text_features,prompt=None)
        print("test Multi-Modal Prompt with point prompt")
        test_clean(model_student, loader, text_features=text_features,prompt=point_prompt)

    def test_IF_Defense(model_student, text_features):
        print("===========================IF Defense==========================")
        '''test_list = {
            'IF-defense clean': "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-test_data_clean_1024.npz",
            'IF-defense PGD whitebox': "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-1024_test_pgd_attack_data.npz",
            'IF-defense Perturb whitebox': "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-1024_perturb_attack_data.npz",
            'IF-defense KNN whitebox': "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-1024_KNN_attack_data.npz",
            'IF-defense ADD CD whitebox': "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-ADD_CD_attack_data.npz",
            'IF-defense ADD HD whitebox': "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-ADD_HD_attack_data.npz",
            'IF-defense Drop 100 whitebox': "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-drop_100_attack_data.npz",
            'IF-defense Drop 200 whitebox': "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-drop_200_attack_data.npz",
            "IF-Defense whitebox AOF": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-MN40_AOF_attack_data.npz",
            "IF-defense AdvPC":"xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ConvONet-Opt/convonet_opt-MN40_AdvPC_attack_data.npz"

                }'''
        MN40_blackbox = {
            "IF-defense KNN_attack": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-KNN_attack_data.npz",
            "IF-defense pgd_attack": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-pgd_attack_data.npz",
            "IF-defense drop_200_attack": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-drop_200_attack_data.npz",
            "IF-defense ADD_HD_attack": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-ADD_HD_attack_data.npz",
            "IF-defense perturb_attack": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-perturb_attack_data.npz",
            "IF-defense drop_100_attack": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-drop_100_attack_data.npz",
            "IF-defense ADD_CD_attack": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-ADD_CD_attack_data.npz",
            "IF-Defense Blackbox AOF": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-MN40_AOF_attack_data.npz",
            "IF-Defense Blackbox AdvPC": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ConvONet-Opt/convonet_opt-MN40_AdvPC_attack_data.npz"

        }

        SN_whitebox = {
            "IF-defense clean": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_2048_test_clean.npz",
            "IF-defense AdvPC_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-Scanobjectnn_AdvPC_attack_data.npz",
            "IF-defense ADD_CD_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_ADD_CD_attack_data.npz",
            "IF-defense AOF_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_AOF_attack_data copy.npz",
            "IF-defense perturb_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_perturb_attack_data.npz",
            "IF-defense pgd_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_2048_test_pgd_attack_data.npz",
            "IF-defense ADD_HD_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_ADD_HD_attack_data.npz",
            "IF-defense KNN_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_KNN_attack_data.npz",
            "IF-defense drop_100_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_drop_100_attack_data.npz",
            "IF-defense drop_200_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_drop_200_attack_data.npz",
            "IF-defense AOF_attack_2": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-scanobjectnn_AOF_attack_data.npz"
        }

        SN_blackbox = {
            "IF-defense pgd_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobjectnn_pgd_attack_data.npz",
            "IF-defense ADD_CD_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobjectnn_ADD_CD_attack_data.npz",
            "IF-defense perturb_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobjectnn_perturb_attack_data.npz",
            "IF-defense ADD_HD_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobjectnn_ADD_HD_attack_data.npz",
            "IF-defense KNN_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobjectnn_KNN_attack_data.npz",
            "IF-defense drop_100_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobjectnn_drop_100_attack_data.npz",
            "IF-defense drop_200_attack": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobjectnn_drop_200_attack_data.npz",
            "IF-Defense Blackbox AOF": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobkectnn_AOF_attack_data.npz",
            "IF-Defense Blackbox AdvPC": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-scanobjectnn_AdvPC_attack_data.npz"
 
        }

        single_test = {
            "IF-Defense Blackbox AOF": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/ConvONet-Opt/convonet_opt-SN_AOF_attack_data.npz",
            "IF-Defense whitebox AOF": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/ConvONet-Opt/convonet_opt-SN_AOF_attack_data.npz"
        }
            
        for type,path in single_test.items():
            dataset = IF_defense_NpzPointCloudDataset(path)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            print("test "+type)
            test_clean(model_student, loader, text_features,prompt=None,defense = None)

    '''logger.info("test data generation")
    test_clean(model_student, test_loader, text_features,prompt=None,defense = None)
    test_attack_pgd(model_student, text_features) #need repair to remove clean data which only used in training stage
    test_attack_pgd_train(model_student, text_features)'''
    
    #1024 robust training data generation
    #test_attack_pgd(model_student, text_features)
    #test_attack_pgd_train(model_student, text_features)

    '''logger.info("test attack perturb")
    test_attack_perturb(model_student, text_features)

    logger.info("test attack drop 100")
    test_attack_drop(model_student, text_features, dropnum =100)
    logger.info("test attack drop 200")
    test_attack_drop(model_student, text_features, dropnum =200)'''

    #logger.info("test attack KNN")
    #test_attack_KNN(model_student, text_features)
    
    '''logger.info("test attack ADD CD")
    test_attack_ADD(model_student, text_features, type = "CD")

    logger.info("test attack ADD HD")
    test_attack_ADD(model_student, text_features, type = "HD")'''
    
    
    

    
    '''
        "clean": "xxxxx/3D/data/test_data_clean_1024.npz",

        "PGD whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/1024_test_pgd_attack_data.npz",
        "Perturb whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/1024_perturb_attack_data.npz",
        "KNN whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/1024_KNN_attack_data.npz",
        "ADD CD whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ADD_CD_attack_data.npz",
        "ADD HD whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/ADD_HD_attack_data.npz",
        "Drop 200 whitebox": "xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/drop_200_attack_data.npz",
        "AdvPC whitebox":"xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/MN40_AdvPC_attack_data.npz",
        "AOF whitebox":"xxxxx/3D/Uni3D-main/adv_data/1024points/whiltebox/MN40_AOF_attack_data.npz",

        "PGD blackbox":"xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/pgd_attack_data.npz",
        "Perturb blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/perturb_attack_data.npz",
        "KNN blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/KNN_attack_data.npz",
        "ADD CD blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ADD_CD_attack_data.npz",
        "ADD HD blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/ADD_HD_attack_data.npz",
        "Drop 200 blackbox": "xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/drop_200_attack_data.npz",
        "AdvPC blacknox":"xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/MN40_AdvPC_attack_data.npz",
        "AOF blackbox":"xxxxx/3D/Uni3D-main/adv_data/1024points/blackbox/MN40_AOF_attack_data.npz",
    '''
    '''"clean":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_2048_test_clean.npz",

        "PGD whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_2048_test_pgd_attack_data.npz",
        "Perturb whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_perturb_attack_data.npz",
        "KNN whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_KNN_attack_data.npz",
        "ADD CD whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_ADD_CD_attack_data.npz",
        "ADD HD whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_ADD_HD_attack_data.npz",
        "Drop 200 whitebox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_drop_200_attack_data.npz",
        "AdvPC whitebox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/Scanobjectnn_AdvPC_attack_data.npz",
        "AOF whitebox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/SN_AOF_attack_data.npz",

        "PGD blackbox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_pgd_attack_data.npz",
        "Perturb blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_perturb_attack_data.npz",
        "KNN blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_KNN_attack_data.npz",
        "ADD CD blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_ADD_CD_attack_data.npz",
        "ADD HD blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_ADD_HD_attack_data.npz",
        "Drop 200 blackbox": "xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_drop_200_attack_data.npz",
        "AdvPC blacknox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/scanobjectnn_AdvPC_attack_data.npz",
        "AOF blackbox":"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/blackbox/SN_AOF_attack_data.npz",
  
    }'''
    
    
    for attack_type, path in attack_datasets.items():
        dataset = NpzPointCloudDataset(path)
        group_test(model_student, dataset, text_features, prompted_text_features, point_prompt, type=attack_type)
    
    
    

    test_IF_Defense(model_student, text_features)
   
    
    

    logger.info("test data generation")
    
    dataset = NpzPointCloudDataset("xxxxx/3D/data/scanobjectnn_2048_test_pgd_attack_data.npz")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    test_clean(model_student, loader, text_features,prompt=None,defense = None)
    
    dataset = NpzPointCloudDataset("xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_2048_test_clean.npz") #"xxxxx/3D/data/test_data_clean_1024.npz"xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_2048_test_clean.npz
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    logger.info("test attack AOF")
    test_attack_AOF(model_student, text_features)
    
    #PGD Done test_attack_pgd(model_student, text_features) #need repair to remove clean data which only used in training stage
    dataset = NpzPointCloudDataset('xxxxx/3D/data/test_data_clean_1024.npz')
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    logger.info("test attack AdvPC")
    dataset = NpzPointCloudDataset('xxxxx/3D/Uni3D-main/adv_data/scanobjectnn/whitebox/scanobjectnn_2048_test_clean.npz')
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    test_attack_AdvPC(model_student, text_features,ae_model_path='xxxxx/3D/IF_defense/logs/scanobjectnn/AE/2025-06-23 20:36:51_1024/BEST_model350_CD_0.0043.pth')
    



    logger.info("test attack ADD HD")
    test_attack_ADD(model_student, text_features, type = "HD")
    
    #1024 robust training data generation
   
    test_attack_pgd(model_student, text_features)
    #test_attack_pgd_train(model_student, text_features)

    logger.info("test attack perturb")
    test_attack_perturb(model_student, text_features)

    logger.info("test attack drop 100")
    test_attack_drop(model_student, text_features, dropnum =100)
    logger.info("test attack drop 200")
    test_attack_drop(model_student, text_features, dropnum =200)

    logger.info("test attack KNN")
    test_attack_KNN(model_student, text_features)

    logger.info("test attack ADD CD")
    test_attack_ADD(model_student, text_features, type = "CD")
    logger.info("test attack ADD HD")
    test_attack_ADD(model_student, text_features, type = "HD")

    
    
    
    
    
    
    
   
    

    #logger.info("test attack pgd")
    #test_attack_pgd(model_student, text_features)

    #logger.info("test attack perturb")
    #test_attack_perturb(model_student, text_features)
    #logger.info("test attack KNN")
    #test_attack_KNN(model_student, text_features)

    #logger.info("test attack drop 1000")
    #test_attack_drop(model_student, text_features, dropnum =1000)
    #logger.info("test attack drop 2000")
    #test_attack_drop(model_student, text_features, dropnum =2000)
    
    #logger.info("test attack ADD CD")
    #test_attack_ADD(model_student, text_features, type = "CD")
    #logger.info("test attack ADD HD")
    #test_attack_ADD(model_student, text_features, type = "HD")
       
    