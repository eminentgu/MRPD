import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
task = "MN_s"
#self.min_log_var = -0.8

import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.uni3d import create_uni3d
import utils.open_clip as open_clip
from loguru import logger
import psutil
import gc
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from utils.projection import process_point_cloud_batch
from utils.tokenizer import SimpleTokenizer
from models.point_encoder import fps6d
import h5py
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

logger.info(f"Task: {task}")

class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path, classnames):
        self.data = np.load(npz_path, mmap_mode='r') 
        self.points = self.data['points']            
        self.labels = self.data['labels']
        self.orginal_points = self.data['orginal_points']
        self.classnames = classnames

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = torch.tensor(self.points[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        original = torch.tensor(self.orginal_points[idx], dtype=torch.torch.float32)
        return original, label, point



tokenizer = SimpleTokenizer()
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

def contrastive_loss(point_features, text_features, tau=0.07):
    B = point_features.size(0)
    logits = torch.matmul(point_features, text_features.t()) / tau
    labels = torch.arange(B, device=point_features.device, dtype=torch.long)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.t(), labels)
    return (loss_p2t + loss_t2p) * 0.5


def selective_contrastive_loss(anchor_features, reference_features, reference_labels, ground_truth_labels, tau=0.07, topk=1):
    """
    Selectively compute contrastive loss between anchor and reference features,
    only for those samples where reference features correctly match the ground truth
    within top-k predictions.

    Args:
        anchor_features: [B, D] - features to be aligned (e.g., from student)
        reference_features: [B, D] - reference features (e.g., teacher)
        reference_labels: [B, D] - label embeddings (e.g., text features)
        ground_truth_labels: [B] - true labels, shape [B]
        tau: temperature
        topk: int, top-k filtering threshold (default: 1)

    Returns:
        loss: scalar contrastive loss
    """
    with torch.no_grad():
        # Compute logits between reference features and label embeddings
        logits_ref = torch.matmul(reference_features, reference_labels.t()) / tau  # [B, C]
        topk_preds = logits_ref.topk(k=topk, dim=1).indices  # [B, k]

        # Check if ground truth is within top-k
        ground_truth_labels = ground_truth_labels.view(-1, 1)  # [B, 1]
        correct_mask = (topk_preds == ground_truth_labels).any(dim=1)  # [B]

        if correct_mask.sum() == 0:
            return torch.tensor(0.0, device=anchor_features.device, requires_grad=True)

    # Filter valid samples
    anchor_selected = anchor_features[correct_mask]
    reference_selected = reference_features[correct_mask]

    # Compute contrastive loss
    B_sel = anchor_selected.size(0)
    logits = torch.matmul(anchor_selected, reference_selected.t()) / tau
    labels_sel = torch.arange(B_sel, device=anchor_features.device)

    loss_a2r = F.cross_entropy(logits, labels_sel)
    loss_r2a = F.cross_entropy(logits.t(), labels_sel)

    return 0.5 * (loss_a2r + loss_r2a)



from attacker import pgd_attack
def test_attack(model, epsilon, text_features=None,batch_text_features= None, prompt=None):
    if(batch_text_features is not None):
        #shape
        logger.info(f"batch_text_features shape: {batch_text_features.shape}")
        logger.info(f"text_features shape: {text_features.shape}")
    else:
        batch_text_features = text_features
    correct = 0
    total = 0
    model.eval()
    save = True
    for data in test_loader:
        logger.info("generate perturbed data")
        #perturbed_data = fgsm_attack(model, data, epsilon=epsilon)
        #perturbed_data = pgd_attack(model, data, epsilon=1.0, alpha=0.2, num_iter=10, text_features=text_features)
        perturbed_data =  data[2].cuda(0) #pgd_attack(model, data, text_features=text_features, budget=0.08, step_size=0.08/20, num_iter=20)
        if(save):
            first_point_cloud = perturbed_data[0].cpu().detach().numpy()  # 取 batch 0，转 numpy
            np.save('first_perturbed_pointcloud.npy', first_point_cloud)
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
            correct += (predictions == data[1].to('cuda:0')).sum().item()
            total += data[1].size(0)
    accuracy = correct / total
    print(f"Adversarial Test Accuracy: {accuracy:.4f}")
    del perturbed_data, point_features, batch_text_features, predictions
    torch.cuda.empty_cache()
    gc.collect()
    return accuracy


class PointPrompt(nn.Module):
    def __init__(self, 
                prompt_len=10, 
                feat_dim=192,
                student_model=None,
                device='cuda:0',
                ckpt_path=None,
                text_prompt_size=3,
                classnames=None,
                clip_model_text=None,):
        super().__init__() 
        self.log_tau = nn.Parameter(torch.tensor([0.07,0.07,0.07]).log())

        #prepare text learnable token
        n_ctx = text_prompt_size #text_prompt_size 
        ctx_dim = 1280 
        ctx_vectors = torch.empty(n_ctx, ctx_dim,device=device)
        nn.init.normal_(ctx_vectors, std=0.05) #0.02
        self.ctx = nn.Parameter(ctx_vectors,requires_grad=True) 
        self.min_log_var = -0.8
        self.log_vars = nn.Parameter(torch.zeros(3))  # log(σ^2) for [image, point, text]
        self.log_vars_clamped = None
        #self.ctx = self.ctx.to(device)
        
        classnames = [name.replace("_", " ") for name in classnames]
    
        classnames_text = [f"A depth picture of {name}." for name in classnames]
        self.text = tokenizer(classnames_text).to("cuda:0")

        #prepare original text prompt
        
        prompt_prefix = " ".join(["X"] * n_ctx)
        text_prompts = [prompt_prefix + " A depth picture of " + name + "." for name in classnames]
        
        tokenized_prompts = torch.stack([tokenizer(p) for p in text_prompts])
        self.tokenized_prompts = tokenized_prompts 
        self.tokenized_prompts = self.tokenized_prompts.to(device)

        self.clip_model_text = clip_model_text
        self.clip_model_text.to(device)

        if ckpt_path is not None:
            self.prompt = torch.load(ckpt_path)
            self.prompt.requires_grad = True
        else:
            self.prompt = nn.Parameter(torch.randn(1, prompt_len, feat_dim), requires_grad=True)
        self.prompt.to(device)

        self.student_model = student_model
        self.student_model.to(device)
        self.student_model.eval()

        for param in self.student_model.parameters():
            param.requires_grad = False
        for param in self.clip_model_text.parameters():
            param.requires_grad = False
    def reset_weights(self):
        self.log_vars = nn.Parameter(torch.zeros(3)) 
    def get_prompt(self):
        return self.prompt.detach()
    def get_text_feat(self):
        return self.clip_model_text.encode_text(self.text,self.ctx,self.tokenized_prompts)
    
    def get_multi_task_loss(self, loss_image, loss_point, loss_text):
        '''loss = (
            torch.exp(-self.log_vars[0]) * loss_image + self.log_vars[0] +
            torch.exp(-self.log_vars[1]) * loss_point + self.log_vars[1] +
            torch.exp(-self.log_vars[2]) * loss_text + self.log_vars[2]
        )'''
        log_vars_clamped = torch.clamp(self.log_vars, min=self.min_log_var)
        
        loss = (
            torch.exp(-log_vars_clamped[0]) * loss_image + log_vars_clamped[0] +
            torch.exp(-log_vars_clamped[1]) * loss_point + log_vars_clamped[1] +
            torch.exp(-log_vars_clamped[2]) * loss_text + log_vars_clamped[2]
        )
        self.log_vars_clamped = log_vars_clamped
        return loss

    def save_prompt(self, path):
        torch.save(self.prompt, path)
        torch.save(self.ctx, path.replace(".pt", "_ctx.pt"))
    
    def forward(self, point_clouds):
        B = point_clouds.shape[0]
        prompt = self.prompt.expand(B, -1, -1)
        text_features = self.clip_model_text.encode_text(self.text,self.ctx,self.tokenized_prompts)#.to("cpu").to("cpu").to("cpu")

        return self.student_model.encode_pc(point_clouds, prompt),text_features

parser = argparse.ArgumentParser(description="Uni3D Model Testing")
parser.add_argument("--pc_model", type=str, default="eva02_base_patch14_448", help="Point cloud model name")
parser.add_argument("--pc_feat_dim", type=int, default=768, help="Point cloud feature dimension")
parser.add_argument("--pretrained_pc", type=str, default=None, help="Path to pretrained point cloud model")
parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Drop path rate")
parser.add_argument("--pc_encoder_dim", type=int, default=512, help="Point cloud encoder dimension")
parser.add_argument("--embed_dim", type=int, default=1024, help="Embedding dimension")
parser.add_argument("--ckpt_path", type=str, default="xxxxxx/3D/model.pt", help="Path to model checkpoint")
parser.add_argument("--num_group", type=int, default=512, help="Number of groups for point cloud processing")
parser.add_argument("--group_size", type=int, default=64, help="Group size for point cloud processing")
args = parser.parse_args()

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
if __name__ == "__main__":
    logger.info("loading dataset")
    
    dataname = "Modelnet40" #scanobjectnnModelnet40
    if dataname == "Modelnet40":
        orgin_train_dataset = ModelNet40Dataset(root_dir='../data/modelnet40_normal_resampled', split='modelnet40_train', transform=transform)
        #orginal_test_dataset = ModelNet40Dataset(root_dir='../data/modelnet40_normal_resampled', split='modelnet40_test', transform=transform)

        train_dataset = NpzPointCloudDataset('xxxxxx/3D/data/train_pgd_attack_data.npz',classnames=orgin_train_dataset.classnames)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
        
        test_dataset = NpzPointCloudDataset('xxxxxx/3D/data/test_pgd_attack_data.npz',classnames=orgin_train_dataset.classnames)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
    else:
        orgin_train_dataset = ScanObjectNNDataset(h5_path='xxxxxx/3D/data/scanobjectnn/main_split_nobg/training_objectdataset_augmentedrot_scale75.h5', transform=transform)
        #orgin_train_datasettest_dataset = ScanObjectNNDataset(h5_path='xxxxxx/3D/data/scanobjectnn/main_split_nobg/test_objectdataset_augmentedrot_scale75.h5', transform=transform)
        train_dataset = NpzPointCloudDataset('xxxxxx/3D/data/scanobjectnn_2048_train_attack_data.npz',classnames=orgin_train_dataset.classnames)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
        test_dataset = NpzPointCloudDataset('xxxxxx/3D/data/scanobjectnn_2048_test_pgd_attack_data.npz',classnames=orgin_train_dataset.classnames)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
        

        
        
    logger.info("dataset loaded")

    logger.info("Loading CLIP Text model and encoding classnames...")
    clip_model_text, _, preprocess = open_clip.create_model_and_transforms_text(
        'EVA02-E-14-plus', pretrained="xxxxxx/3D/open_clip_pytorch_model.bin", device="cpu"
    )
    clip_model_text.to('cuda:0')
    clip_model_text.eval()

    with torch.no_grad():
        classnames_text = [f"X X X A depth picture of {name}." for name in train_dataset.classnames]
        text_inputs = tokenizer(classnames_text).to("cuda:0") #open_clip.
        text_features = clip_model_text.encode_text(text_inputs)

        x_to_save = text_features.detach().cpu().numpy()
        #np.savetxt('raw_feat.txt', x_to_save.reshape(-1, x_to_save.shape[-1]), fmt='%.6f')
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print("Classnames encoded into text features.")
    text_features = text_features.cuda(0)
    #print_memory("Before Releasing CLIP/text_features")
    clip_model_text.to('cuda:0')
    gc.collect()
    #print_memory("After Releasing CLIP/text_features")
    logger.info("CLIP Text feature space inited.")

    logger.info("Loading CLIP visual teacher")
    clip_model_visual, _, preprocess = open_clip.create_model_and_transforms_visual(
        'EVA02-E-14-plus', pretrained="xxxxxx/3D/open_clip_pytorch_model.bin", device="cpu"
    )
    print_memory("Before Releasing CLIP/text_features")
    clip_model_visual = clip_model_visual.to('cuda:1')
    clip_model_visual.eval()
    gc.collect()
    print_memory("After Releasing CLIP/text_features")
    logger.info("CLIP visual teacher inited.")

    logger.info("loading point cloud teacher model")
    point_teacher = create_uni3d(args)
    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path)
        sd = checkpoint['module']
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        point_teacher.load_state_dict(sd)
        point_teacher.eval()
        print(f"Model clean loaded from {args.ckpt_path}")
    else:
        print(f"[Warning] Checkpoint not found at {args.ckpt_path}")
    point_teacher.to('cuda:0')
    logger.info("point cloud teacher model loaded.")

    logger.info("loading point cloud student model")
    args.pc_model = "eva02_tiny_patch14_224"
    args.pc_feat_dim = 192
    args.ckpt_path = "xxxxxx/3D/model_ti.pt"

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
    model_student.to('cuda:0')

    test_accuracy = test_attack(model_student, 0.05, text_features, prompt=None)
    logger.info(f"before prompt Model student clean test accuracy: {test_accuracy:.4f}")
    model_student.eval()
    logger.info("point cloud student model loaded.")

    logger.info("loading point prompt")
    student_with_prompt = PointPrompt(
        prompt_len=10,
        feat_dim=args.pc_feat_dim,
        student_model=model_student,
        device='cuda:0',
        classnames=train_dataset.classnames,
        text_prompt_size=3,
        clip_model_text=clip_model_text,
    )
    student_with_prompt.to('cuda:0')
    logger.info("point prompt loaded.")

    logger.info("begin training")
    optimizer = torch.optim.AdamW(student_with_prompt.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-5)
    '''logger.info("before training")
    batch_text_features = student_with_prompt.get_text_feat()
    batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
    best_accuracy = test_attack(model_student, 0.05, text_features, batch_text_features, prompt=student_with_prompt.get_prompt())'''
    best_accuracy = 0
    for i in range(100):
        total_loss = 0
        for data in train_loader:
            #points, labels = data[0], data[1]
            #perturbed_data = points.to('cuda:0')
            #points = fps6d(points,1024)
            labels = data[1]
            labels = labels.to('cuda:0')
            #perturbed_data = fgsm_attack(model_student, [points, labels], epsilon=0.05)
            #perturbed_data = pgd_attack(model_student,data, text_features=text_features, budget=0.08, step_size=0.08/20, num_iter=20)
            #points = data[2].to('cuda:0')
            lambda_prob = 1
            points = data[0] if i < 50 else data[2]#if random.random() < lambda_prob else data[2]
            '''if i == 50:
                student_with_prompt.reset_weights()'''
            points = points.to('cuda:0')


            imgs = process_point_cloud_batch(points[:, :, :3].detach().cpu())
            with torch.no_grad():
                image_features = clip_model_visual.encode_image(imgs.to('cuda:1')).detach().to('cuda:0')
                image_features /= image_features.norm(dim=-1, keepdim=True)
                point_features = point_teacher.encode_pc(points)
                point_features /= point_features.norm(dim=-1, keepdim=True)
            #batch_text_features = text_features[labels]
            #batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)

            optimizer.zero_grad()
            student_features,batch_text_features = student_with_prompt(points)
           

            student_features = student_features / student_features.norm(dim=-1, keepdim=True)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
            batch_text_features = batch_text_features[labels]

            if 1:
                #loss_image = contrastive_loss(student_features, image_features)
                #loss_point = contrastive_loss(student_features, point_features)
                loss_image = selective_contrastive_loss(student_features, image_features, batch_text_features,labels,tau = student_with_prompt.log_tau[0].exp())
                loss_point = selective_contrastive_loss(student_features, point_features, batch_text_features,labels,tau = student_with_prompt.log_tau[1].exp())

                loss_text = contrastive_loss(student_features, batch_text_features,student_with_prompt.log_tau[2].exp()) #+ F.cross_entropy(torch.matmul(student_features, batch_text_features.T) /1, labels) #

                #loss = 0.1 * loss_image + 0.3 * loss_text + 0.6 * loss_point
                loss = student_with_prompt.get_multi_task_loss(loss_image, loss_point, loss_text)

                total_loss += loss.item()
            else: #ablation
                loss_image = contrastive_loss(student_features, image_features)
                loss_point = contrastive_loss(student_features, point_features)
                loss_text = contrastive_loss(student_features, batch_text_features,student_with_prompt.log_tau[2].exp())
                loss = student_with_prompt.get_multi_task_loss(loss_image, loss_point, loss_text)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_with_prompt.parameters(), max_norm=5.0)
            optimizer.step()
        if 1 :
            sigmas = torch.exp(student_with_prompt.log_vars_clamped / 2).detach().cpu().numpy()
            logger.info(f"Learned sigmas: image={sigmas[0]:.4f}, point={sigmas[1]:.4f}, text={sigmas[2]:.4f}")

        scheduler.step()
        logger.info(f"Epoch {i}: Loss {total_loss:.4f}")
        print(f"Epoch {i}: Loss {total_loss:.4f}")
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']:.6f}")

        
        batch_text_features = student_with_prompt.get_text_feat()
        batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
        accuracy = test_attack(model_student, 0.05, text_features, batch_text_features, prompt=student_with_prompt.get_prompt())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            student_with_prompt.save_prompt(task+"_best_prompt.pt")
            print(f"Best prompt saved with accuracy: {best_accuracy:.4f}")
        student_with_prompt.save_prompt(task + "_latest_prompt.pt")
        del image_features, point_features, student_features, batch_text_features, points, labels #perturbed_data, 
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[Epoch {i}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | Max: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        #student_acc = test_attack(model_student, 0.05, text_features, prompt=None)
        #logger.info(f"just want to make sure student is unchanged: {student_acc:.4f}")