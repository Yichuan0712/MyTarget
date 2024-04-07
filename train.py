import torch

torch.manual_seed(0)
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import yaml
import numpy as np
# import torchmetrics
from time import time
from model import *
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, recall_score, precision_score, \
    f1_score
import pandas as pd
import sys

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from loss import SupConHardLoss
from utils import prepare_tensorboard
from data_clean import prepare_dataloaders as prepare_dataloader_clean
from data_batchsample import prepare_dataloaders as prepare_dataloader_batchsample


def loss_fix(id_frag, motif_logits, target_frag, tools):
    # id_frag [batch]
    # motif_logits [batch, num_clas, seq]
    # target_frag [batch, num_clas, seq]
    fixed_loss = 0
    for i in range(len(id_frag)):
        frag_ind = id_frag[i].split('@')[1]
        target_thylakoid = target_frag[i, -1]  # -1 for Thylakoid, [seq]; -2 for chloroplast
        # label_first = target_thylakoid[0] # 1 or 0
        target_chlo = target_frag[i, -2]
        if frag_ind == '0' and torch.max(target_chlo) == 0 and torch.max(target_thylakoid) == 1:
            # print("case2")
            l = torch.where(target_thylakoid == 1)[0][0]
            true_chlo = target_frag[i, -2, :(l - 1)] == 1
            false_chlo = target_frag[i, -2, :(l - 1)] == 0
            motif_logits[i, -2, :(l - 1)][true_chlo] = 100
            motif_logits[i, -2, :(l - 1)][false_chlo] = -100
    # return fixed_loss
    # return target_frag
    return motif_logits, target_frag


def make_buffer(id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple):
    id_frags_list = []
    seq_frag_list = []
    target_frag_list = []
    for i in range(len(id_frag_list_tuple)):
        id_frags_list.extend(id_frag_list_tuple[i])
        seq_frag_list.extend(seq_frag_list_tuple[i])
        target_frag_list.extend(target_frag_nplist_tuple[i])
    seq_frag_tuple = tuple(seq_frag_list)
    target_frag_pt = torch.from_numpy(np.stack(target_frag_list, axis=0))
    type_protein_pt = torch.stack(list(type_protein_pt_tuple), axis=0)
    return id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt


def train_loop(tools, configs, warm_starting, train_writer):

    global global_step

    # 0404改的, 我怀疑scaler有问题
    # tools["optimizer"].zero_grad()
    # scaler = GradScaler()
    tools['net'].train().to(tools['train_device'])
    size = len(tools['train_loader'].dataset)
    num_batches = len(tools['train_loader'])
    train_loss = 0
    num_train_samples = len(tools['train_loader'])  # steps per epoch


    for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple,
                sample_weight_tuple, pos_neg) in enumerate(tools['train_loader']):
        tools["optimizer"].zero_grad()

        b_size = len(id_tuple)
        flag_batch_extension = False
        if (configs.supcon.apply and not warm_starting and pos_neg is not None) or \
                (configs.supcon.apply and warm_starting):
            """
            For two scenarios (CASE B & C, see Encoder::forward()) where the batch needs to be extended, 
            extend the 6 tuples with pos_neg
            0 - id_tuple, 
            1 - id_frag_list_tuple, 
            2 - seq_frag_list_tuple, 
            3 - target_frag_nplist_tuple, 
            4 - type_protein_pt_tuple, 
            5 - and sample_weight_tuple
            without the extending, each len(tuple) == batch_size
            after extending, len(tuple) == batch_size * (1 + n_pos + n_neg)
            """
            # print("len of pos_neg = "+str(len(pos_neg)))
            flag_batch_extension = True
            pos_transformed = [[[] for _ in range(6)] for _ in range(configs.supcon.n_pos)]
            neg_transformed = [[[] for _ in range(6)] for _ in range(configs.supcon.n_neg)]
            for i in range(b_size):
                # print("pos_neg pos")
                # print(len(pos_neg[i][0]))
                for j in range(configs.supcon.n_pos):
                    for k in range(6):
                        pos_transformed[j][k].append(pos_neg[i][0][j][k])
            # print(len(id_tuple))
            for j in range(configs.supcon.n_pos):
                id_tuple += tuple(pos_transformed[j][0])
                id_frag_list_tuple += tuple(pos_transformed[j][1])
                seq_frag_list_tuple += tuple(pos_transformed[j][2])
                target_frag_nplist_tuple += tuple(pos_transformed[j][3])
                type_protein_pt_tuple += tuple(torch.from_numpy(arr) for arr in pos_transformed[j][4])
                sample_weight_tuple += tuple(pos_transformed[j][5])
            # print(len(id_tuple))
            for i in range(b_size):
                # print("pos_neg neg")
                # print(len(pos_neg[i][1]))
                for j in range(configs.supcon.n_neg):
                    for k in range(6):
                        neg_transformed[j][k].append(pos_neg[i][1][j][k])
            for j in range(configs.supcon.n_neg):
                id_tuple += tuple(neg_transformed[j][0])
                id_frag_list_tuple += tuple(neg_transformed[j][1])
                seq_frag_list_tuple += tuple(neg_transformed[j][2])
                target_frag_nplist_tuple += tuple(neg_transformed[j][3])
                type_protein_pt_tuple += tuple(torch.from_numpy(arr) for arr in neg_transformed[j][4])
                sample_weight_tuple += tuple(neg_transformed[j][5])
            # print(len(id_tuple))

        id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple,
                                                                                     seq_frag_list_tuple,
                                                                                     target_frag_nplist_tuple,
                                                                                     type_protein_pt_tuple)
        # with autocast():
        if True: # 难道是?

            encoded_seq = None

            protein_embeddings = torch.load('5283_esm2_t33_650M_UR50D.pt')
            emb_pro_list = []
            for i in id_tuple:
                emb_pro_list.append(protein_embeddings[i])
            emb_pro = torch.stack(emb_pro_list, dim=0)
            emb_pro_ = emb_pro.view((configs.train_settings.batch_size, 1 + configs.supcon.n_pos + configs.supcon.n_neg, -1))
            projection_head = tools['net'](emb_pro_)


            if configs.supcon.apply and configs.supcon.apply_supcon_loss:  # configs.supcon.apply: # and warm_starting: calculate supcon loss no matter whether warm_starting or not.
                supcon_loss = tools['loss_function_supcon'](
                    projection_head,
                    configs.supcon.temperature,
                    configs.supcon.n_pos)
                # weighted_supcon_loss = configs.supcon.weight * supcon_loss
                # print(f"{global_step} supcon_loss:{weighted_supcon_loss.item()}")
                print(f"{global_step} supcon_loss:{supcon_loss.item()}")
                # train_writer.add_scalar('step supcon_loss', weighted_supcon_loss.item(), global_step=global_step)
                train_writer.add_scalar('step supcon_loss', supcon_loss.item(), global_step=global_step)
                # weighted_loss_sum += weighted_supcon_loss
            if configs.supcon.apply is False and warm_starting:
                raise ValueError("Check configs.supcon.apply and configs.supcon.warm_start")

            supcon_loss.backward()
            tools['optimizer'].step()
            train_loss += supcon_loss.item()

        global_step += 1

    epoch_loss = train_loss / num_batches
    return epoch_loss


def test_loop(tools, dataloader, train_writer, valid_writer):
    customlog(tools["logfilepath"], f'number of test steps per epoch: {len(dataloader)}\n')
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    tools['net'].eval().to(tools["valid_device"])
    num_batches = len(dataloader)
    test_loss = 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    # print("in test loop")
    with torch.no_grad():
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple,
                    sample_weight_tuple, pos_neg) in enumerate(dataloader):
            id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple,
                                                                                         seq_frag_list_tuple,
                                                                                         target_frag_nplist_tuple,
                                                                                         type_protein_pt_tuple)
            encoded_seq = tokenize(tools, seq_frag_tuple)
            if type(encoded_seq) == dict:
                for k in encoded_seq.keys():
                    encoded_seq[k] = encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq = encoded_seq.to(tools['valid_device'])
            # print("ok1")
            classification_head, motif_logits, projection_head = tools['net'](
                encoded_seq,
                id_tuple, id_frags_list, seq_frag_tuple,
                None, False)  # for test_loop always used None and False!
            # print("ok2")
            weighted_loss_sum = 0
            # if not warm_starting:
            motif_logits, target_frag = loss_fix(id_frags_list, motif_logits, target_frag_pt, tools)
            sample_weight_pt = torch.from_numpy(np.array(sample_weight_tuple)).to(tools['valid_device']).unsqueeze(1)
            weighted_loss_sum = tools['loss_function'](motif_logits, target_frag.to(tools['valid_device'])) + \
                                torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(
                                    tools['valid_device'])) * sample_weight_pt)

            """
            if configs.supcon.apply and warm_starting:
                supcon_loss = tools['loss_function_supcon'](
                                    projection_head,
                                    configs.supcon.temperature,
                                    configs.supcon.n_pos)
                weighted_loss_sum += configs.supcon.weight * supcon_loss
            """
            test_loss += weighted_loss_sum.item()
            # label = torch.argmax(label_1hot, dim=1)
            # type_pred = torch.argmax(type_probab, dim=1)
            # accuracy.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # macro_f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))

        test_loss = test_loss / num_batches
        # epoch_acc = np.array(accuracy.compute().cpu())
        # epoch_macro_f1 = macro_f1_score.compute().cpu().item()
        # epoch_f1 = np.array(f1_score.compute().cpu())
        # acc_cs = cs_correct / cs_num
    return test_loss


def frag2protein(data_dict, tools):
    overlap = tools['frag_overlap']
    # no_overlap=tools['max_len']-2-overlap
    for id_protein in data_dict.keys():
        id_frag_list = data_dict[id_protein]['id_frag']
        seq_protein = ""
        motif_logits_protein = np.array([])
        motif_target_protein = np.array([])
        for i in range(len(id_frag_list)):
            id_frag = id_protein + "@" + str(i)
            ind = id_frag_list.index(id_frag)
            seq_frag = data_dict[id_protein]['seq_frag'][ind]
            target_frag = data_dict[id_protein]['target_frag'][ind]
            motif_logits_frag = data_dict[id_protein]['motif_logits'][ind]
            l = len(seq_frag)
            if i == 0:
                seq_protein = seq_frag
                motif_logits_protein = motif_logits_frag[:, :l]
                motif_target_protein = target_frag[:, :l]
            else:
                seq_protein = seq_protein + seq_frag[overlap:]
                # x_overlap = np.maximum(motif_logits_protein[:,-overlap:], motif_logits_frag[:,:overlap])
                x_overlap = (motif_logits_protein[:, -overlap:] + motif_logits_frag[:, :overlap]) / 2
                motif_logits_protein = np.concatenate(
                    (motif_logits_protein[:, :-overlap], x_overlap, motif_logits_frag[:, overlap:l]), axis=1)
                motif_target_protein = np.concatenate((motif_target_protein, target_frag[:, overlap:l]), axis=1)
        data_dict[id_protein]['seq_protein'] = seq_protein
        data_dict[id_protein]['motif_logits_protein'] = motif_logits_protein
        data_dict[id_protein]['motif_target_protein'] = motif_target_protein
    return data_dict


def evaluate_protein(dataloader, tools):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    tools['net'].eval().to(tools["valid_device"])
    n = tools['num_classes']
    customlog(tools["logfilepath"], f'number of evaluateion steps: {len(dataloader)}\n')
    print(f'number of evaluateion steps: {len(dataloader)}\n')
    # cutoff = tools['cutoff']
    data_dict = {}
    with torch.no_grad():
        # for batch, (id, id_frags, seq_frag, target_frag, type_protein) in enumerate(dataloader):
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple,
                    sample_weight_tuple, pos_neg) in enumerate(dataloader):
            # id_frags_list, seq_frag_tuple, target_frag_tuple = make_buffer(id_frags, seq_frag, target_frag)
            id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple,
                                                                                         seq_frag_list_tuple,
                                                                                         target_frag_nplist_tuple,
                                                                                         type_protein_pt_tuple)
            encoded_seq = tokenize(tools, seq_frag_tuple)
            if type(encoded_seq) == dict:
                for k in encoded_seq.keys():
                    encoded_seq[k] = encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq = encoded_seq.to(tools['valid_device'])
            classification_head, motif_logits, projection_head = tools['net'](encoded_seq, id_tuple, id_frags_list,
                                                                              seq_frag_tuple, None, False)
            m = torch.nn.Sigmoid()
            motif_logits = m(motif_logits)
            classification_head = m(classification_head)

            x_frag = np.array(motif_logits.cpu())  # [batch, head, seq]
            y_frag = np.array(target_frag_pt.cpu())  # [batch, head, seq]
            x_pro = np.array(classification_head.cpu())  # [sample, n]
            y_pro = np.array(type_protein_pt.cpu())  # [sample, n]
            for i in range(len(id_frags_list)):
                id_protein = id_frags_list[i].split('@')[0]
                j = id_tuple.index(id_protein)
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id_frags_list[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag_tuple[i])
                    data_dict[id_protein]['target_frag'].append(y_frag[i])  # [[head, seq], ...]
                    data_dict[id_protein]['motif_logits'].append(x_frag[i])  # [[head, seq], ...]
                else:
                    data_dict[id_protein] = {}
                    data_dict[id_protein]['id_frag'] = [id_frags_list[i]]
                    data_dict[id_protein]['seq_frag'] = [seq_frag_tuple[i]]
                    data_dict[id_protein]['target_frag'] = [y_frag[i]]
                    data_dict[id_protein]['motif_logits'] = [x_frag[i]]
                    data_dict[id_protein]['type_pred'] = x_pro[j]
                    data_dict[id_protein]['type_target'] = y_pro[j]

        data_dict = frag2protein(data_dict, tools)

        # IoU_difcut=np.zeros([n, 9])
        # FDR_frag_difcut=np.zeros([1,9])
        IoU_pro_difcut = np.zeros([n, 9])  # just for nuc and nuc_export
        # FDR_pro_difcut=np.zeros([1,9])
        result_pro_difcut = np.zeros([n, 6, 9])
        cs_acc_difcut = np.zeros([n, 9])
        classname = ["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
                     "SIGNAL", "chloroplast", "Thylakoid"]
        criteria = ["roc_auc_score", "average_precision_score", "matthews_corrcoef",
                    "recall_score", "precision_score", "f1_score"]

        cutoffs = [x / 10 for x in range(1, 10)]
        cut_dim = 0
        for cutoff in cutoffs:
            scores = get_scores(tools, cutoff, n, data_dict)
            IoU_pro_difcut[:, cut_dim] = scores['IoU_pro']
            result_pro_difcut[:, :, cut_dim] = scores['result_pro']
            cs_acc_difcut[:, cut_dim] = scores['cs_acc']
            cut_dim += 1

        customlog(tools["logfilepath"], f"===========================================\n")
        customlog(tools["logfilepath"], f" Jaccard Index (protein): \n")
        IoU_pro_difcut = pd.DataFrame(IoU_pro_difcut, columns=cutoffs, index=classname)
        customlog(tools["logfilepath"], IoU_pro_difcut.__repr__())
        # IoU_pro_difcut.to_csv(tools["logfilepath"],mode='a',sep="\t")
        customlog(tools["logfilepath"], f"===========================================\n")
        # customlog(tools["logfilepath"], f"===========================================\n")
        customlog(tools["logfilepath"], f" cs acc: \n")
        cs_acc_difcut = pd.DataFrame(cs_acc_difcut, columns=cutoffs, index=classname)
        customlog(tools["logfilepath"], cs_acc_difcut.__repr__())
        customlog(tools["logfilepath"], f"===========================================\n")
        for i in range(len(classname)):
            customlog(tools["logfilepath"], f" Class prediction performance ({classname[i]}): \n")
            tem = pd.DataFrame(result_pro_difcut[i], columns=cutoffs, index=criteria)
            customlog(tools["logfilepath"], tem.__repr__())
            # tem.to_csv(tools["logfilepath"],mode='a',sep="\t")


def get_scores(tools, cutoff, n, data_dict):
    cs_num = np.zeros(n)
    cs_correct = np.zeros(n)
    cs_acc = np.zeros(n)

    # TP_frag=np.zeros(n)
    # FP_frag=np.zeros(n)
    # FN_frag=np.zeros(n)
    # #Intersection over Union (IoU) or Jaccard Index
    # IoU = np.zeros(n)
    # Negtive_detect_num=0
    # Negtive_num=0

    TPR_pro = np.zeros(n)
    FPR_pro = np.zeros(n)
    FNR_pro = np.zeros(n)
    IoU_pro = np.zeros(n)
    # Negtive_detect_pro=0
    # Negtive_pro=0
    result_pro = np.zeros([n, 6])
    for head in range(n):
        x_list = []
        y_list = []
        for id_protein in data_dict.keys():
            x_pro = data_dict[id_protein]['type_pred'][head]  # [1]
            y_pro = data_dict[id_protein]['type_target'][head]  # [1]
            x_list.append(x_pro)
            y_list.append(y_pro)
            if y_pro == 1:
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]  # [seq]
                y_frag = data_dict[id_protein]['motif_target_protein'][head]
                # Negtive_pro += np.sum(np.max(y)==0)
                # Negtive_detect_pro += np.sum((np.max(y)==0) * (np.max(x>=cutoff)==1))
                TPR_pro[head] += np.sum((x_frag >= cutoff) * (y_frag == 1)) / np.sum(y_frag == 1)
                FPR_pro[head] += np.sum((x_frag >= cutoff) * (y_frag == 0)) / np.sum(y_frag == 0)
                FNR_pro[head] += np.sum((x_frag < cutoff) * (y_frag == 1)) / np.sum(y_frag == 1)
                # x_list.append(np.max(x))
                # y_list.append(np.max(y))

                cs_num[head] += np.sum(y_frag == 1) > 0
                if np.sum(y_frag == 1) > 0:
                    cs_correct[head] += (np.argmax(x_frag) == np.argmax(y_frag))

        pred = np.array(x_list)
        target = np.array(y_list)
        result_pro[head, 0] = roc_auc_score(target, pred)
        result_pro[head, 1] = average_precision_score(target, pred)
        result_pro[head, 2] = matthews_corrcoef(target, pred >= cutoff)
        result_pro[head, 3] = recall_score(target, pred >= cutoff)
        result_pro[head, 4] = precision_score(target, pred >= cutoff)
        result_pro[head, 5] = f1_score(target, pred >= cutoff)

    for head in range(n):
        # IoU[head] = TP_frag[head] / (TP_frag[head] + FP_frag[head] + FN_frag[head])
        IoU_pro[head] = TPR_pro[head] / (TPR_pro[head] + FPR_pro[head] + FNR_pro[head])
        cs_acc[head] = cs_correct[head] / cs_num[head]
    # FDR_frag = Negtive_detect_num / Negtive_num
    # FDR_pro = Negtive_detect_pro / Negtive_pro

    scores = {"IoU_pro": IoU_pro,  # [n]
              "result_pro": result_pro,  # [n, 6]
              "cs_acc": cs_acc}  # [n]
    return scores


def main(config_dict, args, valid_batch_number, test_batch_number):
    configs = load_configs(config_dict, args)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    curdir_path, result_path, checkpoint_path, logfilepath = prepare_saving_dir(configs, args.config_path)

    train_writer, valid_writer = prepare_tensorboard(result_path)
    npz_file = os.path.join(curdir_path, "targetp_data.npz")
    seq_file = os.path.join(curdir_path, "idmapping_2023_08_25.tsv")

    customlog(logfilepath, f'use k-fold index: {valid_batch_number}\n')
    # dataloaders_dict = prepare_dataloaders(valid_batch_number, test_batch_number, npz_file, seq_file, configs)
    if configs.train_settings.dataloader == "batchsample":
        dataloaders_dict = prepare_dataloader_batchsample(configs, valid_batch_number, test_batch_number)
    elif configs.train_settings.dataloader == "clean":
        dataloaders_dict = prepare_dataloader_clean(configs, valid_batch_number, test_batch_number)

    customlog(logfilepath, "Done Loading data\n")
    customlog(logfilepath, f'number of training data: {len(dataloaders_dict["train"])}\n')
    customlog(logfilepath, f'number of valid data: {len(dataloaders_dict["valid"])}\n')
    customlog(logfilepath, f'number of test data: {len(dataloaders_dict["test"])}\n')
    print(f'number of training data: {len(dataloaders_dict["train"])}\n')
    print(f'number of valid data: {len(dataloaders_dict["valid"])}\n')
    print(f'number of test data: {len(dataloaders_dict["test"])}\n')
    tokenizer = prepare_tokenizer(configs, curdir_path)
    customlog(logfilepath, "Done initialize tokenizer\n")

    encoder = prepare_models(configs, logfilepath, curdir_path)
    customlog(logfilepath, "Done initialize model\n")

    optimizer, scheduler = prepare_optimizer(encoder, configs, len(dataloaders_dict["train"]), logfilepath)
    if configs.optimizer.mode == 'skip':
        scheduler = optimizer
    customlog(logfilepath, 'preparing optimizer is done\n')
    if args.predict != 1:
        encoder, start_epoch = load_checkpoints(configs, optimizer, scheduler, logfilepath, encoder)

    # w=(torch.ones([9,1,1])*5).to(configs.train_settings.device)
    w = torch.tensor(configs.train_settings.loss_pos_weight, dtype=torch.float32).to(configs.train_settings.device)

    from model import LayerNormNet2
    tools = {
        'frag_overlap': configs.encoder.frag_overlap,
        'cutoffs': configs.predict_settings.cutoffs,
        'composition': configs.encoder.composition,
        'max_len': configs.encoder.max_len,
        # 'tokenizer': tokenizer,
        'prm4prmpro': configs.encoder.prm4prmpro,
        'net': encoder,
        # 'net': LayerNormNet2(configs),
        'train_loader': dataloaders_dict["train"],
        'valid_loader': dataloaders_dict["valid"],
        'test_loader': dataloaders_dict["test"],
        'train_device': configs.train_settings.device,
        'valid_device': configs.valid_settings.device,
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        'optimizer': optimizer,
        # 'loss_function': torch.nn.CrossEntropyLoss(reduction="none"),
        'loss_function': torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="mean"),
        # 'loss_function_pro': torch.nn.BCEWithLogitsLoss(reduction="mean"),
        'loss_function_pro': torch.nn.BCEWithLogitsLoss(reduction="none"),
        'loss_function_supcon': SupConHardLoss,  # Yichuan
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'logfilepath': logfilepath,
        'num_classes': configs.encoder.num_classes
    }

    if args.predict != 1:
        customlog(logfilepath, f'number of train steps per epoch: {len(tools["train_loader"])}\n')
        customlog(logfilepath, "Start training...\n")

        best_valid_loss = np.inf
        global global_step
        global_step = 0
        for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
            warm_starting = False
            if epoch < configs.supcon.warm_start:
                warm_starting = True
                if epoch == 0:
                    print('== Warm Start Began    ==')
                    customlog(logfilepath, f"== Warm Start Began ==\n")

            if epoch == configs.supcon.warm_start:
                best_valid_loss = np.inf  # reset best_valid_loss when warmend ends
                warm_starting = False
                print('== Warm Start Finished ==')
                customlog(logfilepath, f"== Warm Start Finished ==\n")

            tools['epoch'] = epoch
            if global_step % 100 == 0:
                print(f"Fold {valid_batch_number} Epoch {epoch}\n-------------------------------")
                customlog(logfilepath,
                          f"Fold {valid_batch_number} Epoch {epoch} train...\n-------------------------------\n")

            start_time = time()

            train_loss = train_loop(tools, configs, warm_starting, train_writer)
            train_writer.add_scalar('epoch loss', train_loss, global_step=epoch)
            end_time = time()

            if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
                customlog(logfilepath, f'Epoch {epoch}: train loss: {train_loss:>5f}\n')
                print(f'Epoch {epoch}: train loss: {train_loss:>5f}\n')
                print(f"Fold {valid_batch_number} Epoch {epoch} validation...\n-------------------------------\n")
                customlog(logfilepath,
                          f"Fold {valid_batch_number} Epoch {epoch} validation...\n-------------------------------\n")
                start_time = time()
                dataloader = tools["valid_loader"]

                valid_loss = test_loop(tools, dataloader, train_writer,
                                       valid_writer)  # In test loop, never test supcon loss

                valid_writer.add_scalar('epoch loss', valid_loss, global_step=epoch)
                customlog(logfilepath, f'Epoch {epoch}: valid loss:{valid_loss:>5f}\n')
                print(f'Epoch {epoch}: valid loss:{valid_loss:>5f}\n')
                end_time = time()

                if warm_starting:  # in warm_starting only supcon loss, and train_loss
                    if train_loss < best_valid_loss:
                        customlog(logfilepath,
                                  f"Epoch {epoch}: train loss {train_loss} smaller than best loss {best_valid_loss}\n-------------------------------\n")
                        best_valid_loss = train_loss
                        model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
                        customlog(logfilepath,
                                  f"Epoch {epoch}: A better checkpoint is saved into {model_path} \n-------------------------------\n")
                        save_checkpoint(epoch, model_path, tools)
                else:
                    if valid_loss < best_valid_loss:
                        customlog(logfilepath,
                                  f"Epoch {epoch}: valid loss {valid_loss} smaller than best loss {best_valid_loss}\n-------------------------------\n")
                        best_valid_loss = valid_loss
                        model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
                        customlog(logfilepath,
                                  f"Epoch {epoch}: A better checkpoint is saved into {model_path} \n-------------------------------\n")
                        save_checkpoint(epoch, model_path, tools)

    if args.predict == 1:
        if os.path.exists(configs.resume.resume_path):
            model_path = configs.resume.resume_path
        else:
            model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')

    customlog(logfilepath, f"Loading checkpoint from {model_path}\n")
    model_checkpoint = torch.load(model_path, map_location='cpu')
    tools['net'].load_state_dict(model_checkpoint['model_state_dict'])
    customlog(logfilepath, f"Fold {valid_batch_number} test\n-------------------------------\n")
    start_time = time()
    dataloader = tools["test_loader"]
    evaluate_protein(dataloader, tools)
    train_writer.close()
    valid_writer.close()
    end_time = time()

    del tools, encoder, dataloaders_dict, optimizer, scheduler
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CPM')
    parser.add_argument("--config_path", help="The location of config file", default='./config.yaml')
    parser.add_argument("--predict", type=int,
                        help="predict:1 no training, call evaluate_protein; predict:0 call training loop", default=0)
    parser.add_argument("--result_path", default=None,
                        help="result_path, if setted by command line, overwrite the one in config.yaml, "
                             "by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")

    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)

    for i in range(1):
        valid_num = i
        if valid_num == 4:
            test_num = 0
        else:
            test_num = valid_num + 1
        main(config_dict, args, valid_num, test_num)
        break






