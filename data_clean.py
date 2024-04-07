import torch
torch.manual_seed(0)
from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor, Lambda
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils import calculate_class_weights
import pandas as pd
import random
import yaml
from utils import *


class LocalizationDataset(Dataset):
    def __init__(self, samples, configs, mode="train"):
        # self.label_to_index = {"Other": 0, "SP": 1, "MT": 2, "CH": 3, "TH": 4}
        # self.index_to_label = {0: "Other", 1: "SP", 2: "MT", 3: "CH", 4: "TH"}
        # self.transform = transform
        # self.target_transform = target_transform
        # self.cs_transform = cs_transform
        self.samples = samples
        self.n = configs.encoder.num_classes
        # print(self.count_samples_by_class(self.n, self.samples))
        self.class_weights = calculate_class_weights(self.count_samples_by_class(self.n, self.samples))
        # print(self.class_weights)
        self.apply_supcon = configs.supcon.apply
        self.mode = mode
        if self.apply_supcon:
            self.class_id = get_class_id_dict(self.samples)
            self.n_pos = configs.supcon.n_pos
            self.n_neg = configs.supcon.n_neg
            self.hard_neg = configs.supcon.hard_neg
            self.full_list = list(self.class_id.keys())
            print('LocalizationDataset', self.full_list)
    
    @staticmethod
    def count_samples_by_class(n, samples):
        """Count the number of samples for each class."""
        # class_counts = {}
        class_counts = np.zeros(n)  # one extra is for samples without motif
        # Iterate over the samples
        for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
            class_counts += type_protein
        return class_counts
    def __len__(self):
        if self.apply_supcon and self.mode=="train":
           return self.n 
           #one epoch = totol_number of classes, set epoch number to at least epoch_n x len(self.samples)/self.n
        else:
           return len(self.samples)
        
    def __getitem__(self, idx):
        # print('full_list', self.full_list)
        # print('idx of __getitem__', idx)
        if self.apply_supcon and self.mode=="train":
            # exit(0)
            anchor_type = self.full_list[idx]
            anchor = random.choice(self.class_id[anchor_type])
            # print(anchor)
            # exit(0)
            id, id_frag_list, seq_frag_list, target_frag_list, type_protein = self.samples[anchor]
        else:
            id, id_frag_list, seq_frag_list, target_frag_list, type_protein = self.samples[idx]
        
        labels = np.where(type_protein == 1)[0]
        weights = []
        for label in labels:
            weights.append(self.class_weights[label])
        sample_weight = max(weights)
        # labels=np.where(np.max(target_frags, axis=1)==1)[0]
        # weights=[]
        # for label in labels:
        #     weights.append(self.class_weights[label])
        # if np.max(target_frags)==0:
        #     weights.append(self.class_weights[self.n])

        # sample_weight = max(weights)
        # target_frags = torch.from_numpy(np.stack(target_frags, axis=0))
        type_protein = torch.from_numpy(type_protein)
        pos_neg = None
        if self.apply_supcon and self.mode=="train":
            # Even when not in warm starting, the following code is still executed, although its results are not used
            pos_samples = self.get_pos_samples(anchor)
            neg_samples = self.get_neg_samples(anchor)
            pos_neg = [pos_samples, neg_samples]
        return id, id_frag_list, seq_frag_list, target_frag_list, type_protein, sample_weight, pos_neg
        # return id, type_protein

    def get_pos_samples(self, anchor_idx):
        anchor_type_protein = self.samples[anchor_idx][4]  # class 0000 0001
        print(anchor_type_protein)
        # print()
        # print("*"*16)
        # 找到所有正样本
        all_pos_samples = [sample for sample in self.samples if
                           np.any(np.logical_and(anchor_type_protein == 1, sample[4] == 1))]

        pos_samples = []

        # 直接选择n_pos个正样本，允许重复
        for _ in range(self.n_pos):
            chosen_sample = random.choice(all_pos_samples)
            pos_samples.append(chosen_sample)
        #     print(chosen_sample[4])
        # exit(0)
        # 此时final_pos_samples包含了你需要的正样本，数量为self.n_pos，可能包含重复

        pos_samples_with_weight = []
        for sample in pos_samples:
            labels = np.where(sample[4] == 1)[0]
            weights = [self.class_weights[label] for label in labels]
            sample_weight = np.max(weights)
            sample_with_weight = list(sample) + [sample_weight]
            pos_samples_with_weight.append(sample_with_weight)

        return pos_samples_with_weight  # pos_samples

    def get_neg_samples(self, anchor_idx):
        # filtered_samples = [sample for idx, sample in enumerate(self.samples) if idx != anchor_idx]
        anchor_type_protein = self.samples[anchor_idx][4] #class
        # print(anchor_type_protein)
        # print("*"*16)
        if self.hard_neg:
            pass
            # hneg = self.hard_mining(anchor_type_protein) #similiar
            # all_neg_samples = [sample for sample in self.samples if
            #                np.any(np.logical_and(hneg == 1, sample[4] == 1))]
        else:
            all_neg_samples = [sample for sample in self.samples if
                           not np.any(np.logical_and(anchor_type_protein == 1, sample[4] == 1))]
        neg_samples = []

        # 直接选择n_neg个样本，允许重复
        for _ in range(self.n_neg):
            chosen_sample = random.choice(all_neg_samples)
            neg_samples.append(chosen_sample)
            # print(chosen_sample[4])
        # print('*'*32)
        # exit(0)

        neg_samples_with_weight = []
        for sample in neg_samples:
            labels = np.where(sample[4] == 1)[0]
            weights = [self.class_weights[label] for label in labels]
            sample_weight = np.max(weights)
            sample_with_weight = list(sample) + [sample_weight]
            neg_samples_with_weight.append(sample_with_weight)

        return neg_samples_with_weight  # neg_samples

    @staticmethod
    def hard_mining(anchor_type_protein, file_path='distance_map.txt'):
        with open(file_path, 'r') as source_file:
            content = eval(source_file.read())

        # print(content)
        min_non_zero_keys = {}

        for main_key, sub_dict in content.items():
            # Filter out zero values and find the minimum
            non_zero_values = {k: v for k, v in sub_dict.items() if v > 0}
            min_key = min(non_zero_values, key=non_zero_values.get)
            min_non_zero_keys[main_key] = min_key

        label2idx = {"Nucleus": 0, "ER": 1, "Peroxisome": 2, "Mitochondrion": 3, "Nucleus_export": 4,
                     "SIGNAL": 5, "chloroplast": 6, "Thylakoid": 7}
        mapped_dict = {label2idx[key]: label2idx[value] for key, value in min_non_zero_keys.items()}

        neg = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(anchor_type_protein)):
            if anchor_type_protein[i] == 1:
                neg[mapped_dict[i]] = 1
        for i in range(len(anchor_type_protein)):
            if anchor_type_protein[i] == 1:
                neg[i] = 0
        if all(v == 0 for v in neg):
            neg = [1 if x == 0 else 0 for x in anchor_type_protein]
        return np.array(neg)

def custom_collate(batch):
    id, id_frags, fragments, target_frags, type_protein, sample_weight, pos_neg = zip(*batch)
    return id, id_frags, fragments, target_frags, type_protein, sample_weight, pos_neg


def prot_id_to_seq(seq_file):
    id2seq = {}
    with open(seq_file) as file:
        for line in file:
            id = line.strip().split("\t")[0]
            seq = line.strip().split("\t")[2]
            id2seq[id] = seq
    return id2seq

# def prepare_samples(exclude, fold_num_list, id2seq_dic, npz_file):
#     samples = []
#     data = np.load(npz_file)
#     fold = data['fold']
#     if exclude:
#         index = [all(num != target for target in fold_num_list) for num in fold]
#         # index = fold != fold_num
#     else:
#         index = [any(num == target for target in fold_num_list) for num in fold]
#         # index = fold == fold_num
#     prot_ids = data['ids'][index]
#     y_type = data['y_type'][index]
#     y_cs = data['y_cs'][index]
#     for idx, prot_id in enumerate(prot_ids):
#         seq = id2seq_dic[prot_id]
#         # if len(seq)>200:
#         #   seq=seq[:200]
#         label = y_type[idx]
#         position = np.argmax(y_cs[idx])
#         samples.append((seq, int(label), position))
#     return samples

def split_protein_sequence(prot_id, sequence, targets, configs):
    fragment_length = configs.encoder.max_len - 2
    overlap = configs.encoder.frag_overlap
    fragments = []
    target_frags = []
    id_frags = []
    sequence_length = len(sequence)
    start = 0
    ind = 0

    while start < sequence_length:
        end = start + fragment_length
        if end > sequence_length:
            end = sequence_length
        fragment = sequence[start:end]
        target_frag = targets[:, start:end]
        if target_frag.shape[1] < fragment_length:
            pad = np.zeros([targets.shape[0], fragment_length-target_frag.shape[1]])
            target_frag = np.concatenate((target_frag, pad), axis=1)
        target_frags.append(target_frag)
        fragments.append(fragment)
        id_frags.append(prot_id+"@"+str(ind))
        ind += 1
        if start + fragment_length > sequence_length:
            break
        start += fragment_length - overlap

    return id_frags, fragments, target_frags


#["Nucleus","ER","Peroxisome","Mitochondrion","Nucleus_export","dual","SIGNAL","chloroplast","Thylakoid"]
def fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets):
    if motif_left == "None":
        motif_left = 0
    else:
        motif_left = int(motif_left)-1
    motif_right = int(motif_right)
    if label == "Thylakoid" and motif_left != 0:
        index_row = label2idx["chloroplast"]
        type_protein[index_row] = 1
        targets[index_row, motif_left-1] = 1
    return motif_left, motif_right, type_protein, targets


def prepare_samples(csv_file, configs):
    # label2idx = {"Nucleus":0, "ER":1, "Peroxisome":2, "Mitochondrion":3, "Nucleus_export":4,
    #              "dual":5, "SIGNAL":6, "chloroplast":7, "Thylakoid":8}
    label2idx = {"Nucleus": 0, "ER": 1, "Peroxisome": 2, "Mitochondrion": 3, "Nucleus_export": 4,
                 "SIGNAL": 5, "chloroplast": 6, "Thylakoid": 7}
    samples = []
    n = configs.encoder.num_classes
    df = pd.read_csv(csv_file)
    row, col = df.shape
    for i in range(row):
        prot_id = df.loc[i, "Entry"]
        seq = df.loc[i, "Sequence"]
        targets = np.zeros([n, len(seq)])
        type_protein = np.zeros(n)
        # motifs = df.iloc[i,1:-2]
        motifs = df.loc[i, "MOTIF"].split("|")
        for motif in motifs:
            if not pd.isnull(motif):
                # label = motif.split("|")[0].split(":")[1]
                label = motif.split(":")[1]
                # motif_left = motif.split("|")[0].split(":")[0].split("-")[0]
                motif_left = motif.split(":")[0].split("-")[0]
                motif_right = motif.split(":")[0].split("-")[1]
                
                motif_left, motif_right, type_protein, targets = fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets)
                if label in label2idx:
                    index_row = label2idx[label]
                    type_protein[index_row] = 1
                    if label in ["SIGNAL", "chloroplast", "Thylakoid", "Mitochondrion"]:
                        targets[index_row, motif_right-1] = 1
                    elif label == "Peroxisome" and motif_left == 0:
                        targets[index_row, motif_right-1] = 1
                    elif label == "Peroxisome" and motif_left != 0:
                        targets[index_row, motif_left] = 1
                    elif label == "ER":
                        targets[index_row, motif_left] = 1
                    elif label == "Nucleus" or label == "Nucleus_export":
                        targets[index_row, motif_left:motif_right] = 1
        
        id_frag_list, seq_frag_list, target_frag_list = split_protein_sequence(prot_id, seq, targets, configs)
        samples.append((prot_id, id_frag_list, seq_frag_list, target_frag_list, type_protein))
        # for j in range(len(fragments)):
        #     id=prot_id+"@"+str(j)
        #     samples.append((id, fragments[j], target_frags[j], type_protein))
    return samples


def prepare_dataloaders(configs, valid_batch_number, test_batch_number):
    # id_to_seq = prot_id_to_seq(seq_file)
    if configs.train_settings.dataset == 'v2':
        samples = prepare_samples("./parsed_EC7_v2/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v2/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v2/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v2/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v3':
        samples = prepare_samples("./parsed_EC7_v3/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v3/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v3/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v3/split/type/partition.csv")

    train_id = []
    val_id = []
    test_id = []
    id = cv.loc[:, 'entry']
    # split=cv.loc[:,'split']
    # fold=cv.loc[:,'fold']
    partition = cv.loc[:, 'partition']
    for i in range(len(id)):
        # f=fold[i]
        # s=split[i]
        p = partition[i]
        d = id[i]
        if p == valid_batch_number:
            val_id.append(d)
        elif p == test_batch_number:
            test_id.append(d)
        else:
            train_id.append(d)


    # print(train_id)


    train_sample = []
    valid_sample = []
    test_sample = []

    for i in samples:
        # id=i[0].split("@")[0]
        id = i[0]
        # print(id)
        if id in train_id:
            train_sample.append(i)
        elif id in val_id:
            valid_sample.append(i)
        elif id in test_id:
            test_sample.append(i)
    print(len(train_sample))
    print(len(valid_sample))
    print(len(test_sample))

    # train_samples = prepare_samples(exclude=True, fold_num_list=[valid_batch_number, test_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    # valid_samples = prepare_samples(exclude=False, fold_num_list=[valid_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    # test_samples = prepare_samples(exclude=False, fold_num_list=[test_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    random.seed(configs.fix_seed)
    # Shuffle the list
    random.shuffle(samples)
    # train_dataset = LocalizationDataset(samples, configs=configs)
    # dataset_size = len(train_dataset)
    # train_size = int(0.8 * dataset_size)  # 80% for training, adjust as needed
    # test_size = dataset_size - train_size
    # train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
    # val_size = train_size - int(0.8 * train_size)
    # train_dataset, valid_dataset = random_split(train_dataset, [int(0.8 * train_size), val_size])



    # print(train_dataset)
    train_dataset = LocalizationDataset(train_sample, configs=configs,mode="train")
    valid_dataset = LocalizationDataset(valid_sample, configs=configs,mode="valid")
    test_dataset = LocalizationDataset(test_sample, configs=configs,mode="test")
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=custom_collate,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size, shuffle=False, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.valid_settings.batch_size, shuffle=False, collate_fn=custom_collate)
    return {'train': train_dataloader, 'test': test_dataloader, 'valid': valid_dataloader}


if __name__ == '__main__':
    config_path = './config.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dataloaders_dict = prepare_dataloaders(configs_file, 0, 1)

    for batch in dataloaders_dict['train']:
        # id_batch, fragments_batch, target_frags_batch, weights_batch = batch
        (prot_id, id_frag_list, seq_frag_list, target_frag_nplist, type_protein_pt, sample_weight) = batch
        # id, type_protein = batch
        # print(len(id_batch))
        # print(len(fragments_batch))
        # print(np.array(target_frags_batch).shape)
        # print(len(weights_batch))
        print("==========================")
        print(type(prot_id))
        print(prot_id)
        print(type(id_frag_list))
        print(id_frag_list)
        print(type(seq_frag_list))
        print(seq_frag_list)
        print(type(target_frag_nplist))
        print(target_frag_nplist)
        print(type(type_protein_pt))
        print(type_protein_pt)
        print(type(sample_weight))
        print(sample_weight)
        # print(next(iter(dataloaders_dict['test'])))
        # a=np.array(target_frags_batch)
        # print(np.max(a, axis=2))
        # print(target_frags_batch.size())
        # print(target_frags_batch[1])
        # print(weights_batch)
        break

    print('done')

