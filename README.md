# MUTarget + CLEAN 240315

The primary objective of this branch is to integrate MUTarget with contrastive learning while ensuring minimal modifications to the original MUTarget codebase.

## New Warm Starting Strategies 240315

In the train.train_loop, determine in advance whether there is a need to extend the batch; if so, do the extension on the batched data. 
Afterwards, in model.Encoder::forward(), process batched data through the corresponding conditional branches based on different scenarios to obtain the various classification_head, motif_logits, and projection_head.

Changes in train.train_loop, model.Encoder::forward, data.LocalizationDataset::get_pos_samples & get_neg_samples (sample_with_weight)

## New Configuration Parameters Description 240304

Below are the descriptions for some important new configuration parameters, specifically regarding the use of SupCon and other related settings:

- **`apply`**: This parameter determines whether to use SupCon in the model training process. Setting it to `False` disables the use of SupCon.

- **`n_pos`**: Represents the number of positive samples to be used for each anchor in the contrastive learning setup. 

- **`n_neg`**: Indicates the number of negative samples for each anchor. 

- **`temperature`**: A scaling parameter used in the loss function of contrastive learning models. 

- **`hard_neg`**: A boolean parameter that, when set to `True`, indicates the model should select harder negative samples for computing the loss. Hard negative samples are those that are more challenging for the model to correctly distinguish.

- **`weight`**: Set it as 1. This parameter was previously used but is no longer necessary.

- **`warm_start`**: Specifies the epoch at which warm starting ends. 


## Changes in `data.py` 240304

### `LocalizationDataset` Class
- **`__getitem__` Method Enhancement**: A new return value `pos_neg` is added. When using SupCon, this return value contains a list `[pos_samples, neg_samples]`, where `pos_samples` and `neg_samples` are lists of samples used for contrastive learning. The code to get `pos_neg` is executed even when not in warm starting, although its results are not utilized.

- **`get_pos_samples` Method**: Identifies the positive samples for a given anchor index. A sample qualifies as positive if it matches at least one category with the anchor. For instance, an anchor `[0100 0000]` and a positive `[1100 0000]`. If the selected number of positive samples is less than `n_pos`, it is randomly multiplied to match `n_pos`.

- **`get_neg_samples` Method**: Finds the negative samples for a specified anchor index. The categories of negative samples must not overlap with the anchor. For example, an anchor `[0100 0000]` and a negative `[0011 0000]`.

- **`hard_mining` Method**: If `get_neg_samples` opts for hard mining mode, this function is called to select negative sample template. For each category, it selects the hardest negative template based on a distance map file, ensures there's no overlap among hardest negative template with the anchor, and excludes any overlapping items in the negative template. If exclusion results in `[0000 0000]`, it abandons hard mining for a standard negative sample selection.

- **`prepare_samples` Method**: Fixed a bug that caused failures when reading datasets containing dual data.

## Changes in `train.py` 240304

- **`train_loop` Function**: Added conditions to check for SupCon usage and warm_starting status. If in warm starting and using SupCon, `loss_function` and `loss_function_pro` are not computed, and their corresponding networks are not engaged. Only `loss_function_supcon` is calculated.

## Changes in `model.py` 240304

- **`Encoder` Class**: Depending on the use of SupCon and warm_starting status, it chooses between connecting (`ParallelLinearDecoders` and `Linear`) or only `LayerNormNet`. `LayerNormNet` is from CLEAN. When connecting `LayerNormNet`, `pos_neg` is processed and input as `[bsz, 2(0:pos, 1:neg), n_pos(or n_neg), 5(variables)] -> [n_pos, 5, bsz] + [n_neg, 5, bsz]`. For each positive and negative sample, embeddings are obtained and concatenated into `[bcz, (1+npos+nneg), len(embedding)]`. The projection head is then fetched, formatting the concatenation as `[bcz, (1+npos+nneg), len(projection)]`.

## Changes in `loss.py` 240304

- **`SupConHardLoss` Function**: Originates from CLEAN.

