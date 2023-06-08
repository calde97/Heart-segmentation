import nrrd
import numpy as np
from scipy import ndimage


# %%

def largest_nonzero_segment(vector):
    # indices of non-zero elements
    nonzero_indices = np.nonzero(vector)[0]
    # compute the difference between consecutive indices
    diff = np.diff(nonzero_indices)
    # put to 0 the values that are not 1
    diff[diff != 1] = 0
    # save the indices of the zero values
    zero_indices = np.where(diff == 0)[0]
    if len(zero_indices) == 1:
        if zero_indices[0] - 1 > len(nonzero_indices) - zero_indices[0]:
            return nonzero_indices[0], nonzero_indices[zero_indices[0]]
        return nonzero_indices[zero_indices[0] + 1], nonzero_indices[-1]
    # compute the difference between consecutive indices
    diff2 = np.diff(zero_indices)
    max_arg = np.argmax(diff2)

    start_index = zero_indices[max_arg] + 1

    return nonzero_indices[start_index], nonzero_indices[start_index] + diff2[max_arg] - 1


def largest_positive_component_indices(arr):
    max_start, max_length = -1, 0
    curr_start, curr_length = -1, 0

    for i, value in enumerate(arr):
        if value > 0:
            if curr_start == -1:  # Start of a new positive component
                curr_start = i
            curr_length += 1
        else:
            if curr_length > max_length:  # Update the largest component
                max_start, max_length = curr_start, curr_length
            curr_start, curr_length = -1, 0  # Reset current component

    # Check if the last positive component is the largest
    if curr_length > max_length:
        max_start, max_length = curr_start, curr_length

    return list(range(max_start, max_start + max_length)) if max_length > 0 else []


def filter_by_pixel_area(segmentation_mask, largest_segment):
    # all outputs before largest segment[0] are set to 0 and largest segment[1] to 0
    for i in range(largest_segment[0]):
        segmentation_mask[i] = 0

    for i in range(largest_segment[1] + 1, len(segmentation_mask)):
        segmentation_mask[i] = 0

    return segmentation_mask


def filter_non_connected_components(segmentation_mask, positive_indices):
    # put to 0 all the values that are not in the positive indices
    for i in range(len(segmentation_mask)):
        if i not in positive_indices:
            segmentation_mask[i] = 0

    return segmentation_mask


def dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    if (np.sum(mask1) + np.sum(mask2)) == 0:
        return 0
    return 2 * intersection / (np.sum(mask1) + np.sum(mask2))


def create_dice_scores(segmentation_mask):
    dice_scores = []
    for i in range(len(segmentation_mask) - 1):
        mask1 = segmentation_mask[i]
        mask2 = segmentation_mask[i + 1]
        dice_score = dice_coefficient(mask1, mask2)
        dice_scores.append(dice_score)

    return dice_scores


'''plt.bar(range(len(dice_scores)), dice_scores)
plt.show()'''


def define_threshold(dice_scores, number_of_standard_deviations=3.0):
    # extract the non zero dice scores and put them in a new vector
    positive_dice_scores = [dice_score for dice_score in dice_scores if dice_score != 0]
    # number of non zero dice scores
    non_zero_dice_scores = len(positive_dice_scores)
    # sum of all elements dice scores
    sum_dice_scores = np.sum(dice_scores)
    # average dice score
    average_dice_score = sum_dice_scores / non_zero_dice_scores
    variance_dice_score = np.std(positive_dice_scores)
    threshold = average_dice_score - number_of_standard_deviations * variance_dice_score

    # if threshold is below 0.6 set it to 0.6
    if threshold < 0.6:
        threshold = 0.6

    return threshold


def filter_mask_by_dice_score2(dice_scores, segmentation_mask, threshold):
    for i in range(len(dice_scores)):
        if dice_scores[i] < threshold:
            segmentation_mask[i] = 0

    return segmentation_mask


def filter_mask_by_dice_score(dice_scores, segmentation_mask, threshold):
    index = 0
    # Initial step: set all the masks to 0
    for i in range(len(dice_scores)):
        if dice_scores[i] < threshold:
            segmentation_mask[i] = 0
        else:
            index = i
            break

    # Iterate over the rest of the masks
    for i in range(index, len(dice_scores)):
        if dice_scores[i] < threshold:
            index = i + 1
            break
    # put to 0 all masks from index to the end
    for i in range(index, len(dice_scores)):
        segmentation_mask[i] = 0

    return segmentation_mask


def keep_largest_connected_component(mask):
    if np.sum(mask) == 0:
        return mask

    # discretize the mask
    mask = np.where(mask > 0.5, 1, 0)

    # Label connected components (objects)
    labeled_mask, num_labels = ndimage.label(mask)

    # Count the size of each connected component
    label_sizes = np.bincount(labeled_mask.ravel())

    # Ignore the background (label 0) when finding the largest label
    largest_label = label_sizes[1:].argmax() + 1

    # Create a new mask with only the largest connected component (object)
    new_mask = np.where(labeled_mask == largest_label, 1, 0)

    return new_mask


# %%
'''patient = 'AM29'
# type = 'Data_ipertrofia'
type = 'Data_amiloidosi_new'
# type = 'Data_stenosi_new'
images = f'/home/calde/Desktop/master-thesis-corino/test_predictions/{type}/{patient}/ct_scan_{patient}.nrrd'
outputs = f'/home/calde/Desktop/master-thesis-corino/test_predictions/{type}/{patient}/bm_mask_{patient}.nrrd'''


# %%

def post_process(outputs):
    segmentation_mask = nrrd.read(outputs)[0]
    pixels = [np.count_nonzero(mask == 1) / mask.size for mask in segmentation_mask]

    # Keep only the largest non-zero segment
    largest_segment = largest_positive_component_indices(pixels)
    # Filter out all the masks that are not part of the largest segment
    segmentation_mask = filter_non_connected_components(segmentation_mask, largest_segment)

    dice_scores = create_dice_scores(segmentation_mask)
    thr = define_threshold(dice_scores, number_of_standard_deviations=2)

    '''plt.bar(range(len(dice_scores)), dice_scores)
    plt.show()'''

    segmentation_mask = filter_mask_by_dice_score2(dice_scores, segmentation_mask, thr)
    # Filter out again the new empty segments generated by the previous step
    # Keep only the largest non-zero segment

    pixels = [np.count_nonzero(mask == 1) / mask.size for mask in segmentation_mask]
    largest_segment = largest_positive_component_indices(pixels)
    # Filter out all the masks that are not part of the largest segment
    segmentation_mask = filter_non_connected_components(segmentation_mask, largest_segment)

    segmentation_mask = [keep_largest_connected_component(mask) for mask in segmentation_mask]
    segmentation_mask = np.array(segmentation_mask)

    return segmentation_mask
