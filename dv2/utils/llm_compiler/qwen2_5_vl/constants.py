PATCH_EMBED = "patch_embed_conv3d"
HIDDEN = "hidden"
INPUT = "vision_input"
ROPE_INPUT = "rope_input"

ONNX_PATCH_EMBED_PROJ_WEIGHT = 'visual.patch_embed.proj.weight'
ONNX_PATCH_EMBED_ROTATION_WEIGHT = 'visual.patch_embed.rotation.weight'

QWEN2_5_VL_PREFIX = 'qwen2_5_vl'

# dimension names
IN_CHANNELS = "in_channels"
N = "n"
BATCH = "batch"
CHANNEL_DIM = "channel_dim"
PIXELS_HEIGHT = "pixels_height"
PIXELS_WIDTH = "pixels_width"
TEMPORAL_DIM = "temporal_dim"
HEIGHT_TEMPORAL_DIM = "height_temporal_dim"
WIDTH_TEMPORAL_DIM = "width_temporal_dim"
WIDTH_WINDOWS_AND_ALL_TEMPORAL = "width_windows_and_all_temporal"
HEIGHT_WINDOWS = "height_windows"
WIDTH_WINDOWS = "width_windows"
NUM_WINDOWS_WIDTH = "num_windows_width"
PATCH_DIM_0 = "patch_dim_0"


UNSQUEEZE = "unsqueeze"
TKN = "tkn"
NUM_HEAD = "num_head"
HEAD_DIM = "head_dim"
NUM_WINDOWS = "num_windows"
WINDOWED_TKNS = "windowed_tkns"
WINDOWED_TKNS_ACC = "windowed_tkns_acc"
TKN_ACC = "tkn_acc"


FIXED_HEIGHT = 12
FIXED_WIDTH = 12

DEFAULT_GROUP_SIZE = 64

VISION_ROPE = "vision_rope"
VISION_SIN_CACHE = 'vision_sin_cache'
VISION_COS_CACHE = 'vision_cos_cache'