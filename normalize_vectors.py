from behaviors import ALL_BEHAVIORS, get_vector_path
from utils.helpers import get_model_path
import torch as t
import os

all_behaviors = ["gsm8k","multia","addsub" ,"asdiv" ,"svamp","mawps"]
def normalize_vectors(model_size: str, is_base: bool, n_layers: int):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    for layer in range(n_layers):
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in all_behaviors: #ALL_BEHAVIORS:
            vec_path = get_vector_path(behavior, layer, get_model_path(model_size, is_base=is_base))
            vec = t.load(vec_path)
            norm = vec.norm().item()  # 计算 norm	即 L2 范数，衡量向量大小
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path
        print(norms)
        mean_norm = t.tensor(list(norms.values())).mean().item()  #  计算当前层所有行为向量的 平均范数
        # normalize all vectors to have the same norm
        for behavior in all_behaviors: #ALL_BEHAVIORS:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]  # 重新缩放所有向量到 mean norm
        # save the normalized vectors
        for behavior in all_behaviors: #ALL_BEHAVIORS:
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])
    
    
if __name__ == "__main__":
    # normalize_vectors("7b", True, 32)
    normalize_vectors("7b", False, 32)
    # normalize_vectors("13b", False, 36)