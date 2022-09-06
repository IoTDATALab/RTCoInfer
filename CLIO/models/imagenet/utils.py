try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def process_state_dict(state_dict):
    # process state dict so that it can be loaded by normal models
    for k in list(state_dict.keys()):
        state_dict[k.replace('module.', '')] = state_dict.pop(k)
    return state_dict