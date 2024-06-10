from model import Net



if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")