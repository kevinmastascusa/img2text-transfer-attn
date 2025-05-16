import torch
from training.metrics import compute_bleu, compute_meteor, compute_cider

def evaluate_model(model, dataloader, device):
    model.eval()
    bleu_scores, meteor_scores, cider_scores = [], [], []

    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions[:, :-1])
            bleu_scores.append(compute_bleu(outputs, captions))
            meteor_scores.append(compute_meteor(outputs, captions))
            cider_scores.append(compute_cider(outputs, captions))

    return {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
        "CIDEr": sum(cider_scores) / len(cider_scores),
    }
