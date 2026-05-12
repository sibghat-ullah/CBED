import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, balanced_accuracy_score, classification_report
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd
from tqdm import tqdm
import json
import os
from scipy import stats

class Evaluator:
    def __init__(self, config, data_dict):
        self.config = config
        self.data_dict = data_dict
        self.class_names = config.classes
        self.idx_to_class = data_dict['idx_to_class']
        
    def evaluate_classification(self, model, dataloader, split_name='test'):
        """Evaluate intent classification"""
        model.eval()
        all_preds = []
        all_labels = []
        all_attention_weights = []
        
        print(f"\n{'='*50}")
        print(f"Evaluating Classification on {split_name}")
        print(f"{'='*50}")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Classifying"):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['intent'].to(self.config.device)
                
                # Get predictions with attention weights
                logits, h_fused, attention_weights = model(
                    input_ids, attention_mask, return_attention=True
                )
                
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_attention_weights.extend(attention_weights.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_attention_weights = np.array(all_attention_weights)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=range(self.config.num_classes)
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Attention analysis
        attention_confidence = np.abs(all_attention_weights[:, 0] - all_attention_weights[:, 1])
        
        # Stratify accuracy by attention confidence
        high_conf_mask = attention_confidence > 0.5
        medium_conf_mask = (attention_confidence >= 0.2) & (attention_confidence <= 0.5)
        low_conf_mask = attention_confidence < 0.2
        
        high_conf_acc = accuracy_score(
            all_labels[high_conf_mask], all_preds[high_conf_mask]
        ) if high_conf_mask.sum() > 0 else 0
        
        medium_conf_acc = accuracy_score(
            all_labels[medium_conf_mask], all_preds[medium_conf_mask]
        ) if medium_conf_mask.sum() > 0 else 0
        
        low_conf_acc = accuracy_score(
            all_labels[low_conf_mask], all_preds[low_conf_mask]
        ) if low_conf_mask.sum() > 0 else 0
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Overall Metrics:")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        
        print(f"\n{'='*50}")
        print(f"Per-Class Metrics:")
        print(f"{'='*50}")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 70)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {support[i]:<10}")
        
        print(f"\n{'='*50}")
        print(f"Attention Analysis:")
        print(f"{'='*50}")
        print(f"Average Confidence: {attention_confidence.mean():.4f}")
        print(f"Std Confidence: {attention_confidence.std():.4f}")
        print(f"\nAttention Distribution:")
        print(f"  High confidence (>0.5): {100*high_conf_mask.sum()/len(attention_confidence):.1f}%")
        print(f"  Medium confidence (0.2-0.5): {100*medium_conf_mask.sum()/len(attention_confidence):.1f}%")
        print(f"  Low confidence (<0.2): {100*low_conf_mask.sum()/len(attention_confidence):.1f}%")
        
        print(f"\nAccuracy by Confidence Level:")
        print(f"  High confidence: {high_conf_acc:.4f}")
        print(f"  Medium confidence: {medium_conf_acc:.4f}")
        print(f"  Low confidence: {low_conf_acc:.4f}")
        
        # Rare class analysis
        rare_class_indices = [self.class_names.index(cls) for cls in self.config.rare_classes]
        rare_mask = np.isin(all_labels, rare_class_indices)
        if rare_mask.sum() > 0:
            rare_acc = accuracy_score(all_labels[rare_mask], all_preds[rare_mask])
            rare_f1 = f1[rare_class_indices].mean()
            print(f"\n{'='*50}")
            print(f"Rare Class Performance:")
            print(f"{'='*50}")
            print(f"Rare Class Accuracy: {rare_acc:.4f}")
            print(f"Rare Class Avg F1: {rare_f1:.4f}")
        
        # Save detailed results
        results = {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'per_class': {
                self.class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(self.config.num_classes)
            },
            'attention': {
                'avg_confidence': float(attention_confidence.mean()),
                'std_confidence': float(attention_confidence.std()),
                'high_conf_pct': float(100*high_conf_mask.sum()/len(attention_confidence)),
                'medium_conf_pct': float(100*medium_conf_mask.sum()/len(attention_confidence)),
                'low_conf_pct': float(100*low_conf_mask.sum()/len(attention_confidence)),
                'high_conf_acc': float(high_conf_acc),
                'medium_conf_acc': float(medium_conf_acc),
                'low_conf_acc': float(low_conf_acc)
            },
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def evaluate_generation(self, encoder1, encoder2, fusion, classifier, 
                           smollm_decoder, phi_decoder, dataloader, 
                           smollm_tokenizer, phi_tokenizer, split_name='test'):
        """Evaluate response generation"""
        encoder1.eval()
        encoder2.eval()
        fusion.eval()
        classifier.eval()
        smollm_decoder.eval()
        phi_decoder.eval()
        
        print(f"\n{'='*50}")
        print(f"Evaluating Generation on {split_name}")
        print(f"{'='*50}")
        
        all_references = []
        all_hypotheses = []
        decoder_usage = {'smollm': 0, 'phi': 0}
        
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        bleu_scores = []
        rouge_scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating"):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                intent_labels = batch['intent'].to(self.config.device)
                response_texts = batch['response_mask']  # Original text
                
                # Get intent predictions
                h1 = encoder1(input_ids, attention_mask)
                h2 = encoder2(input_ids, attention_mask)
                h_fused, _ = fusion(h1, h2)
                logits = classifier(h_fused)
                predicted_intents = torch.argmax(logits, dim=-1)
                
                # Route to appropriate decoder
                for i in range(len(predicted_intents)):
                    intent_name = self.idx_to_class[predicted_intents[i].item()]
                    
                    # Select decoder based on intent
                    if intent_name in self.config.open_domain_classes:
                        decoder = phi_decoder
                        tokenizer = phi_tokenizer
                        decoder_usage['phi'] += 1
                    else:
                        decoder = smollm_decoder
                        tokenizer = smollm_tokenizer
                        decoder_usage['smollm'] += 1
                    
                    # Generate response
                    max_length = 100
                    generated_ids = decoder.generate(
                        input_ids[i:i+1],
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    hypothesis = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    reference = tokenizer.decode(batch['response_ids'][i], skip_special_tokens=True)
                    
                    all_hypotheses.append(hypothesis)
                    all_references.append(reference)
                    
                    # Calculate BLEU for this sample
                    ref_tokens = reference.split()
                    hyp_tokens = hypothesis.split()
                    smoothie = SmoothingFunction().method4
                    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
                    bleu_scores.append(bleu)
                    
                    # Calculate ROUGE for this sample
                    rouge = rouge_scorer_obj.score(reference, hypothesis)
                    rouge_scores.append(rouge['rougeL'].fmeasure)
        
        # Calculate BERTScore
        print("\nCalculating BERTScore...")
        P, R, F1 = bert_score(all_hypotheses, all_references, lang='en', verbose=False)
        bert_scores = F1.numpy()
        
        # Aggregate metrics
        avg_bleu = np.mean(bleu_scores)
        avg_rouge = np.mean(rouge_scores)
        avg_bert_score = np.mean(bert_scores)
        
        print(f"\n{'='*50}")
        print(f"Generation Metrics:")
        print(f"{'='*50}")
        print(f"BLEU-4: {avg_bleu:.4f}")
        print(f"ROUGE-L: {avg_rouge:.4f}")
        print(f"BERTScore: {avg_bert_score:.4f}")
        
        print(f"\n{'='*50}")
        print(f"Decoder Usage:")
        print(f"{'='*50}")
        total = decoder_usage['smollm'] + decoder_usage['phi']
        print(f"SmolLM: {decoder_usage['smollm']} ({100*decoder_usage['smollm']/total:.1f}%)")
        print(f"Phi-1.5: {decoder_usage['phi']} ({100*decoder_usage['phi']/total:.1f}%)")
        
        results = {
            'bleu': float(avg_bleu),
            'rouge_l': float(avg_rouge),
            'bert_score': float(avg_bert_score),
            'decoder_usage': {
                'smollm': int(decoder_usage['smollm']),
                'phi': int(decoder_usage['phi']),
                'smollm_pct': float(100*decoder_usage['smollm']/total),
                'phi_pct': float(100*decoder_usage['phi']/total)
            }
        }
        
        return results
    
    def statistical_significance(self, results_list, baseline_idx=0):
        """Perform paired t-tests for statistical significance"""
        print(f"\n{'='*50}")
        print(f"Statistical Significance Testing")
        print(f"{'='*50}")
        
        baseline = results_list[baseline_idx]
        
        for i, results in enumerate(results_list):
            if i == baseline_idx:
                continue
            
            # Paired t-test on F1 scores
            t_stat, p_value = stats.ttest_rel(baseline['f1_scores'], results['f1_scores'])
            
            print(f"\nModel {i} vs Baseline:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

def main():
    """Example evaluation script"""
    from config import Config
    from data_loader import load_data, create_dataloaders
    from models import TransformerEncoder, ChannelAttentionFusion, IntentClassifier, DualEncoderSystem
    from transformers import AutoTokenizer
    
    config = Config()
    torch.manual_seed(config.seed)
    
    print("="*70)
    print("FULL SYSTEM EVALUATION")
    print("="*70)
    
    # Load data
    data_dict = load_data(config)
    val_loader = create_dataloaders(data_dict, config, stage='stage2')['val']
    
    # Load full system
    print("\nLoading trained models...")
    
    # Encoders
    encoder1 = TransformerEncoder(config)
    encoder2 = TransformerEncoder(config)
    checkpoint1 = torch.load(f"{config.checkpoint_dir}/stage1_encoders/encoder1_frequent.pt")
    checkpoint2 = torch.load(f"{config.checkpoint_dir}/stage1_encoders/encoder2_rare.pt")
    encoder1.load_state_dict(checkpoint1['encoder'])
    encoder2.load_state_dict(checkpoint2['encoder'])
    
    # Fusion + Classifier
    fusion = ChannelAttentionFusion(config)
    classifier = IntentClassifier(config)
    checkpoint_fusion = torch.load(f"{config.checkpoint_dir}/stage2_fusion/fusion_classifier.pt")
    fusion.load_state_dict(checkpoint_fusion['fusion'])
    classifier.load_state_dict(checkpoint_fusion['classifier'])
    
    # Create full model
    full_model = DualEncoderSystem(config)
    full_model.encoder1 = encoder1
    full_model.encoder2 = encoder2
    full_model.fusion = fusion
    full_model.classifier = classifier
    full_model = full_model.to(config.device)
    full_model.eval()
    
    # Initialize evaluator
    evaluator = Evaluator(config, data_dict)
    
    # Evaluate classification
    classification_results = evaluator.evaluate_classification(full_model, val_loader, 'validation')
    
    # Save results
    os.makedirs(config.results_dir, exist_ok=True)
    with open(f"{config.results_dir}/classification_results.json", 'w') as f:
        json.dump(classification_results, f, indent=2)
    
    print(f"\n✓ Results saved to {config.results_dir}/")

if __name__ == "__main__":
    main()