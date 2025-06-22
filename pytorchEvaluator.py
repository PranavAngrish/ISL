import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime


class ISLEvaluator:
    """PyTorch model evaluator for ISL detection"""
    
    def __init__(self, model, test_loader, device, class_names, save_dir="evaluation_results"):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def evaluate(self):
        """Evaluate the model on test data"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                frames, landmarks = data
                frames, landmarks, target = frames.to(self.device), landmarks.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(frames, landmarks)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                # Get predictions
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Store results for detailed analysis
        self.all_predictions = np.array(all_predictions)
        self.all_targets = np.array(all_targets)
        self.all_probabilities = np.array(all_probabilities)
        
        return accuracy, avg_loss
    
    def calculate_top_k_accuracy(self, k=3):
        """Calculate top-k accuracy"""
        return top_k_accuracy_score(self.all_targets, self.all_probabilities, k=k)
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        report = classification_report(
            self.all_targets, 
            self.all_predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        return report
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.all_targets, self.all_predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_class_accuracy(self, save_path=None):
        """Plot per-class accuracy"""
        cm = confusion_matrix(self.all_targets, self.all_predictions)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(self.class_names)), class_accuracy)
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'class_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_accuracy
    
    def analyze_misclassifications(self, top_n=10):
        """Analyze most common misclassifications"""
        cm = confusion_matrix(self.all_targets, self.all_predictions)
        
        # Find misclassifications (off-diagonal elements)
        misclassifications = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    misclassifications.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': cm[i, j],
                        'percentage': (cm[i, j] / cm[i].sum()) * 100
                    })
        
        # Sort by count
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        
        print(f"\nTop {top_n} Misclassifications:")
        print("-" * 80)
        for i, misc in enumerate(misclassifications[:top_n]):
            print(f"{i+1:2d}. {misc['true_class']} -> {misc['predicted_class']}: "
                  f"{misc['count']} times ({misc['percentage']:.1f}%)")
        
        return misclassifications
    
    def calculate_confidence_statistics(self):
        """Calculate confidence statistics"""
        max_probs = np.max(self.all_probabilities, axis=1)
        correct_mask = (self.all_predictions == self.all_targets)
        
        correct_confidence = max_probs[correct_mask]
        incorrect_confidence = max_probs[~correct_mask]
        
        stats = {
            'overall_mean_confidence': np.mean(max_probs),
            'overall_std_confidence': np.std(max_probs),
            'correct_mean_confidence': np.mean(correct_confidence),
            'correct_std_confidence': np.std(correct_confidence),
            'incorrect_mean_confidence': np.mean(incorrect_confidence),
            'incorrect_std_confidence': np.std(incorrect_confidence)
        }
        
        return stats
    
    def plot_confidence_distribution(self, save_path=None):
        """Plot confidence distribution for correct vs incorrect predictions"""
        max_probs = np.max(self.all_probabilities, axis=1)
        correct_mask = (self.all_predictions == self.all_targets)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs[correct_mask], bins=50, alpha=0.7, label='Correct', color='green')
        plt.hist(max_probs[~correct_mask], bins=50, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confidence_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("=" * 80)
        print("ISL MODEL EVALUATION REPORT")
        print("=" * 80)
        
        # Basic metrics
        accuracy, avg_loss = self.evaluate()
        top3_accuracy = self.calculate_top_k_accuracy(k=3)
        top5_accuracy = self.calculate_top_k_accuracy(k=5)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        
        # Classification report
        class_report = self.generate_classification_report()
        
        print("\nPER-CLASS METRICS:")
        print("-" * 50)
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"{class_name:20s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Overall metrics
        print(f"\nOVERALL METRICS:")
        print("-" * 50)
        macro_avg = class_report['macro avg']
        weighted_avg = class_report['weighted avg']
        print(f"Macro Avg    : Precision={macro_avg['precision']:.3f}, "
              f"Recall={macro_avg['recall']:.3f}, F1={macro_avg['f1-score']:.3f}")
        print(f"Weighted Avg : Precision={weighted_avg['precision']:.3f}, "
              f"Recall={weighted_avg['recall']:.3f}, F1={weighted_avg['f1-score']:.3f}")
        
        # Confidence statistics
        conf_stats = self.calculate_confidence_statistics()
        print(f"\nCONFIDENCE STATISTICS:")
        print("-" * 50)
        print(f"Overall Mean Confidence: {conf_stats['overall_mean_confidence']:.3f} ± {conf_stats['overall_std_confidence']:.3f}")
        print(f"Correct Mean Confidence: {conf_stats['correct_mean_confidence']:.3f} ± {conf_stats['correct_std_confidence']:.3f}")
        print(f"Incorrect Mean Confidence: {conf_stats['incorrect_mean_confidence']:.3f} ± {conf_stats['incorrect_std_confidence']:.3f}")
        
        # Generate plots
        print(f"\nGenerating visualizations...")
        cm = self.plot_confusion_matrix()
        class_acc = self.plot_class_accuracy()
        self.plot_confidence_distribution()
        
        # Analyze misclassifications
        misclassifications = self.analyze_misclassifications()
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'test_accuracy': float(accuracy),
            'test_loss': float(avg_loss),
            'top3_accuracy': float(top3_accuracy),
            'top5_accuracy': float(top5_accuracy),
            'classification_report': class_report,
            'confidence_statistics': conf_stats,
            'confusion_matrix': cm.tolist(),
            'class_accuracy': class_acc.tolist(),
            'top_misclassifications': misclassifications[:20]
        }
        
        report_path = os.path.join(self.save_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        print("=" * 80)
        
        return report_data