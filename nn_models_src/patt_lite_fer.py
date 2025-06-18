import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import timm
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import json
import warnings
import math
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import time

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FERDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((
                            os.path.join(class_path, img_name),
                            self.class_to_idx[class_name]
                        ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L').convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImprovedPatchExtraction(nn.Module):
    """Enhanced patch extraction with better feature learning"""
    def __init__(self, in_channels=512, out_channels=256):
        super(ImprovedPatchExtraction, self).__init__()
        
        # Enhanced depthwise separable convolutions
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=4, 
                                   stride=2, padding=1, groups=in_channels, bias=False)
        self.pointwise1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.depthwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=2, 
                                   stride=1, padding=0, groups=out_channels, bias=False)
        self.pointwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Enhanced final convolution
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation for better feature recalibration
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU6(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        # Input: 512 x 14 x 14, pad to 16x16
        if x.size(-1) == 14:
            x = F.pad(x, (1, 1, 1, 1))  # Pad to 16x16
        
        # First depthwise separable: 16x16 -> 8x8
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second depthwise separable: 8x8 -> 7x7
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Fixed size pooling for ONNX compatibility
        x = F.avg_pool2d(x, kernel_size=x.size(-1), stride=1)  # Pool to 1x1
        x = F.interpolate(x, size=(2, 2), mode='bilinear', align_corners=False)  # Resize to 2x2
        
        # SE attention
        se_weights = self.se(x)
        x = x * se_weights
        
        # Final pointwise
        x = self.final_conv(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        return x

class MultiHeadAttentionClassifier(nn.Module):
    """Enhanced multi-head attention classifier"""
    def __init__(self, input_dim=256, hidden_dim=128, num_heads=4, num_classes=7):
        super(MultiHeadAttentionClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Input projection
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        # Multi-head attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout_attn = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input projection
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Prepare for multi-head attention
        residual = x
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Multi-head attention
        q = self.query_proj(x).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.key_proj(x).view(batch_size, 1, self.num_heads, self.head_dim)
        v = self.value_proj(x).view(batch_size, 1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)
        
        # Output projection and residual connection
        attn_output = self.out_proj(attn_output.squeeze(1))
        x = self.layer_norm(residual + attn_output)
        
        # Final classification
        output = self.classifier(x)
        
        return output

class PAttLiteEnhanced(nn.Module):
    """Enhanced PAtt-Lite model"""
    def __init__(self, num_classes=7, pretrained=True):
        super(PAttLiteEnhanced, self).__init__()
        
        # Load MobileNetV1 backbone
        if pretrained:
            print("Loading MobileNetV1 pretrained weights...")
            self.backbone = timm.create_model('mobilenetv1_100', pretrained=True, features_only=True)
            print("✓ MobileNetV1 loaded successfully")
        else:
            self.backbone = timm.create_model('mobilenetv1_100', pretrained=False, features_only=True)
        
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Enhanced patch extraction
        self.patch_extraction = ImprovedPatchExtraction(in_channels=512, out_channels=256)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Enhanced attention classifier
        self.attention_classifier = MultiHeadAttentionClassifier(
            input_dim=256, 
            hidden_dim=128, 
            num_heads=4,
            num_classes=num_classes
        )
        
    def forward(self, x):
        # Extract features using MobileNetV1
        features = self.backbone(x)
        
        # Use stage 3 (512 channels, 14x14)
        x = features[3]
        
        # Apply enhanced patch extraction
        x = self.patch_extraction(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Enhanced attention classification
        x = self.attention_classifier(x)
        
        return x
    
    def unfreeze_layers(self, num_layers=60):
        """Unfreeze specified number of layers for fine-tuning"""
        all_params = list(self.backbone.parameters())
        total_layers = len(all_params)
        layers_to_unfreeze = min(num_layers, total_layers)
        
        # Unfreeze from the end
        for param in all_params[-layers_to_unfreeze:]:
            param.requires_grad = True
            
        print(f"✓ Unfrozen {layers_to_unfreeze}/{total_layers} backbone layers")

def calculate_class_weights(dataset):
    """Calculate class weights for handling imbalance"""
    labels = [sample[1] for sample in dataset.samples]
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate inverse frequency weights
    weights = {}
    for class_idx in range(7):  # 7 classes
        if class_idx in class_counts:
            weights[class_idx] = total_samples / (7 * class_counts[class_idx])
        else:
            weights[class_idx] = 1.0
    
    weight_tensor = torch.FloatTensor([weights[i] for i in range(7)])
    print(f"Class weights: {weight_tensor}")
    
    return weight_tensor

def get_enhanced_data_loaders(data_dir, batch_size=16, num_workers=4):
    """Create enhanced data loaders with better augmentation"""
    
    # Enhanced normalization
    normalize = transforms.Normalize(
        mean=[0.4873, 0.4873, 0.4873], 
        std=[0.2593, 0.2593, 0.2593]
    )
    
    # Strong training augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, fill=128),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=128),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=128),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        normalize,
    ])
    
    # Clean validation/test transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Create datasets
    train_dataset = FERDataset(data_dir, 'train', train_transform)
    val_dataset = FERDataset(data_dir, 'validation', val_transform)
    test_dataset = FERDataset(data_dir, 'test', val_transform)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    
    # Create weighted sampler for balanced training
    labels = [sample[1] for sample in train_dataset.samples]
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_weights

def warmup_cosine_schedule(optimizer, epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    """Warmup + Cosine annealing learning rate schedule"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        )
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def train_epoch_enhanced(model, train_loader, criterion, optimizer, device, epoch):
    """Enhanced training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx:4d}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_comprehensive(model, val_loader, criterion, device, classes):
    """Comprehensive validation with detailed metrics"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.cuda.amp.autocast():
                output = model(data)
                val_loss += criterion(output, target).item()
            
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    
    print(f"\nPer-class metrics:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name:>10}: Precision: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1: {f1[i]:.3f}")
    
    return val_loss, val_acc, all_preds, all_targets, all_probs

def train_patt_lite_enhanced():
    """Enhanced training pipeline with proper stage continuity"""
    
    # Configuration
    DATA_DIR = "dataset"
    BATCH_SIZE = 16
    EPOCHS_STAGE1 = 40
    EPOCHS_STAGE2 = 40
    LR_BASE = 1e-3
    LR_FINE = 1e-4
    WARMUP_EPOCHS = 5
    
    print("🚀 Enhanced PAtt-Lite Training Started")
    print(f"📱 Device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
    
    # Setup tensorboard logging
    writer = SummaryWriter('runs/patt_lite_enhanced')
    
    # Load data with enhanced augmentation
    train_loader, val_loader, test_loader, class_weights = get_enhanced_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # Initialize model
    model = PAttLiteEnhanced(num_classes=7, pretrained=True).to(device)
    
    # Enhanced loss function with label smoothing and class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'lr': [], 'best_val_acc': 0.0
    }
    
    classes = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    
    print("\n" + "="*60)
    print("🎯 STAGE 1: Training new components")
    print("="*60)
    # Stage 1: Train new components only
    optimizer = optim.AdamW([
        {'params': model.patch_extraction.parameters()},
        {'params': model.attention_classifier.parameters()}
    ], lr=LR_BASE, weight_decay=1e-4, betas=(0.9, 0.999))
    
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(EPOCHS_STAGE1):
        print(f'\n📅 Epoch {epoch+1}/{EPOCHS_STAGE1}')
        print('-' * 40)
        
        # Dynamic learning rate with warmup
        current_lr = warmup_cosine_schedule(optimizer, epoch, WARMUP_EPOCHS, EPOCHS_STAGE1, LR_BASE)
        
        # Training
        train_loss, train_acc = train_epoch_enhanced(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _, _ = validate_comprehensive(model, val_loader, criterion, device, classes)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Tensorboard logging
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f'📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Loss: {train_loss:.4f}/{val_loss:.4f} | LR: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            torch.save(model.state_dict(), 'patt_lite_enhanced_stage1.pth')
            patience_counter = 0
            print(f'💾 New best model saved! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'⏰ Early stopping at epoch {epoch+1}')
            break
    
    print(f'✅ Stage 1 completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    print("\n" + "="*60)

    print("🔥 STAGE 2: Fine-tuning entire network")
    print("="*60)
    
    # Load best stage 1 model
    model.load_state_dict(torch.load('patt_lite_enhanced_stage1.pth'))
    print("✅ Stage 1 weights loaded successfully")
    
    # Unfreeze backbone layers
    model.unfreeze_layers(num_layers=60)
    
    # New optimizer for all trainable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR_FINE, 
        weight_decay=1e-4
    )
    
    # Reset best accuracy for stage 2
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS_STAGE2):
        print(f'\n📅 Epoch {epoch+1}/{EPOCHS_STAGE2}')
        print('-' * 40)
        
        # Dynamic learning rate
        current_lr = warmup_cosine_schedule(optimizer, epoch, 3, EPOCHS_STAGE2, LR_FINE, min_lr=1e-6)
        
        # Training
        train_loss, train_acc = train_epoch_enhanced(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _, _ = validate_comprehensive(model, val_loader, criterion, device, classes)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Tensorboard logging
        writer.add_scalar('Stage2_Train/Loss', train_loss, epoch + EPOCHS_STAGE1)
        writer.add_scalar('Stage2_Train/Accuracy', train_acc, epoch + EPOCHS_STAGE1)
        writer.add_scalar('Stage2_Val/Loss', val_loss, epoch + EPOCHS_STAGE1)
        writer.add_scalar('Stage2_Val/Accuracy', val_acc, epoch + EPOCHS_STAGE1)
        
        print(f'📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Loss: {train_loss:.4f}/{val_loss:.4f} | LR: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            torch.save(model.state_dict(), 'patt_lite_enhanced_final.pth')
            patience_counter = 0
            print(f'💾 New best model saved! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'⏰ Early stopping at epoch {epoch+1}')
            break
    
    # Load best final model
    model.load_state_dict(torch.load('patt_lite_enhanced_final.pth'))
    print(f'✅ Stage 2 completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    print("\n" + "="*60)
    print("🎯 FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Final evaluation
    test_loss, test_acc, test_preds, test_targets, test_probs = validate_comprehensive(
        model, test_loader, criterion, device, classes
    )
    
    print(f'\n🏆 FINAL RESULTS:')
    print(f'📈 Test Accuracy: {test_acc:.2f}%')
    print(f'📉 Test Loss: {test_loss:.4f}')
    
    # Detailed evaluation and plots
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar_kws={'shrink': 0.8})
    plt.title(f'Enhanced PAtt-Lite Confusion Matrix\nTest Accuracy: {test_acc:.2f}%', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    report = classification_report(test_targets, test_preds, target_names=classes, output_dict=True)
    print("\n📋 Detailed Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=classes))
    
    # Per-class accuracy
    print("\n🎯 Per-Class Accuracy:")
    for i, class_name in enumerate(classes):
        class_mask = np.array(test_targets) == i
        if class_mask.sum() > 0:
            class_preds = np.array(test_preds)[class_mask]
            class_acc = (class_preds == i).mean() * 100
            class_count = class_mask.sum()
            print(f'  {class_name:>10}: {class_acc:6.2f}% ({class_count:4d} samples)')
    
    # Export to ONNX (fixed version)
    print("\n🔄 Exporting to ONNX...")
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model, 
            dummy_input, 
            'patt_lite_enhanced.onnx',
            export_params=True, 
            opset_version=11, 
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )
        print("✅ ONNX export successful: 'patt_lite_enhanced.onnx'")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
    
    # Training curves
    plt.figure(figsize=(20, 8))
    
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 2)
    plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
    plt.title('Training & Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 3)
    plt.plot(history['lr'], label='Learning Rate', linewidth=2, color='red')
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 4, 4)
    per_class_f1 = [report[cls]['f1-score'] for cls in classes]
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    bars = plt.bar(classes, per_class_f1, color=colors)
    plt.title('Per-Class F1-Score', fontsize=14)
    plt.xlabel('Expression Class')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, f1 in zip(bars, per_class_f1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comprehensive results
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'training_history': history,
        'classification_report': report,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'class_weights': class_weights.tolist()
    }
    
    with open('patt_lite_enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    writer.close()
    
    print(f'\n💾 Files saved:')
    print(f'  🏆 patt_lite_enhanced_final.pth (trained model)')
    print(f'  📊 patt_lite_enhanced_results.json (detailed results)')
    print(f'  📈 enhanced_training_results.png (training curves)')
    print(f'  🔍 enhanced_confusion_matrix.png (confusion matrix)')
    print(f'  🔄 patt_lite_enhanced.onnx (ONNX export)')
    
    print(f'\n🎉 Enhanced training completed successfully!')
    print(f'🏆 Final Test Accuracy: {test_acc:.2f}%')
    
    return model, results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Run enhanced training
    trained_model, results = train_patt_lite_enhanced()